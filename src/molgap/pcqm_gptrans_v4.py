"""Reference-screen V4 runner for the fixed PCQM-100K GPTrans-T baseline."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .screen_policy import (
    REFERENCE_MATCH_FIELDS,
    REFERENCE_PROVENANCE_FIELDS,
    canonical_fingerprint,
    validate_runtime_certificate,
    validate_screen_arm,
)
from .training_reproducibility import (
    assert_finite_state_dict,
    atomic_json,
    atomic_torch_save,
    build_runtime_manifest,
    capture_rng_state,
    configure_fp32_determinism,
    restore_rng_state,
    sha256_file,
)


SEED = 42
PHYSICAL_BATCH = 128
TRAIN_ROWS = 100_000
DEVELOPMENT_ROWS = 50_000
BATCHES_PER_EPOCH = TRAIN_ROWS // PHYSICAL_BATCH
TAIL_ROWS_PER_EPOCH = TRAIN_ROWS % PHYSICAL_BATCH
EPOCHS = 60
WARMUP_EPOCHS = 4
SAMPLE_PRESENTATIONS = BATCHES_PER_EPOCH * PHYSICAL_BATCH * EPOCHS
LEARNING_RATE = 1.0e-3
MIN_LEARNING_RATE = 1.0e-6
WEIGHT_DECAY = 0.05
GRADIENT_CLIP = 1.0
EMA_DECAY = 0.9999
LOADER_WORKERS = 4
EXPECTED_PARAMETERS = 5_246_817
EXPECTED_INITIAL_MODEL_SHA256 = (
    "8988db8659c6c7e2b27401312f43684215c34cd8d69309aed7ce946ee9cb1ec6"
)
EXPECTED_INITIAL_STATE_ARTIFACT_SHA256 = (
    "073fce25752f9fc5e15670177cd9e69286d681985a3f3e6bed757efef00b124a"
)
EXPECTED_ARCHITECTURE_SHA256 = (
    "602d79eaf77f0623ec2b68057bd48793b0132b594823748b0aa84633fb24cb60"
)
MANIFEST_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
GEOMETRY_AGGREGATE_SHA256 = (
    "bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5"
)
KAGGLE_SHARD_LIST_SHA256 = (
    "fe691f64c64d929829ae4c0ba489a2c5cabff6ac420ee54578c0352bb95d14ad"
)
OFFICIAL_ROW_MANIFEST_SHA256 = (
    "c329fbde935324118f4f62aa6e2b01b209d6f54642549637853419deb1761c4b"
)
STOCHASTICITY_FLOOR_EV = 0.003
MINIMUM_MATERIAL_GAIN_EV = 0.003
PREFLIGHT_WARMUP_STEPS = 5
PREFLIGHT_MEASURED_STEPS = 30
MAX_ESTIMATED_TRAIN_HOURS = 6.0
FINITE_CHECK_EVERY_STEPS = 50
RUN_FORMAT = "molgap-pcqm-gptrans-t-100k-reference-v4"
CHECKPOINT_FORMAT = "molgap-pcqm-gptrans-t-100k-checkpoint-v4"


@dataclass(frozen=True)
class FixedAssets:
    manifest: dict
    train_paths: tuple[Path, ...]
    development_paths: tuple[Path, ...]


def _aggregate(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in records:
        digest.update(
            f"{item['role']}\t{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    return digest.hexdigest()


def validate_fixed_assets(
    dataset_root: Path, manifest_path: Path, *, verify_content: bool
) -> FixedAssets:
    """Accept only the published 100K/50K graph identity."""
    dataset_root = dataset_root.resolve()
    manifest_path = manifest_path.resolve()
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("Fixed 100K manifest byte identity changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "molgap-pcqm4mv2-kaggle-fixed-subset-v1":
        raise RuntimeError("Fixed 100K manifest format changed")
    if manifest.get("status") != "complete":
        raise RuntimeError("Fixed 100K manifest is incomplete")
    if manifest.get("identity") != {
        "name": "ogb-train-100k",
        "train_rows": TRAIN_ROWS,
        "development_rows": DEVELOPMENT_ROWS,
        "kaggle1": True,
        "scnet_compatible": False,
    }:
        raise RuntimeError("Fixed 100K identity changed")
    if manifest.get("roles") != {
        "train": {"source_idx_start": 0, "source_idx_stop": 100_000, "rows": 100_000},
        "development": {
            "source_idx_start": 100_000,
            "source_idx_stop": 150_000,
            "rows": 50_000,
        },
    }:
        raise RuntimeError("Fixed 100K role boundaries changed")
    source = manifest.get("source", {})
    if source.get("official_row_manifest_sha256") != OFFICIAL_ROW_MANIFEST_SHA256:
        raise RuntimeError("Official row identity changed")
    if source.get("external_data_used") is not False:
        raise RuntimeError("External data entered the baseline")
    graph_contract = manifest.get("graph_contract", {})
    expected_graph_contract = {
        "feature_schema": "ogb",
        "node_feature_dim": 9,
        "edge_feature_dim": 3,
        "rwse_dim": 16,
        "geometry_method": "ETKDGv3",
        "optimization_method": "MMFF94s",
    }
    if graph_contract != expected_graph_contract:
        raise RuntimeError("Fixed graph contract changed")
    for key in (
        "official_validation_role_read",
        "test_dev_role_read",
        "test_challenge_role_read",
    ):
        if manifest.get(key) is not False:
            raise RuntimeError(f"Sealed role changed: {key}")

    if manifest.get("geometry_aggregate_sha256") != GEOMETRY_AGGREGATE_SHA256:
        raise RuntimeError("Fixed parent geometry aggregate changed")
    records = manifest.get("geometry_shards", [])
    if len(records) != 3 or _aggregate(records) != KAGGLE_SHARD_LIST_SHA256:
        raise RuntimeError("Fixed 100K shard aggregate changed")
    train_paths: list[Path] = []
    development_paths: list[Path] = []
    expected = [("train", 0, 49_999), ("train", 50_000, 99_999), ("development", 100_000, 149_999)]
    for item, (role, start, stop) in zip(records, expected, strict=True):
        if (
            item.get("role") != role
            or int(item.get("source_idx_min", -1)) != start
            or int(item.get("source_idx_max", -1)) != stop
        ):
            raise RuntimeError("Fixed 100K shard role or row order changed")
        path = (dataset_root / item["file"]).resolve()
        try:
            path.relative_to(dataset_root)
        except ValueError as error:
            raise RuntimeError(f"Shard path escaped dataset root: {path}") from error
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"Fixed shard missing or resized: {path.name}")
        if verify_content and sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Fixed shard content changed: {path.name}")
        (train_paths if role == "train" else development_paths).append(path)
    return FixedAssets(manifest, tuple(train_paths), tuple(development_paths))


class DeterministicEpochBatchSampler:
    def __init__(self, dataset_size: int, epoch: int) -> None:
        import torch

        if dataset_size != TRAIN_ROWS:
            raise ValueError("V4 sampler requires exactly 100K training rows")
        generator = torch.Generator().manual_seed(SEED + int(epoch))
        self.indices = torch.randperm(dataset_size, generator=generator)[
            : BATCHES_PER_EPOCH * PHYSICAL_BATCH
        ]

    def __iter__(self):
        for batch_index in range(BATCHES_PER_EPOCH):
            left = batch_index * PHYSICAL_BATCH
            yield self.indices[left : left + PHYSICAL_BATCH].tolist()

    def __len__(self) -> int:
        return BATCHES_PER_EPOCH


def _load_datasets(paths: tuple[Path, ...]):
    import torch
    from torch.utils.data import ConcatDataset
    from torch_geometric.data import InMemoryDataset

    class PackedGraphs(InMemoryDataset):
        def __init__(self, path: Path):
            super().__init__(root=None)
            self.data, self.slices = torch.load(path, map_location="cpu", weights_only=False)

    shards = [PackedGraphs(path) for path in paths]
    return ConcatDataset(shards), shards


def _training_loader(graphs, epoch: int):
    import torch
    from torch_geometric.loader import DataLoader

    return DataLoader(
        graphs,
        batch_sampler=DeterministicEpochBatchSampler(len(graphs), epoch),
        num_workers=LOADER_WORKERS,
        persistent_workers=LOADER_WORKERS > 0,
        pin_memory=True,
        prefetch_factor=2 if LOADER_WORKERS > 0 else None,
        generator=torch.Generator().manual_seed(SEED + epoch),
    )


def _development_loader(graphs):
    from torch_geometric.loader import DataLoader

    return DataLoader(graphs, batch_size=PHYSICAL_BATCH, shuffle=False, num_workers=0)


def _target_stats(shards) -> tuple[float, float]:
    import torch

    values = torch.cat([shard._data.y.view(-1).double() for shard in shards])
    if values.numel() != TRAIN_ROWS or not bool(torch.isfinite(values).all()):
        raise RuntimeError("Training targets are incomplete or non-finite")
    # SCNet's frozen PyTorch predates the ``correction`` keyword overload.
    return float(values.mean()), float(values.std(unbiased=True).clamp_min(1e-6))


def _make_model(initial_state_path: Path | None = None):
    import torch

    from .gptrans import OGBGPTransTiny

    model = OGBGPTransTiny(
        node_channels=256,
        pair_channels=32,
        num_layers=12,
        num_heads=8,
        shortest_path_cap=20,
        dropout=0.1,
        drop_path=0.1,
        layer_scale=1.0,
        n_targets=1,
    )
    if initial_state_path is None:
        return model
    if sha256_file(initial_state_path) != EXPECTED_INITIAL_STATE_ARTIFACT_SHA256:
        raise RuntimeError("Frozen GPTrans-T initial-state artifact changed")
    payload = torch.load(initial_state_path, map_location="cpu")
    if payload.get("format") != "molgap-gptrans-t-seed42-initial-state-v1":
        raise RuntimeError("Frozen GPTrans-T initial-state format changed")
    model.load_state_dict(payload["model_state"], strict=True)
    if payload.get("state_sha256") != _state_sha256(model):
        raise RuntimeError("Frozen GPTrans-T initial-state payload is inconsistent")
    return model


def _state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _batch_sha256(batch) -> str:
    digest = hashlib.sha256()
    for name in ("x", "edge_index", "edge_attr", "y", "batch"):
        value = getattr(batch, name).detach().cpu().contiguous()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _verify_model_identity(model) -> tuple[int, str]:
    architecture_path = Path(__file__).with_name("gptrans.py")
    architecture_sha256 = sha256_file(architecture_path)
    if architecture_sha256 != EXPECTED_ARCHITECTURE_SHA256:
        raise RuntimeError("Frozen GPTrans-T source changed")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Frozen GPTrans-T parameter count changed: {parameters}")
    initial_sha256 = _state_sha256(model)
    if initial_sha256 != EXPECTED_INITIAL_MODEL_SHA256:
        raise RuntimeError(
            "Frozen GPTrans-T seed-42 initialization changed: "
            f"observed={initial_sha256} expected={EXPECTED_INITIAL_MODEL_SHA256}"
        )
    return parameters, architecture_sha256


def _forward(model, batch):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        getattr(batch, "random_walk_pe", None),
    ).view(-1)


class ExponentialMovingAverage:
    def __init__(self, model) -> None:
        self.state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def update(self, model) -> None:
        for name, value in model.state_dict().items():
            target = self.state[name]
            if value.is_floating_point():
                target.mul_(EMA_DECAY).add_(value.detach(), alpha=1.0 - EMA_DECAY)
            else:
                target.copy_(value)

    def state_dict(self):
        return self.state

    def load_state_dict(self, state) -> None:
        if state.keys() != self.state.keys():
            raise RuntimeError("EMA state identity changed")
        self.state = {name: value.clone() for name, value in state.items()}


class FrozenEpochScheduler:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.epoch = -1

    @staticmethod
    def learning_rate(epoch: int) -> float:
        if epoch < WARMUP_EPOCHS:
            return LEARNING_RATE * float(epoch + 1) / WARMUP_EPOCHS
        progress = float(epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * cosine

    def step(self, epoch: int) -> float:
        self.epoch = int(epoch)
        learning_rate = self.learning_rate(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def state_dict(self) -> dict:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        self.epoch = int(state["epoch"])


def _make_training_state(initial_state_path: Path):
    import torch

    model = _make_model(initial_state_path).to("cuda")
    _verify_model_identity(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
    )
    scheduler = FrozenEpochScheduler(optimizer)
    ema = ExponentialMovingAverage(model)
    return model, optimizer, scheduler, ema


def _optimizer_step(model, optimizer, ema, batch, mean, std, *, check_finite: bool):
    import torch
    import torch.nn.functional as functional

    optimizer.zero_grad(set_to_none=True)
    prediction = _forward(model, batch)
    target = (batch.y.view(-1).float() - mean) / std
    loss = functional.l1_loss(prediction, target)
    if check_finite and not bool(torch.isfinite(loss)):
        raise RuntimeError("Training loss became non-finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    if check_finite and not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("Gradient norm became non-finite")
    optimizer.step()
    ema.update(model)
    return loss.detach()


def _evaluate(model, ema, graphs, mean, std) -> dict:
    import torch

    live_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(ema.state_dict(), strict=True)
    model.eval()
    predictions = []
    targets = []
    source_indices = []
    with torch.no_grad():
        for batch in _development_loader(graphs):
            batch = batch.to("cuda", non_blocking=True)
            predictions.append((_forward(model, batch) * std + mean).cpu())
            targets.append(batch.y.view(-1).float().cpu())
            source_idx = getattr(batch, "source_idx", getattr(batch, "row_index", None))
            if source_idx is None:
                raise RuntimeError("Development graphs have no source identity")
            source_indices.append(source_idx.view(-1).long().cpu())
    model.load_state_dict(live_state, strict=True)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    source_idx = torch.cat(source_indices)
    if prediction.numel() != DEVELOPMENT_ROWS:
        raise RuntimeError("Development prediction count changed")
    expected = torch.arange(100_000, 150_000, dtype=torch.long)
    if not torch.equal(source_idx, expected):
        raise RuntimeError("Development source order changed")
    return {
        "mae_eV": float((prediction - target).abs().mean()),
        "prediction_eV": prediction,
        "target_eV": target,
        "source_idx": source_idx,
    }


def _scientific_fields() -> dict:
    return {
        "benchmark_id": "ogb-lsc-pcqm4mv2-gap-internal-100k-v4",
        "data_role_fingerprint": canonical_fingerprint(
            {"manifest_sha256": MANIFEST_SHA256, "train": [0, 100_000], "development": [100_000, 150_000]}
        ),
        "row_order_fingerprint": canonical_fingerprint(
            {"algorithm": "seed-plus-epoch-global-randperm-v1", "seed": SEED, "epochs": EPOCHS, "drop_last": TAIL_ROWS_PER_EPOCH}
        ),
        "feature_fingerprint": canonical_fingerprint(
            {"payload": "ogb-geometry-v1", "used": ["atom9", "bond3", "shortest-path-cap20"], "geometry_used": False}
        ),
        "target_fingerprint": "pcqm4mv2-gap-eV-direct",
        "seed": SEED,
        "precision": "fp32",
        "optimizer_fingerprint": canonical_fingerprint(
            {"name": "torch-adamw", "foreach": False, "fused": False, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}
        ),
        "schedule_fingerprint": canonical_fingerprint(
            {"name": "epoch-warmup-cosine-v1", "epochs": EPOCHS, "warmup_epochs": WARMUP_EPOCHS, "minimum_lr": MIN_LEARNING_RATE}
        ),
        "loss_fingerprint": "normalized-gap-l1",
        "target_transform_fingerprint": canonical_fingerprint(
            {"source": "fixed-train-100k", "mean": "population-mean", "std": "sample-std-correction1"}
        ),
        "selection_fingerprint": "best-development-ema9999-60epochs",
        "role_access_fingerprint": canonical_fingerprint(
            {"train": True, "development": True, "official_validation": False, "test_dev": False, "test_challenge": False}
        ),
        "sample_exposure": SAMPLE_PRESENTATIONS,
        "tail_batch_policy": "drop_last",
    }


def _gpu_utilization_percent() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "--id=0"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(result.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def validate_source_archive(
    source_archive: Path, source_archive_sha256: str, source_commit: str
) -> str:
    """Bind the extracted runtime to an immutable tracked-file archive."""
    source_archive = source_archive.resolve()
    if not source_archive.is_file():
        raise FileNotFoundError(source_archive)
    if sha256_file(source_archive) != source_archive_sha256:
        raise RuntimeError("Source archive identity changed")
    commit_path = source_archive.with_name("SOURCE_COMMIT.txt")
    sha_path = source_archive.with_name("SOURCE_ARCHIVE_SHA256.txt")
    inventory_path = source_archive.with_name("SOURCE_FILES.json")
    for path in (commit_path, sha_path, inventory_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if commit_path.read_text(encoding="utf-8").strip() != source_commit:
        raise RuntimeError("Source commit sidecar changed")
    if sha_path.read_text(encoding="utf-8").strip() != source_archive_sha256:
        raise RuntimeError("Source archive SHA sidecar changed")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    expected = {item["path"]: item["sha256"] for item in inventory["files"]}
    observed: dict[str, str] = {}
    with tarfile.open(source_archive, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"Cannot read source member: {member.name}")
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            observed[member.name] = digest.hexdigest()
    if observed != expected:
        raise RuntimeError("Source archive file inventory changed")
    return source_archive_sha256


def run_preflight(
    *,
    dataset_root: Path,
    manifest_path: Path,
    source_archive: Path,
    source_archive_sha256: str,
    source_commit: str,
    output: Path,
    platform_id: str,
    initial_state_path: Path,
) -> dict:
    determinism = configure_fp32_determinism(SEED)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V4 preflight requires exactly one visible accelerator")
    validate_source_archive(source_archive, source_archive_sha256, source_commit)
    validate_screen_arm(physical_batch_per_device=PHYSICAL_BATCH)
    assets = validate_fixed_assets(dataset_root, manifest_path, verify_content=True)
    runtime = build_runtime_manifest(determinism)
    train_graphs, train_shards = _load_datasets(assets.train_paths)
    if len(train_graphs) != TRAIN_ROWS:
        raise RuntimeError("Loaded training row count changed")
    mean_value, std_value = _target_stats(train_shards)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    batch = next(iter(_training_loader(train_graphs, 0))).to("cuda", non_blocking=True)
    if int(batch.num_graphs) != PHYSICAL_BATCH:
        raise RuntimeError("Preflight did not receive physical batch 128")
    fixture_sha256 = _batch_sha256(batch)

    repeat_hashes = []
    repeat_losses = []
    for _ in range(2):
        configure_fp32_determinism(SEED)
        model, optimizer, scheduler, ema = _make_training_state(initial_state_path)
        scheduler.step(0)
        repeat_losses.append(
            float(
                _optimizer_step(
                    model, optimizer, ema, batch, mean, std, check_finite=True
                ).cpu()
            )
        )
        torch.cuda.synchronize()
        repeat_hashes.append(_state_sha256(model))
        del model, optimizer, scheduler, ema
        torch.cuda.empty_cache()
    if repeat_hashes[0] != repeat_hashes[1] or repeat_losses[0] != repeat_losses[1]:
        raise RuntimeError("Seeded optimizer-step calibration is not deterministic")

    configure_fp32_determinism(SEED)
    model, optimizer, scheduler, ema = _make_training_state(initial_state_path)
    scheduler.step(0)
    batches = iter(_training_loader(train_graphs, 0))
    torch.cuda.reset_peak_memory_stats()
    for _ in range(PREFLIGHT_WARMUP_STEPS):
        _optimizer_step(
            model,
            optimizer,
            ema,
            next(batches).to("cuda", non_blocking=True),
            mean,
            std,
            check_finite=True,
        )
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(PREFLIGHT_MEASURED_STEPS):
        _optimizer_step(
            model,
            optimizer,
            ema,
            next(batches).to("cuda", non_blocking=True),
            mean,
            std,
            check_finite=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    reserve = 1.0 - peak_reserved / total_memory
    graphs_per_second = PREFLIGHT_MEASURED_STEPS * PHYSICAL_BATCH / elapsed
    estimated_hours = SAMPLE_PRESENTATIONS / graphs_per_second / 3600.0
    if reserve < 0.15 or estimated_hours > MAX_ESTIMATED_TRAIN_HOURS:
        raise RuntimeError(f"V4 preflight failed: reserve={reserve:.3f}, estimate={estimated_hours:.2f}h")

    certificate = {
        "format": "molgap-runtime-certificate-v1",
        "status": "accepted",
        "platform_id": platform_id,
        "accelerator": torch.cuda.get_device_name(0),
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": PHYSICAL_BATCH,
        "tail_batch_policy": "drop_last",
        "software_fingerprint": runtime["installed_distributions_sha256"],
        "determinism_fingerprint": canonical_fingerprint(determinism),
        "calibration_fixture_sha256": fixture_sha256,
        "calibration_output_sha256": repeat_hashes[0],
        "calibration_checks_passed": True,
        "runtime_fingerprint": runtime["runtime_fingerprint"],
    }
    certificate_id = canonical_fingerprint(certificate)
    provisional_contract = {
        **_scientific_fields(),
        "platform_id": platform_id,
        "accelerator": certificate["accelerator"],
        "runtime_certificate_id": certificate_id,
    }
    validate_runtime_certificate(certificate, provisional_contract)
    result = {
        "format": "molgap-pcqm-gptrans-t-100k-preflight-v4",
        "accepted": True,
        "runtime_certificate_id": certificate_id,
        "runtime_certificate": certificate,
        "runtime_manifest": runtime,
        "target_stats": {"mean_eV": mean_value, "sample_std_eV": std_value},
        "optimizer_step_calibration": {
            "warmup_steps": PREFLIGHT_WARMUP_STEPS,
            "measured_steps": PREFLIGHT_MEASURED_STEPS,
            "graphs_per_second": graphs_per_second,
            "elapsed_seconds": elapsed,
            "estimated_training_hours": estimated_hours,
            "maximum_estimated_training_hours": MAX_ESTIMATED_TRAIN_HOURS,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "total_memory_bytes": total_memory,
            "memory_reserve_fraction_from_reserved": reserve,
            "gpu_utilization_percent": _gpu_utilization_percent(),
        },
        "manifest_sha256": MANIFEST_SHA256,
        "source_archive_sha256": source_archive_sha256,
        "source_commit": source_commit,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "runtime_manifest.json", runtime)
    atomic_json(output / "runtime_certificate.json", certificate)
    atomic_json(output / "preflight.json", result)
    return result


def _save_checkpoint(path: Path, *, epoch: int, model, optimizer, scheduler, ema, trace, best, best_epoch, target_stats, runtime_certificate_id, source_archive_sha256) -> None:
    atomic_torch_save(
        path,
        {
            "format": CHECKPOINT_FORMAT,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "trace": trace,
            "best_development_mae_eV": best,
            "best_epoch": best_epoch,
            "target_stats": target_stats,
            "runtime_certificate_id": runtime_certificate_id,
            "source_archive_sha256": source_archive_sha256,
            "rng_state": capture_rng_state(),
            "scientific_fields": _scientific_fields(),
        },
    )


def run_training(
    *,
    dataset_root: Path,
    manifest_path: Path,
    preflight_path: Path,
    source_archive: Path,
    source_archive_sha256: str,
    source_commit: str,
    output: Path,
    platform_id: str,
    initial_state_path: Path,
) -> dict:
    determinism = configure_fp32_determinism(SEED)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V4 training requires exactly one visible accelerator")
    output.mkdir(parents=True, exist_ok=True)
    completion_path = output / "completion_manifest.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("complete") is True:
            return completion
        raise RuntimeError("Existing completion manifest is incompatible")
    validate_source_archive(source_archive, source_archive_sha256, source_commit)
    assets = validate_fixed_assets(dataset_root, manifest_path, verify_content=True)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("accepted") is not True:
        raise RuntimeError("V4 preflight was not accepted")
    if preflight.get("source_archive_sha256") != source_archive_sha256:
        raise RuntimeError("Preflight source archive changed")
    if preflight.get("source_commit") != source_commit:
        raise RuntimeError("Preflight source commit changed")
    certificate = preflight["runtime_certificate"]
    certificate_id = preflight["runtime_certificate_id"]
    runtime = build_runtime_manifest(determinism)
    if runtime["runtime_fingerprint"] != certificate["runtime_fingerprint"]:
        raise RuntimeError("Training runtime differs from certified runtime")
    provisional_contract = {
        **_scientific_fields(),
        "platform_id": platform_id,
        "accelerator": certificate["accelerator"],
        "runtime_certificate_id": certificate_id,
    }
    validate_runtime_certificate(certificate, provisional_contract)

    train_graphs, train_shards = _load_datasets(assets.train_paths)
    development_graphs, _ = _load_datasets(assets.development_paths)
    if len(train_graphs) != TRAIN_ROWS or len(development_graphs) != DEVELOPMENT_ROWS:
        raise RuntimeError("Loaded role count changed")
    mean_value, std_value = _target_stats(train_shards)
    target_stats = {"mean_eV": mean_value, "sample_std_eV": std_value}
    if target_stats != preflight["target_stats"]:
        raise RuntimeError("Target statistics differ from preflight")
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    model, optimizer, scheduler, ema = _make_training_state(initial_state_path)
    checkpoint_path = output / "last_checkpoint.pt"
    start_epoch = 0
    trace: list[dict] = []
    best = float("inf")
    best_epoch = -1
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError("Checkpoint format changed")
        if checkpoint.get("scientific_fields") != _scientific_fields():
            raise RuntimeError("Checkpoint scientific contract changed")
        if checkpoint.get("runtime_certificate_id") != certificate_id:
            raise RuntimeError("Checkpoint runtime certificate changed")
        if checkpoint.get("source_archive_sha256") != source_archive_sha256:
            raise RuntimeError("Checkpoint source identity changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        ema.load_state_dict(checkpoint["ema"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        trace = list(checkpoint["trace"])
        best = float(checkpoint["best_development_mae_eV"])
        best_epoch = int(checkpoint["best_epoch"])

    for epoch in range(start_epoch, EPOCHS):
        learning_rate = scheduler.step(epoch)
        model.train()
        train_loss_sum = torch.zeros((), device="cuda")
        train_count = 0
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(_training_loader(train_graphs, epoch)):
            if int(batch.num_graphs) != PHYSICAL_BATCH:
                raise RuntimeError("Optimizer received a non-128 batch")
            batch = batch.to("cuda", non_blocking=True)
            global_step = epoch * BATCHES_PER_EPOCH + batch_index + 1
            loss = _optimizer_step(
                model,
                optimizer,
                ema,
                batch,
                mean,
                std,
                check_finite=global_step % FINITE_CHECK_EVERY_STEPS == 0,
            )
            train_loss_sum.add_(loss * int(batch.num_graphs))
            train_count += int(batch.num_graphs)
        if train_count != BATCHES_PER_EPOCH * PHYSICAL_BATCH:
            raise RuntimeError("Epoch sample exposure changed")
        development = _evaluate(model, ema, development_graphs, mean, std)
        improved = development["mae_eV"] < best
        if improved:
            best = development["mae_eV"]
            best_epoch = epoch
            best_payload = {
                "format": RUN_FORMAT,
                "model_config": {
                    "node_channels": 256,
                    "pair_channels": 32,
                    "num_layers": 12,
                    "num_heads": 8,
                    "shortest_path_cap": 20,
                    "dropout": 0.1,
                    "drop_path": 0.1,
                    "layer_scale": 1.0,
                    "n_targets": 1,
                },
                "model": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
                "target_stats": target_stats,
                "epoch": epoch,
                "development_mae_eV": best,
                "architecture_sha256": EXPECTED_ARCHITECTURE_SHA256,
                "runtime_certificate_id": certificate_id,
                "manifest_sha256": MANIFEST_SHA256,
                "source_archive_sha256": source_archive_sha256,
                "source_commit": source_commit,
            }
            assert_finite_state_dict(best_payload["model"], label="best GPTrans-T model")
            atomic_torch_save(output / "best_model.pt", best_payload)
            atomic_torch_save(
                output / "development_predictions.pt",
                {
                    "prediction_eV": development["prediction_eV"],
                    "target_eV": development["target_eV"],
                    "source_idx": development["source_idx"],
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                    "test_challenge_role_read": False,
                },
            )
        row = {
            "epoch": epoch,
            "train_mae_eV": float(train_loss_sum.cpu()) * std_value / train_count,
            "development_mae_eV": development["mae_eV"],
            "best_development_mae_eV": best,
            "best_epoch": best_epoch,
            "learning_rate": learning_rate,
            "optimizer_steps": BATCHES_PER_EPOCH,
            "sample_presentations": train_count,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        trace.append(row)
        atomic_json(output / "trace.json", {"format": RUN_FORMAT, "rows": trace})
        _save_checkpoint(
            checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            trace=trace,
            best=best,
            best_epoch=best_epoch,
            target_stats=target_stats,
            runtime_certificate_id=certificate_id,
            source_archive_sha256=source_archive_sha256,
        )
        print(
            f"gptrans_t_100k_v4 ep{epoch:02d} train={row['train_mae_eV']:.6f} "
            f"dev={row['development_mae_eV']:.6f}eV best={best:.6f}@{best_epoch} "
            f"lr={learning_rate:.3e} {row['elapsed_seconds']:.1f}s"
            + (" *" if improved else ""),
            flush=True,
        )

    best_model_path = output / "best_model.pt"
    predictions_path = output / "development_predictions.pt"
    result_sha256 = sha256_file(best_model_path)
    reference = {
        **_scientific_fields(),
        "run_id": "gptrans-t-100k-v4-seed42",
        "model_id": "gptrans_t_core_12x256_pair32",
        "architecture_fingerprint": EXPECTED_ARCHITECTURE_SHA256,
        "source_archive_sha256": source_archive_sha256,
        "result_artifact_sha256": result_sha256,
        "platform_id": platform_id,
        "accelerator": certificate["accelerator"],
        "runtime_certificate_id": certificate_id,
        "physical_batch_per_device": PHYSICAL_BATCH,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
        "frozen_reference": True,
        "stochasticity_floor_eV": STOCHASTICITY_FLOOR_EV,
        "minimum_material_gain_eV": MINIMUM_MATERIAL_GAIN_EV,
        "best_development_mae_eV": best,
        "best_epoch": best_epoch,
    }
    missing = [field for field in (*REFERENCE_MATCH_FIELDS, *REFERENCE_PROVENANCE_FIELDS) if field not in reference]
    if missing:
        raise RuntimeError(f"Reference record is incomplete: {missing}")
    validate_runtime_certificate(certificate, reference)
    atomic_json(output / "frozen_reference.json", reference)
    completion = {
        "format": RUN_FORMAT,
        "complete": True,
        "epochs": EPOCHS,
        "optimizer_steps": BATCHES_PER_EPOCH * EPOCHS,
        "sample_presentations": SAMPLE_PRESENTATIONS,
        "best_development_mae_eV": best,
        "best_epoch": best_epoch,
        "manifest_sha256": MANIFEST_SHA256,
        "runtime_certificate_id": certificate_id,
        "source_archive_sha256": source_archive_sha256,
        "source_commit": source_commit,
        "best_model_sha256": result_sha256,
        "development_predictions_sha256": sha256_file(predictions_path),
        "frozen_reference_sha256": sha256_file(output / "frozen_reference.json"),
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(completion_path, completion)
    return completion
