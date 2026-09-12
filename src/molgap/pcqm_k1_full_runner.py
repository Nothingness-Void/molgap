"""Hardened, resumable full-role trainer for the frozen Neural-Atom K1 model."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

from .training_reproducibility import (
    CHECKPOINT_FORMAT,
    MODEL_BUNDLE_FORMAT,
    assert_finite_state_dict,
    atomic_json,
    atomic_torch_save,
    build_runtime_manifest,
    canonical_fingerprint,
    capture_rng_state,
    configure_fp32_determinism,
    restore_rng_state,
    sha256_file,
)


SEED = 42
PHYSICAL_BATCH = 128
TRAIN_ROWS = 3_378_606
REFERENCE_TRAIN_ROWS = 500_000
REFERENCE_EPOCHS = 40
SAMPLE_PRESENTATIONS = REFERENCE_TRAIN_ROWS * REFERENCE_EPOCHS
MAX_OPTIMIZER_STEPS = SAMPLE_PRESENTATIONS // PHYSICAL_BATCH
BATCHES_PER_PASS = TRAIN_ROWS // PHYSICAL_BATCH
TAIL_ROWS_PER_PASS = TRAIN_ROWS % PHYSICAL_BATCH
LEARNING_RATE = 4e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0
LOADER_WORKERS = 2
CHECKPOINT_EVERY_STEPS = 500
FINITE_CHECK_EVERY_STEPS = 50
MAX_ESTIMATED_TRAIN_HOURS = 10.5
PREFLIGHT_WARMUP_STEPS = 10
PREFLIGHT_MEASURED_STEPS = 100
EXPECTED_PARAMETERS = 3_658_817
EXPECTED_INITIAL_MODEL_SHA256 = (
    "ce878c7a6e71519e4bd25ef10e45e6c9f693687f7f8c76dbf37c66f3c5708dc3"
)
ARCHITECTURE_SOURCE_COMMIT = "36215d9539acdd75542608637ec1e2db5341d3ff"
ARCHITECTURE_FILE_SHA256 = {
    "src/molgap/gps.py": "2744e98e9f190c7f9ea9281d41487f74ff49175d45f8e86b3d011fc2b9cc5ee5",
    "src/molgap/pcqm_gap_architecture.py": "0de3c77c50531d7f49e884c4c2bd5c67a554935c1996bd0eae952c4b5434c045",
    "src/molgap/qm9_gape.py": "619d6c3719032441cbc01082904f512bbbe5370202263ce60fa428e0e1d373f3",
    "src/molgap/qm9_local_hierarchy.py": "b9d5b72c02d4ce459cf7b6666daee07e402e493fea64eca9ac08335364efd25b",
    "src/molgap/qm9_neural_atom.py": "39873ef5f171fdac660c4a06caf282ead778d908befd2bfbf4553e50558e8d49",
}
FULL_MANIFEST_CANONICAL_SHA256 = (
    "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"
)
FULL_TOPOLOGY_AGGREGATE_SHA256 = (
    "be18a9964754c8428b5c62433136bd0bafb6c75016811c79e223aec30f3dbe98"
)
OFFICIAL_ROW_MANIFEST_SHA256 = (
    "c329fbde935324118f4f62aa6e2b01b209d6f54642549637853419deb1761c4b"
)

TRAINING_CONTRACT = {
    "format": "molgap-pcqm-k1-full-training-contract-v1",
    "benchmark_id": "ogb-lsc-pcqm4mv2-gap",
    "model_id": "neural_atom_k1",
    "architecture_source_commit": ARCHITECTURE_SOURCE_COMMIT,
    "architecture_file_sha256": ARCHITECTURE_FILE_SHA256,
    "parameter_count": EXPECTED_PARAMETERS,
    "seed42_initial_model_sha256": EXPECTED_INITIAL_MODEL_SHA256,
    "input_modality": "accepted-topology-only",
    "feature_fingerprint": "ogb-atom9-bond3-rwse16-v1",
    "target_fingerprint": "pcqm4mv2-gap-eV-direct",
    "data_role": "official-train-full",
    "train_rows": TRAIN_ROWS,
    "official_row_manifest_sha256": OFFICIAL_ROW_MANIFEST_SHA256,
    "full_manifest_canonical_sha256": FULL_MANIFEST_CANONICAL_SHA256,
    "topology_aggregate_sha256": FULL_TOPOLOGY_AGGREGATE_SHA256,
    "seed": SEED,
    "precision": "fp32",
    "tf32_enabled": False,
    "deterministic_algorithms": True,
    "physical_batch_per_device": PHYSICAL_BATCH,
    "device_count": 1,
    "gradient_accumulation_steps": 1,
    "tail_batch_policy": "drop_last-global-pass",
    "rows_dropped_per_complete_pass": TAIL_ROWS_PER_PASS,
    "row_order_policy": "seed42-global-randperm-by-pass-v1",
    "optimizer": "fused-adamw",
    "learning_rate": LEARNING_RATE,
    "minimum_learning_rate": MIN_LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "gradient_clip_norm": GRADIENT_CLIP,
    "scheduler": "cosine-per-optimizer-step",
    "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
    "sample_presentations": SAMPLE_PRESENTATIONS,
    "loss": "normalized-gap-l1",
    "target_transform": "full-train-mean-sample-std",
    "selection": "fixed-final-step-no-development-or-official-validation",
    "checkpoint_every_optimizer_steps": CHECKPOINT_EVERY_STEPS,
    "finite_check_every_optimizer_steps": FINITE_CHECK_EVERY_STEPS,
    "loader_workers": LOADER_WORKERS,
}
TRAINING_CONTRACT_SHA256 = canonical_fingerprint(TRAINING_CONTRACT)


def _canonical_manifest_sha256(payload: dict) -> str:
    return canonical_fingerprint(payload)


def _aggregate(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in records:
        digest.update(
            f"{item['role']}\t{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    return digest.hexdigest()


def _architecture_source_sha256(path: Path) -> str:
    """Hash source with canonical LF endings so Windows packaging is stable."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_full_manifest(
    dataset_root: Path, manifest_path: Path, *, verify_content: bool
) -> tuple[dict, list[Path]]:
    """Validate the full pure-2D role without reading any sealed role."""
    dataset_root = dataset_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _canonical_manifest_sha256(manifest) != FULL_MANIFEST_CANONICAL_SHA256:
        raise RuntimeError("Full manifest canonical identity changed")
    expected_identity = {
        "name": "ogb-train-full",
        "train_rows": TRAIN_ROWS,
        "development_rows": 0,
        "kaggle1": False,
        "scnet_compatible": False,
    }
    if manifest.get("status") != "complete" or manifest.get("identity") != expected_identity:
        raise RuntimeError("Full data role changed")
    if manifest.get("roles") != {
        "train": {"source_idx_start": 0, "source_idx_stop": TRAIN_ROWS, "rows": TRAIN_ROWS},
        "development": None,
    }:
        raise RuntimeError("Full row boundaries changed")
    graph_contract = manifest.get("graph_contract", {})
    for key, expected in {
        "feature_schema": "ogb",
        "node_feature_dim": 9,
        "edge_feature_dim": 3,
        "rwse_dim": 16,
    }.items():
        if graph_contract.get(key) != expected:
            raise RuntimeError(f"Full graph contract changed: {key}")
    source = manifest.get("source", {})
    if source.get("official_row_manifest_sha256") != OFFICIAL_ROW_MANIFEST_SHA256:
        raise RuntimeError("Official row identity changed")
    if source.get("external_data_used") is not False:
        raise RuntimeError("External data entered the full role")
    for key in (
        "official_validation_role_read",
        "test_dev_role_read",
        "test_challenge_role_read",
    ):
        if manifest.get(key) is not False:
            raise RuntimeError(f"Sealed role changed: {key}")

    records = manifest.get("assets", {}).get("topology", [])
    if len(records) != 68 or sum(int(item["rows"]) for item in records) != TRAIN_ROWS:
        raise RuntimeError("Full topology shard inventory changed")
    if _aggregate(records) != FULL_TOPOLOGY_AGGREGATE_SHA256:
        raise RuntimeError("Full topology aggregate changed")
    cursor = 0
    paths = []
    for item in records:
        if item.get("role") != "train" or int(item["source_idx_min"]) != cursor:
            raise RuntimeError(f"Full topology rows are not contiguous at {cursor}")
        cursor = int(item["source_idx_max"]) + 1
        path = (dataset_root / item["file"]).resolve()
        try:
            path.relative_to(dataset_root)
        except ValueError as error:
            raise RuntimeError(f"Topology path escaped dataset root: {path}") from error
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"Full topology shard missing or resized: {path.name}")
        if verify_content and sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Full topology shard hash changed: {path.name}")
        paths.append(path)
    if cursor != TRAIN_ROWS:
        raise RuntimeError("Full topology source-index endpoint changed")
    return manifest, paths


class DeterministicPassBatchSampler:
    """Global random permutation with a resumable, full-batch cursor."""

    def __init__(
        self, dataset_size: int, *, pass_index: int, start_batch: int = 0
    ) -> None:
        import torch

        if dataset_size < PHYSICAL_BATCH:
            raise ValueError("Sampler requires at least one physical batch")
        self.batches_per_pass = dataset_size // PHYSICAL_BATCH
        if not 0 <= start_batch <= self.batches_per_pass:
            raise ValueError("Invalid resume batch cursor")
        generator = torch.Generator().manual_seed(SEED + int(pass_index))
        self.indices = torch.randperm(dataset_size, generator=generator)[
            : self.batches_per_pass * PHYSICAL_BATCH
        ]
        self.start_batch = int(start_batch)

    def __iter__(self):
        for batch_index in range(self.start_batch, self.batches_per_pass):
            left = batch_index * PHYSICAL_BATCH
            yield self.indices[left : left + PHYSICAL_BATCH].tolist()

    def __len__(self) -> int:
        return self.batches_per_pass - self.start_batch


def progress_from_step(global_step: int) -> tuple[int, int]:
    if not 0 <= global_step <= MAX_OPTIMIZER_STEPS:
        raise ValueError("Global step is outside the frozen schedule")
    return divmod(global_step, BATCHES_PER_PASS)


def _load_graphs(paths: list[Path]):
    import torch
    from torch.utils.data import ConcatDataset
    from torch_geometric.data import InMemoryDataset

    class PackedGraphDataset(InMemoryDataset):
        def __init__(self, path: Path):
            super().__init__(root=None)
            self.data, self.slices = torch.load(
                path, map_location="cpu", weights_only=False
            )

    shards = [PackedGraphDataset(path) for path in paths]
    graphs = ConcatDataset(shards)
    if len(graphs) != TRAIN_ROWS:
        raise RuntimeError("Loaded full-role row count changed")
    return graphs, shards


def _target_stats(shards) -> tuple[float, float]:
    import torch

    values = torch.cat([shard._data.y.view(-1).double() for shard in shards])
    if values.numel() != TRAIN_ROWS or not bool(torch.isfinite(values).all()):
        raise RuntimeError("Full-role targets are incomplete or non-finite")
    mean = float(values.mean())
    std = float(values.std(correction=1).clamp_min(1e-6))
    return mean, std


def _loader(graphs, *, pass_index: int, start_batch: int):
    import torch
    from torch_geometric.loader import DataLoader

    sampler = DeterministicPassBatchSampler(
        len(graphs), pass_index=pass_index, start_batch=start_batch
    )
    return DataLoader(
        graphs,
        batch_sampler=sampler,
        num_workers=LOADER_WORKERS,
        persistent_workers=LOADER_WORKERS > 0,
        pin_memory=True,
        prefetch_factor=2 if LOADER_WORKERS > 0 else None,
        generator=torch.Generator().manual_seed(SEED + pass_index),
    )


def _state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _batch_sha256(batch) -> str:
    digest = hashlib.sha256()
    for name in ("x", "edge_index", "edge_attr", "random_walk_pe", "y", "batch"):
        value = getattr(batch, name).detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _gpu_utilization_percent() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(result.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _make_model_and_optimizer():
    import torch

    from .qm9_neural_atom import make_encoder

    package_root = Path(__file__).resolve().parents[2]
    for relative, expected in ARCHITECTURE_FILE_SHA256.items():
        path = package_root / relative
        if _architecture_source_sha256(path) != expected:
            raise RuntimeError(f"Frozen K1 architecture source changed: {relative}")

    model = make_encoder("neural_atom_k1").to("cuda")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Frozen K1 parameter count changed: {parameters}")
    initial_model_sha256 = _state_sha256(model)
    if initial_model_sha256 != EXPECTED_INITIAL_MODEL_SHA256:
        raise RuntimeError(
            "Frozen K1 seed-42 initialization changed: "
            f"{initial_model_sha256}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_OPTIMIZER_STEPS, eta_min=MIN_LEARNING_RATE
    )
    return model, optimizer, scheduler


def validate_source_archive(source_archive: Path, source_commit: str) -> str:
    """Bind the source zip to its packager sidecars and committed identity."""
    source_archive = source_archive.resolve()
    if not source_archive.is_file():
        raise FileNotFoundError(source_archive)
    sidecars = {
        "commit": source_archive.with_name("SOURCE_COMMIT.txt"),
        "archive_sha256": source_archive.with_name("SOURCE_ARCHIVE_SHA256.txt"),
        "files": source_archive.with_name("SOURCE_FILES.json"),
    }
    missing = [name for name, path in sidecars.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Source archive sidecars are missing: {missing}")
    if sidecars["commit"].read_text(encoding="utf-8").strip() != source_commit:
        raise RuntimeError("Source archive commit sidecar changed")
    archive_sha256 = sha256_file(source_archive)
    if (
        sidecars["archive_sha256"].read_text(encoding="utf-8").strip()
        != archive_sha256
    ):
        raise RuntimeError("Source archive SHA sidecar changed")
    declared_files = json.loads(sidecars["files"].read_text(encoding="utf-8"))
    if declared_files != sorted(set(declared_files)):
        raise RuntimeError("Source file inventory is not sorted and unique")
    if any(
        "__pycache__" in name
        or ".egg-info/" in name
        or name.endswith((".pyc", ".pyo"))
        for name in declared_files
    ):
        raise RuntimeError("Generated metadata leaked into the source inventory")
    with zipfile.ZipFile(source_archive) as archive:
        archived_files = archive.namelist()
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"Corrupt source archive member: {bad_member}")
    if archived_files != declared_files:
        raise RuntimeError("Source archive and file inventory differ")
    missing_architecture = sorted(set(ARCHITECTURE_FILE_SHA256) - set(archived_files))
    if missing_architecture:
        raise RuntimeError(
            f"Source archive omits frozen architecture files: {missing_architecture}"
        )
    return archive_sha256


def _optimizer_step(model, optimizer, batch, mean, std, *, check_finite: bool):
    import torch
    import torch.nn.functional as functional

    from .qm9_gape import forward_gap

    optimizer.zero_grad(set_to_none=True)
    prediction = forward_gap(model, batch, augmented=False)
    target = (batch.y.view(-1) - mean) / std
    loss = functional.l1_loss(prediction, target)
    if check_finite and not bool(torch.isfinite(loss)):
        raise RuntimeError("Training loss became non-finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    if check_finite and not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("Gradient norm became non-finite")
    optimizer.step()
    # Keep the value on-device. Converting every batch to float would force a
    # CUDA synchronization and materially reduce throughput.
    return loss.detach()


def run_preflight(
    *,
    dataset_root: Path,
    manifest_path: Path,
    output: Path,
    platform_id: str,
) -> dict:
    """Run optimizer-inclusive determinism, memory, utilization, and budget gates."""
    determinism = configure_fp32_determinism(SEED)
    import torch

    runtime = build_runtime_manifest(determinism)
    _, paths = validate_full_manifest(
        dataset_root, manifest_path, verify_content=True
    )
    graphs, shards = _load_graphs(paths)
    mean_value, std_value = _target_stats(shards)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    batch = next(iter(_loader(graphs, pass_index=0, start_batch=0))).to(
        "cuda", non_blocking=True
    )
    if int(batch.num_graphs) != PHYSICAL_BATCH:
        raise RuntimeError("Preflight did not receive physical batch 128")
    fixture_sha256 = _batch_sha256(batch)

    repeat_hashes = []
    repeat_losses = []
    for _ in range(2):
        configure_fp32_determinism(SEED)
        model, optimizer, _ = _make_model_and_optimizer()
        model.train()
        repeat_losses.append(
            float(
                _optimizer_step(
                    model, optimizer, batch, mean, std, check_finite=True
                ).cpu()
            )
        )
        torch.cuda.synchronize()
        repeat_hashes.append(_state_sha256(model))
        del model, optimizer
        torch.cuda.empty_cache()
    deterministic_repeat = repeat_hashes[0] == repeat_hashes[1] and repeat_losses[0] == repeat_losses[1]
    if not deterministic_repeat:
        raise RuntimeError("Seeded optimizer-step calibration is not deterministic")

    configure_fp32_determinism(SEED)
    model, optimizer, _ = _make_model_and_optimizer()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    benchmark_batches = iter(_loader(graphs, pass_index=0, start_batch=0))
    for _ in range(PREFLIGHT_WARMUP_STEPS):
        benchmark_batch = next(benchmark_batches).to("cuda", non_blocking=True)
        _optimizer_step(
            model, optimizer, benchmark_batch, mean, std, check_finite=True
        )
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(PREFLIGHT_MEASURED_STEPS):
        benchmark_batch = next(benchmark_batches).to("cuda", non_blocking=True)
        _optimizer_step(
            model, optimizer, benchmark_batch, mean, std, check_finite=False
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # Utilization sampling is deliberately outside the throughput interval;
    # nvidia-smi process startup would otherwise contaminate the estimate.
    utilization = []
    for _ in range(3):
        benchmark_batch = next(benchmark_batches).to("cuda", non_blocking=True)
        _optimizer_step(
            model, optimizer, benchmark_batch, mean, std, check_finite=True
        )
        observed = _gpu_utilization_percent()
        if observed is not None:
            utilization.append(observed)
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    memory_reserve = 1.0 - peak_reserved / total_memory
    graphs_per_second = PREFLIGHT_MEASURED_STEPS * PHYSICAL_BATCH / elapsed
    estimated_hours = SAMPLE_PRESENTATIONS / graphs_per_second / 3600.0
    accepted = (
        memory_reserve >= 0.15
        and estimated_hours <= MAX_ESTIMATED_TRAIN_HOURS
        and deterministic_repeat
    )
    if not accepted:
        raise RuntimeError(
            "Full-run preflight failed: "
            f"reserve={memory_reserve:.3f}, estimate={estimated_hours:.2f}h"
        )

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
    result = {
        "format": "molgap-pcqm-k1-full-preflight-v1",
        "accepted": True,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "runtime_certificate_id": certificate_id,
        "runtime_certificate": certificate,
        "runtime_manifest": runtime,
        "target_stats": {"mean_eV": mean_value, "sample_std_eV": std_value},
        "optimizer_step_calibration": {
            "warmup_steps": PREFLIGHT_WARMUP_STEPS,
            "measured_steps": PREFLIGHT_MEASURED_STEPS,
            "graphs_per_second": graphs_per_second,
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "total_memory_bytes": total_memory,
            "memory_reserve_fraction_from_reserved": memory_reserve,
            "gpu_utilization_samples_percent": utilization,
            "mean_gpu_utilization_percent": (
                float(np.mean(utilization)) if utilization else None
            ),
            "estimated_training_hours": estimated_hours,
            "maximum_estimated_training_hours": MAX_ESTIMATED_TRAIN_HOURS,
        },
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "runtime_manifest.json", runtime)
    atomic_json(output / "runtime_certificate.json", certificate)
    atomic_json(output / "preflight.json", result)
    return result


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    global_step: int,
    target_stats: dict,
    trace: list[dict],
    runtime_fingerprint: str,
    runtime_certificate_id: str,
    source_commit: str,
    source_archive_sha256: str,
) -> None:
    pass_index, next_batch = progress_from_step(global_step)
    atomic_torch_save(
        path,
        {
            "format": CHECKPOINT_FORMAT,
            "training_contract_sha256": TRAINING_CONTRACT_SHA256,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "pass_index": pass_index,
            "next_batch_in_pass": next_batch,
            "target_stats": target_stats,
            "trace": trace,
            "rng_state": capture_rng_state(),
            "runtime_fingerprint": runtime_fingerprint,
            "runtime_certificate_id": runtime_certificate_id,
            "source_commit": source_commit,
            "source_archive_sha256": source_archive_sha256,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
        },
    )


def train_full(
    *,
    dataset_root: Path,
    manifest_path: Path,
    output: Path,
    preflight_path: Path,
    source_commit: str,
    source_archive: Path,
    resume: bool,
    max_wall_seconds: int | None,
) -> dict:
    """Train to the exact sample-exposure budget or atomically pause for resume."""
    if len(source_commit) != 40:
        raise ValueError("A committed SHA-1 source identity is required")
    source_archive_sha256 = validate_source_archive(source_archive, source_commit)
    determinism = configure_fp32_determinism(SEED)
    import torch

    runtime = build_runtime_manifest(determinism)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("accepted") is not True:
        raise RuntimeError("Accepted full-run preflight is required")
    if preflight.get("training_contract_sha256") != TRAINING_CONTRACT_SHA256:
        raise RuntimeError("Preflight training contract changed")
    certificate = preflight.get("runtime_certificate", {})
    if certificate.get("runtime_fingerprint") != runtime["runtime_fingerprint"]:
        raise RuntimeError("Runtime changed since the accepted preflight")
    runtime_certificate_id = canonical_fingerprint(certificate)
    if runtime_certificate_id != preflight.get("runtime_certificate_id"):
        raise RuntimeError("Runtime certificate identity changed")

    _, paths = validate_full_manifest(
        dataset_root, manifest_path, verify_content=True
    )
    graphs, shards = _load_graphs(paths)
    mean_value, std_value = _target_stats(shards)
    target_stats = {"mean_eV": mean_value, "sample_std_eV": std_value}
    if preflight.get("target_stats") != target_stats:
        raise RuntimeError("Full-role target statistics changed since preflight")
    model, optimizer, scheduler = _make_model_and_optimizer()
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    checkpoint_path = output / "last_checkpoint.pt"
    trace_path = output / "trace.json"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "runtime_manifest.json", runtime)
    atomic_json(output / "runtime_certificate.json", certificate)
    atomic_json(output / "training_contract.json", TRAINING_CONTRACT)
    trace: list[dict] = []
    global_step = 0

    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError("Resume requested without last_checkpoint.pt")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        for key, expected in {
            "format": CHECKPOINT_FORMAT,
            "training_contract_sha256": TRAINING_CONTRACT_SHA256,
            "target_stats": target_stats,
            "runtime_fingerprint": runtime["runtime_fingerprint"],
            "runtime_certificate_id": runtime_certificate_id,
            "source_commit": source_commit,
            "source_archive_sha256": source_archive_sha256,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
        }.items():
            if checkpoint.get(key) != expected:
                raise RuntimeError(f"Resume checkpoint contract changed: {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = int(checkpoint["global_step"])
        trace = list(checkpoint["trace"])
        restore_rng_state(checkpoint["rng_state"])
        if progress_from_step(global_step) != (
            int(checkpoint["pass_index"]),
            int(checkpoint["next_batch_in_pass"]),
        ):
            raise RuntimeError("Resume cursor is inconsistent with global step")
    elif checkpoint_path.exists():
        raise FileExistsError("Existing checkpoint requires --resume")

    started = time.perf_counter()
    segment_started = started
    segment_loss = torch.zeros((), dtype=torch.float64, device="cuda")
    segment_rows = 0
    stop_for_walltime = False
    while global_step < MAX_OPTIMIZER_STEPS:
        pass_index, start_batch = progress_from_step(global_step)
        loader = _loader(graphs, pass_index=pass_index, start_batch=start_batch)
        model.train()
        for batch in loader:
            if int(batch.num_graphs) != PHYSICAL_BATCH:
                raise RuntimeError("Training observed a non-128 physical batch")
            batch = batch.to("cuda", non_blocking=True)
            loss = _optimizer_step(
                model,
                optimizer,
                batch,
                mean,
                std,
                check_finite=(global_step + 1) % FINITE_CHECK_EVERY_STEPS == 0,
            )
            scheduler.step()
            global_step += 1
            segment_loss += loss.double() * PHYSICAL_BATCH
            segment_rows += PHYSICAL_BATCH

            checkpoint_due = global_step % CHECKPOINT_EVERY_STEPS == 0
            walltime_due = (
                max_wall_seconds is not None
                and time.perf_counter() - started >= max_wall_seconds - 300
            )
            complete = global_step == MAX_OPTIMIZER_STEPS
            if checkpoint_due or walltime_due or complete:
                torch.cuda.synchronize()
                now = time.perf_counter()
                trace.append(
                    {
                        "global_step": global_step,
                        "sample_presentations": global_step * PHYSICAL_BATCH,
                        "train_normalized_mae": float(segment_loss.cpu()) / segment_rows,
                        "training_seconds": now - segment_started,
                        "graphs_per_second": segment_rows / (now - segment_started),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    target_stats=target_stats,
                    trace=trace,
                    runtime_fingerprint=runtime["runtime_fingerprint"],
                    runtime_certificate_id=runtime_certificate_id,
                    source_commit=source_commit,
                    source_archive_sha256=source_archive_sha256,
                )
                atomic_json(trace_path, {"segments": trace})
                segment_started = now
                segment_loss = torch.zeros((), dtype=torch.float64, device="cuda")
                segment_rows = 0
            if walltime_due:
                stop_for_walltime = True
                break
            if complete:
                break
        if stop_for_walltime:
            break

    status = "paused" if global_step < MAX_OPTIMIZER_STEPS else "complete"
    result = {
        "format": "molgap-pcqm-k1-full-training-result-v1",
        "status": status,
        "training_contract": TRAINING_CONTRACT,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "runtime_certificate_id": runtime_certificate_id,
        "runtime_fingerprint": runtime["runtime_fingerprint"],
        "global_step": global_step,
        "sample_presentations": global_step * PHYSICAL_BATCH,
        "target_stats": target_stats,
        "trace_segments": len(trace),
        "training_seconds_this_invocation": time.perf_counter() - started,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "resume_required": status == "paused",
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(output / "run_summary.json", result)
    if status == "paused":
        return result

    state = model.state_dict()
    assert_finite_state_dict(state, label="K1 final model")
    bundle = {
        "format": MODEL_BUNDLE_FORMAT,
        "model_id": "neural_atom_k1",
        "architecture": {
            "factory": "molgap.qm9_neural_atom.make_encoder",
            "factory_argument": "neural_atom_k1",
            "architecture_source_commit": ARCHITECTURE_SOURCE_COMMIT,
            "architecture_file_sha256": ARCHITECTURE_FILE_SHA256,
            "parameter_count": EXPECTED_PARAMETERS,
            "seed42_initial_model_sha256": EXPECTED_INITIAL_MODEL_SHA256,
        },
        "state_dict": state,
        "target_stats": target_stats,
        "feature_contract": {
            "node_feature_dim": 9,
            "edge_feature_dim": 3,
            "rwse_dim": 16,
            "input_modality": "topology",
        },
        "training_contract": TRAINING_CONTRACT,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "data_identity": {
            "manifest_canonical_sha256": FULL_MANIFEST_CANONICAL_SHA256,
            "topology_aggregate_sha256": FULL_TOPOLOGY_AGGREGATE_SHA256,
            "official_row_manifest_sha256": OFFICIAL_ROW_MANIFEST_SHA256,
        },
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "runtime_fingerprint": runtime["runtime_fingerprint"],
        "runtime_certificate_id": runtime_certificate_id,
        "completed_optimizer_steps": global_step,
        "final_learning_rate": optimizer.param_groups[0]["lr"],
    }
    atomic_torch_save(output / "model_bundle.pt", bundle)
    result["model_bundle_sha256"] = sha256_file(output / "model_bundle.pt")
    result["complete"] = True
    atomic_json(output / "run_summary.json", result)
    artifacts = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "completion_manifest.json"
    }
    atomic_json(output / "completion_manifest.json", {**result, "artifact_sha256": artifacts})
    return result
