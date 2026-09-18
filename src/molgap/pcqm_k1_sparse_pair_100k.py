"""V5 100K profile and training for the K1 sparse-pair candidate."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import struct
import tarfile
import time
from pathlib import Path

import numpy as np

from .k1_sparse_pair import (
    BASE_MODE,
    MODE,
    PARAMETERS,
    make_encoder,
    mechanism_summary,
)
from .screen_policy import canonical_fingerprint, validate_screen_arm
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
from .v4_runtime import (
    sample_std_compat,
    torch_load_compat,
    validate_standard_source_bundle,
)


SEED = 42
TRAIN_ROWS = 100_000
DEVELOPMENT_ROWS = 50_000
BATCH_SIZE = 128
EPOCHS = 40
STEPS_PER_EPOCH = TRAIN_ROWS // BATCH_SIZE
ROWS_PER_EPOCH = STEPS_PER_EPOCH * BATCH_SIZE
SAMPLE_PRESENTATIONS = ROWS_PER_EPOCH * EPOCHS
ROW_ORDER_FINGERPRINT = (
    "e85736669a04029e0fa40e993a085b2e0226b4be98a923a141f999a672ce3f34"
)
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-5
MINIMUM_GAIN_EV = 0.003
LOADER_WORKERS = 2
PROFILE_WARMUP_STEPS = 5
PROFILE_MEASURED_STEPS = 30
MAX_PROFILE_HOURS = 6.0
BASELINE_MAE_EV = 0.1413736343
MANIFEST_SHA256 = (
    "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
)
GEOMETRY_AGGREGATE_SHA256 = (
    "bc83a4bd9a7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5"
)
BENCHMARK_ID = "pcqm4mv2-ogb-fixed-100k-gap-v5"
CHECKPOINT_FORMAT = "molgap-k1-sparse-pair-100k-checkpoint-v1"
FORBIDDEN_FIELDS = (
    "pos",
    "edge_distance",
    "shortest_path",
    "wedge_angle_cos",
    "wedge_edge_ids",
    "geometry_valid",
)


def epoch_order(epoch: int) -> list[int]:
    if not 0 <= epoch < EPOCHS:
        raise ValueError(f"epoch outside frozen range: {epoch}")
    values = list(range(TRAIN_ROWS))
    random.Random(SEED * 1_000_003 + epoch).shuffle(values)
    return values[:ROWS_PER_EPOCH]


def row_order_fingerprint() -> str:
    digest = hashlib.sha256()
    for epoch in range(EPOCHS):
        digest.update(struct.pack("<I", epoch))
        for index in epoch_order(epoch):
            digest.update(struct.pack("<I", index))
    return digest.hexdigest()


def _load_roles(dataset_root: Path, manifest_path: Path, *, verify_content: bool):
    from .pcqm_gptrans_v4 import _load_datasets, validate_fixed_assets

    assets = validate_fixed_assets(
        dataset_root, manifest_path, verify_content=verify_content
    )
    train, train_shards = _load_datasets(assets.train_paths)
    development, development_shards = _load_datasets(assets.development_paths)
    for shard in [*train_shards, *development_shards]:
        for field in FORBIDDEN_FIELDS:
            if field in shard._data:
                del shard._data[field]
                shard.slices.pop(field, None)
    if len(train) != TRAIN_ROWS or len(development) != DEVELOPMENT_ROWS:
        raise RuntimeError("Fixed role count changed")
    return assets, train, train_shards, development


def _target_stats(shards) -> tuple[float, float]:
    import torch

    values = torch.cat([shard._data.y.view(-1).double() for shard in shards])
    if values.numel() != TRAIN_ROWS or not bool(torch.isfinite(values).all()):
        raise RuntimeError("Training targets are incomplete or non-finite")
    return float(values.mean()), float(sample_std_compat(values).clamp_min(1e-6))


def _train_loader(graphs, epoch: int):
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    selected = Subset(graphs, epoch_order(epoch))
    return DataLoader(
        selected,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=LOADER_WORKERS,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        generator=torch.Generator().manual_seed(SEED + epoch),
    )


def _development_loader(graphs):
    from torch_geometric.loader import DataLoader

    return DataLoader(
        graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def _forward(model, batch):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    ).view(-1)


def _state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _base_state_sha256(model) -> str:
    base = model.base if hasattr(model, "base") else model
    return _state_sha256(base)


def _batch_sha256(batch) -> str:
    digest = hashlib.sha256()
    for name in (
        "x",
        "edge_index",
        "edge_attr",
        "batch",
        "random_walk_pe",
        "y",
        "source_idx",
    ):
        value = getattr(batch, name).detach().cpu().contiguous()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _make_optimizer(model):
    import torch

    return torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )


def _optimizer_step(model, optimizer, batch, mean, std):
    import torch
    import torch.nn.functional as functional

    optimizer.zero_grad(set_to_none=True)
    prediction = _forward(model, batch)
    target = (batch.y.view(-1).float() - mean) / std
    loss = functional.l1_loss(prediction, target)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite training loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("Non-finite gradient norm")
    optimizer.step()
    return loss.detach()


def _architecture_preflight(batch) -> dict:
    import torch

    from .qm9_neural_atom import make_encoder as make_k1

    configure_fp32_determinism(SEED)
    baseline = make_k1(BASE_MODE).to("cuda").eval()
    baseline_parameters = sum(value.numel() for value in baseline.parameters())
    baseline_sha256 = _state_sha256(baseline)
    configure_fp32_determinism(SEED)
    candidate = make_encoder().to("cuda").eval()
    parameters = sum(value.numel() for value in candidate.parameters())
    if parameters != PARAMETERS:
        raise RuntimeError(
            f"Sparse-pair parameter count changed: {parameters} != {PARAMETERS}"
        )
    if _base_state_sha256(candidate) != baseline_sha256:
        raise RuntimeError("Candidate K1 initialization changed")
    with torch.no_grad():
        baseline_prediction = _forward(baseline, batch)
        candidate_prediction = _forward(candidate, batch)
    if not torch.equal(baseline_prediction, candidate_prediction):
        raise RuntimeError("Sparse-pair candidate is not exactly nested in K1")
    probe = torch.linspace(
        -1.0,
        1.0,
        steps=int(batch.num_nodes) * 192,
        device=batch.x.device,
    ).reshape(int(batch.num_nodes), 192)
    mechanism = mechanism_summary(
        candidate,
        probe,
        batch.edge_index,
        batch.batch,
        batch.random_walk_pe,
    )
    candidate.train()
    optimizer = _make_optimizer(candidate)
    mean = batch.y.view(-1).float().mean()
    std = sample_std_compat(batch.y.view(-1).float()).clamp_min(1e-6)
    _optimizer_step(candidate, optimizer, batch, mean, std)
    _optimizer_step(candidate, optimizer, batch, mean, std)
    gradients = {
        name: parameter.grad
        for name, parameter in candidate.sparse_pair.named_parameters()
    }
    active_gradients = {
        name: gradient is not None
        and bool(torch.isfinite(gradient).all())
        and float(gradient.abs().sum()) > 0.0
        for name, gradient in gradients.items()
    }
    if not all(active_gradients.values()):
        raise RuntimeError(
            f"Sparse-pair gradients are inactive after two steps: {active_gradients}"
        )
    return {
        "baseline_parameters": baseline_parameters,
        "candidate_parameters": parameters,
        "added_parameters": parameters - baseline_parameters,
        "base_initial_state_sha256": baseline_sha256,
        "exact_nested_initial_predictions": True,
        "mechanism": mechanism,
        "candidate_gradients_after_two_steps": active_gradients,
    }


def _repeatability_calibration(batch, mean, std) -> dict:
    losses = []
    states = []
    for _ in range(2):
        configure_fp32_determinism(SEED)
        model = make_encoder().to("cuda").train()
        optimizer = _make_optimizer(model)
        step_losses = []
        for _ in range(2):
            step_losses.append(
                float(_optimizer_step(model, optimizer, batch, mean, std).cpu())
            )
        losses.append(step_losses)
        states.append(_state_sha256(model))
    exact = losses[0] == losses[1] and states[0] == states[1]
    if not exact:
        raise RuntimeError(
            f"Optimizer-step repeatability failed: losses={losses}, states={states}"
        )
    return {
        "steps": 2,
        "repeat_count": 2,
        "losses": losses,
        "state_sha256": states,
        "exact": True,
    }


def run_profile(
    *,
    dataset_root: Path,
    manifest_path: Path,
    source_archive: Path,
    source_archive_sha256: str,
    source_commit: str,
    output: Path,
    platform_id: str,
) -> dict:
    determinism = configure_fp32_determinism(SEED)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Profile requires exactly one visible accelerator")
    validate_screen_arm(physical_batch_per_device=BATCH_SIZE)
    validate_standard_source_bundle(
        source_archive, source_archive_sha256, source_commit
    )
    assets, train, train_shards, _ = _load_roles(
        dataset_root, manifest_path, verify_content=True
    )
    mean_value, std_value = _target_stats(train_shards)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    batch = next(iter(_train_loader(train, 0))).to("cuda", non_blocking=True)
    if int(batch.num_graphs) != BATCH_SIZE:
        raise RuntimeError("Profile did not receive physical batch 128")
    fixture_sha256 = _batch_sha256(batch)
    architecture = _architecture_preflight(batch)
    repeatability = _repeatability_calibration(batch, mean, std)

    configure_fp32_determinism(SEED)
    model = make_encoder().to("cuda").train()
    optimizer = _make_optimizer(model)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(PROFILE_WARMUP_STEPS):
        _optimizer_step(model, optimizer, batch, mean, std)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(PROFILE_MEASURED_STEPS):
        _optimizer_step(model, optimizer, batch, mean, std)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    graphs_per_second = PROFILE_MEASURED_STEPS * BATCH_SIZE / elapsed
    estimated_hours = SAMPLE_PRESENTATIONS / graphs_per_second / 3600.0
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    reserve = 1.0 - peak_reserved / total_memory
    if reserve < 0.15 or estimated_hours > MAX_PROFILE_HOURS:
        raise RuntimeError(
            f"Profile gate failed: reserve={reserve:.3f}, estimate={estimated_hours:.2f}h"
        )
    runtime = build_runtime_manifest(determinism)
    certificate = {
        "format": "molgap-k1-sparse-pair-runtime-certificate-v1",
        "status": "accepted",
        "platform_id": platform_id,
        "accelerator": torch.cuda.get_device_name(0),
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": BATCH_SIZE,
        "tail_batch_policy": "drop_last",
        "runtime_fingerprint": runtime["runtime_fingerprint"],
        "fixture_sha256": fixture_sha256,
        "source_archive_sha256": source_archive_sha256,
        "source_commit": source_commit,
        "optimizer_step_repeatability": repeatability,
    }
    certificate["certificate_id"] = canonical_fingerprint(certificate)
    result = {
        "format": "molgap-k1-sparse-pair-100k-profile-v1",
        "status": "accepted",
        "benchmark_id": BENCHMARK_ID,
        "fixed_manifest_sha256": sha256_file(manifest_path),
        "fixed_geometry_aggregate_sha256": assets.manifest[
            "geometry_aggregate_sha256"
        ],
        "source_archive_sha256": source_archive_sha256,
        "source_commit": source_commit,
        "architecture": architecture,
        "optimizer_step_repeatability": repeatability,
        "runtime_certificate": certificate,
        "runtime_manifest": runtime,
        "target_stats": {
            "mean_eV": mean_value,
            "sample_std_eV": std_value,
        },
        "profile": {
            "warmup_steps": PROFILE_WARMUP_STEPS,
            "measured_steps": PROFILE_MEASURED_STEPS,
            "elapsed_seconds": elapsed,
            "graphs_per_second": graphs_per_second,
            "estimated_training_hours": estimated_hours,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "total_memory_bytes": total_memory,
            "memory_reserve_fraction": reserve,
        },
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "runtime_manifest.json", runtime)
    atomic_json(output / "runtime_certificate.json", certificate)
    atomic_json(output / "profile.json", result)
    atomic_json(
        output / "completion_manifest.json",
        {
            "format": "molgap-k1-sparse-pair-profile-completion-v1",
            "complete": True,
            "profile_sha256": sha256_file(output / "profile.json"),
            "source_commit": source_commit,
        },
    )
    return result


def _evaluate(model, graphs, mean, std) -> dict:
    import torch

    model.eval()
    predictions = []
    targets = []
    source_indices = []
    with torch.no_grad():
        for batch in _development_loader(graphs):
            batch = batch.to("cuda", non_blocking=True)
            predictions.append((_forward(model, batch) * std + mean).float().cpu())
            targets.append(batch.y.view(-1).float().cpu())
            source_indices.append(batch.source_idx.view(-1).long().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    source_idx = torch.cat(source_indices)
    expected = torch.arange(TRAIN_ROWS, TRAIN_ROWS + DEVELOPMENT_ROWS)
    if not torch.equal(source_idx, expected):
        raise RuntimeError("Development source order changed")
    return {
        "mae_eV": float((prediction - target).abs().mean()),
        "prediction_eV": prediction,
        "target_eV": target,
        "source_idx": source_idx,
    }


def _save_recovery_chunk(output: Path, epoch: int) -> None:
    chunk = output / f"recovery_epoch_{epoch + 1:02d}.tar"
    temporary = chunk.with_suffix(".tmp")
    with tarfile.open(temporary, "w") as archive:
        for name in (
            "last_checkpoint.pt",
            "best_model.pt",
            "best_development_payload.pt",
            "trace.json",
            "profile.json",
        ):
            path = output / name
            if path.is_file():
                archive.add(path, arcname=name)
    os.replace(temporary, chunk)


def run_training(
    *,
    dataset_root: Path,
    manifest_path: Path,
    profile_path: Path,
    source_archive: Path,
    source_archive_sha256: str,
    source_commit: str,
    output: Path,
    platform_id: str,
) -> dict:
    determinism = configure_fp32_determinism(SEED)
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Training requires exactly one visible accelerator")
    validate_standard_source_bundle(
        source_archive, source_archive_sha256, source_commit
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("status") != "accepted":
        raise RuntimeError("Training requires an accepted profile")
    for key, expected in {
        "source_archive_sha256": source_archive_sha256,
        "source_commit": source_commit,
    }.items():
        if profile.get(key) != expected:
            raise RuntimeError(f"Profile identity mismatch: {key}")
    certificate = profile["runtime_certificate"]
    if certificate.get("platform_id") != platform_id:
        raise RuntimeError("Profile platform changed")
    runtime = build_runtime_manifest(determinism)
    if runtime["runtime_fingerprint"] != certificate["runtime_fingerprint"]:
        raise RuntimeError("Training runtime differs from profile runtime")

    assets, train, train_shards, development = _load_roles(
        dataset_root, manifest_path, verify_content=True
    )
    mean_value, std_value = _target_stats(train_shards)
    if profile["target_stats"] != {
        "mean_eV": mean_value,
        "sample_std_eV": std_value,
    }:
        raise RuntimeError("Target statistics changed after profile")
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    configure_fp32_determinism(SEED)
    model = make_encoder().to("cuda")
    optimizer = _make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    output.mkdir(parents=True, exist_ok=True)
    copied_profile = output / "profile.json"
    if copied_profile.resolve() != profile_path.resolve():
        copied_profile.write_bytes(profile_path.read_bytes())
    checkpoint_path = output / "last_checkpoint.pt"
    start_epoch = 0
    trace = []
    best = math.inf
    best_epoch = -1
    if checkpoint_path.is_file():
        checkpoint = torch_load_compat(
            checkpoint_path, map_location="cuda", weights_only=False
        )
        for key, expected in {
            "format": CHECKPOINT_FORMAT,
            "source_archive_sha256": source_archive_sha256,
            "source_commit": source_commit,
            "runtime_certificate_id": certificate["certificate_id"],
            "row_order_fingerprint": ROW_ORDER_FINGERPRINT,
        }.items():
            if checkpoint.get(key) != expected:
                raise RuntimeError(f"Checkpoint identity mismatch: {key}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trace = list(checkpoint["trace"])
        best = float(checkpoint["best_development_mae_eV"])
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint["rng_state"])

    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        normalized_absolute = 0.0
        rows = 0
        started = time.perf_counter()
        for batch in _train_loader(train, epoch):
            batch = batch.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, batch)
            target = (batch.y.view(-1).float() - mean) / std
            loss = functional.l1_loss(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("Non-finite gradient norm")
            optimizer.step()
            normalized_absolute += float(
                (prediction.detach() - target).abs().sum()
            )
            rows += int(target.numel())
        if rows != ROWS_PER_EPOCH:
            raise RuntimeError("Training exposure changed")
        development_result = _evaluate(model, development, mean, std)
        improved = development_result["mae_eV"] < best
        if improved:
            best = development_result["mae_eV"]
            best_epoch = epoch
            assert_finite_state_dict(model.state_dict(), label="best sparse-pair model")
            atomic_torch_save(output / "best_model.pt", model.state_dict())
            atomic_torch_save(
                output / "best_development_payload.pt",
                {
                    "prediction_eV": development_result["prediction_eV"],
                    "target_eV": development_result["target_eV"],
                    "source_idx": development_result["source_idx"],
                },
            )
        row = {
            "epoch": epoch,
            "optimizer_steps": (epoch + 1) * STEPS_PER_EPOCH,
            "sample_presentations": (epoch + 1) * ROWS_PER_EPOCH,
            "train_normalized_mae": normalized_absolute / rows,
            "development_gap_mae_eV": development_result["mae_eV"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
            "improved": improved,
        }
        trace.append(row)
        scheduler.step()
        atomic_torch_save(
            checkpoint_path,
            {
                "format": CHECKPOINT_FORMAT,
                "mode": MODE,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "trace": trace,
                "best_development_mae_eV": best,
                "best_epoch": best_epoch,
                "rng_state": capture_rng_state(),
                "source_archive_sha256": source_archive_sha256,
                "source_commit": source_commit,
                "runtime_certificate_id": certificate["certificate_id"],
                "row_order_fingerprint": ROW_ORDER_FINGERPRINT,
            },
        )
        atomic_json(output / "trace.json", {"epochs": trace})
        if (epoch + 1) % 10 == 0:
            _save_recovery_chunk(output, epoch)
        print(
            f"{MODE} ep{epoch:02d} train={row['train_normalized_mae']:.6f} "
            f"dev={development_result['mae_eV']:.6f}eV "
            f"{row['seconds']:.1f}s{' *' if improved else ''}",
            flush=True,
        )

    gain = BASELINE_MAE_EV - best
    training = {
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "development_gap_mae_eV": best,
        "baseline_k1_gap_mae_eV": BASELINE_MAE_EV,
        "scalar_gain_vs_k1_eV": gain,
        "minimum_gain_eV": MINIMUM_GAIN_EV,
        "scalar_gate_passed": gain >= MINIMUM_GAIN_EV,
        "epochs_completed": len(trace),
        "optimizer_steps": EPOCHS * STEPS_PER_EPOCH,
        "sample_presentations": SAMPLE_PRESENTATIONS,
        "mean_epoch_seconds": float(np.mean([row["seconds"] for row in trace])),
        "mean_graphs_per_second": SAMPLE_PRESENTATIONS
        / sum(row["seconds"] for row in trace),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "best_model_sha256": sha256_file(output / "best_model.pt"),
        "payload_sha256": sha256_file(output / "best_development_payload.pt"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    record = {
        "format": "molgap-k1-sparse-pair-100k-result-v1",
        "complete": True,
        "benchmark_id": BENCHMARK_ID,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "fixed_manifest_sha256": sha256_file(manifest_path),
        "fixed_geometry_aggregate_sha256": assets.manifest[
            "geometry_aggregate_sha256"
        ],
        "row_order_fingerprint": ROW_ORDER_FINGERPRINT,
        "runtime_certificate": certificate,
        "architecture": profile["architecture"],
        "training": training,
        "strict_paired_comparison_status": "pending-reference-payload-acceptance",
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(output / "arm_record.json", record)
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "completion_manifest.json"
    }
    completion = {
        **record,
        "artifact_sha256": artifacts,
    }
    atomic_json(output / "completion_manifest.json", completion)
    return completion
