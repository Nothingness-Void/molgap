"""Bounded, train-role-only MetaGIN runtime diagnosis; never a model screen."""
from __future__ import annotations

import math
from pathlib import Path
import statistics
import time

from .pcqm_k1_variants_runner import (
    BATCH_SIZE, EPOCHS, ROWS_PER_EPOCH, SEED, _train_loader,
    find_fixed_cache, load_roles,
)
from .pcqm_metagin import MetaGIN2D
from .pcqm_metagin_screen import _step, _target_transform
from .pcqm_metagin_sidecar import attach_accepted_sidecar
from .training_reproducibility import atomic_json, configure_fp32_determinism


WARMUP_STEPS = 8
MEASURED_STEPS = 72
EVALUATION_STEPS = 32
MAX_PROFILE_WALL_SECONDS = 1200


def projected_40_epoch_seconds(train_step_seconds: float,
                               evaluation_step_seconds: float) -> float:
    if not all(math.isfinite(value) and value > 0 for value in (
        train_step_seconds, evaluation_step_seconds,
    )):
        raise ValueError("Runtime projection requires positive finite observations")
    return EPOCHS * (
        ROWS_PER_EPOCH / BATCH_SIZE * train_step_seconds
        + math.ceil(50_000 / BATCH_SIZE) * evaluation_step_seconds
    )


def profile_runtime(output: Path, *, source_commit: str, sidecar_root: Path,
                    transform_path: Path) -> dict:
    """Measure a representative optimizer loop after CUDA warm-up on real rows."""
    import torch

    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    configure_fp32_determinism(SEED)
    fixed_root, fixed = find_fixed_cache()
    raw_roles = load_roles(fixed_root, fixed)
    transform, observation = _target_transform(raw_roles, transform_path)
    roles, sidecar, acceptance = attach_accepted_sidecar(
        raw_roles, sidecar_root, expected_source_commit=source_commit,
    )
    model = MetaGIN2D().to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=4e-4, weight_decay=1e-5, fused=False,
    )
    mean = torch.tensor(float(transform["mean"]), device="cuda")
    std = torch.tensor(float(transform["std"]), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    timings = []
    previous_step_end = time.perf_counter()
    for index, batch in enumerate(_train_loader(roles["train"], 0)):
        if index >= WARMUP_STEPS + MEASURED_STEPS:
            break
        tick = previous_step_end
        batch = batch.to("cuda", non_blocking=True)
        if int(batch.num_graphs) != BATCH_SIZE:
            raise RuntimeError("MetaGIN profiling physical batch differs from 128")
        _step(model, optimizer, batch, mean, std)
        torch.cuda.synchronize()
        previous_step_end = time.perf_counter()
        timings.append(previous_step_end - tick)
        if time.monotonic() - started > MAX_PROFILE_WALL_SECONDS:
            raise RuntimeError("MetaGIN profiling exceeded its bounded wall budget")
    if len(timings) != WARMUP_STEPS + MEASURED_STEPS:
        raise RuntimeError("MetaGIN profile did not cover planned real-batch steps")
    model.eval()
    evaluation = []
    previous_step_end = time.perf_counter()
    with torch.no_grad():
        for index, batch in enumerate(_train_loader(roles["train"], 1)):
            if index >= EVALUATION_STEPS:
                break
            tick = previous_step_end
            batch = batch.to("cuda", non_blocking=True)
            prediction = model(batch)
            if not bool(torch.isfinite(prediction).all()):
                raise RuntimeError("Non-finite MetaGIN profile prediction")
            torch.cuda.synchronize()
            previous_step_end = time.perf_counter()
            evaluation.append(previous_step_end - tick)
    if len(evaluation) != EVALUATION_STEPS:
        raise RuntimeError("MetaGIN profile evaluation sample incomplete")
    train_mean = statistics.mean(timings[WARMUP_STEPS:])
    eval_mean = statistics.mean(evaluation)
    projected_total = projected_40_epoch_seconds(train_mean, eval_mean)
    projected_epoch = projected_total / EPOCHS
    result = {
        "format": "molgap-metagin-2d-train-role-runtime-profile-v1",
        "complete": True, "training_screen_executed": False,
        "source_commit": source_commit,
        "sidecar_aggregate_sha256": sidecar["aggregate_sha256"],
        "sidecar_accepted": acceptance["accepted"],
        "target_transform_asset_id": transform["asset_id"],
        "target_transform_observation": observation,
        "hardware": torch.cuda.get_device_name(0),
        "allocated_device_count": torch.cuda.device_count(),
        "precision": "fp32", "tf32_enabled": False,
        "physical_batch_per_device": BATCH_SIZE,
        "warmup_step_seconds": timings[:WARMUP_STEPS],
        "measured_step_seconds": timings[WARMUP_STEPS:],
        "evaluation_forward_seconds": evaluation,
        "steady_train_step_mean_seconds": train_mean,
        "steady_train_step_median_seconds": statistics.median(timings[WARMUP_STEPS:]),
        "steady_train_step_max_seconds": max(timings[WARMUP_STEPS:]),
        "evaluation_step_mean_seconds": eval_mean,
        "projected_epoch_seconds": projected_epoch,
        "projected_40_epoch_seconds": projected_total,
        "projected_with_20_percent_reserve_seconds": 1.2 * projected_total,
        "profile_wall_seconds": time.monotonic() - started,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "total_memory_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2,
        "train_role_read": True, "internal_development_graphs_loaded": True,
        "internal_development_labels_used": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False, "test_challenge_role_read": False,
    }
    atomic_json(output / "runtime_profile.json", result)
    return result
