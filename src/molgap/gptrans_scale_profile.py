"""Bounded G1 scale qualification using the owning GPTrans step and fixed loader.

Disposable optimizer updates are profiling, never scientific training evidence.
No development tensors, selected checkpoints or validation scores are consumed.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .pcqm_gptrans_v4 import (
    ExponentialMovingAverage, FrozenEpochScheduler, _batch_sha256, _forward,
    _load_datasets, _make_training_state, _optimizer_step, _run_target_stats,
    _state_sha256,
)
from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
from .pcqm_k1_scale_runner import find_cache
from .training_reproducibility import (
    atomic_json, build_runtime_manifest, configure_fp32_determinism, sha256_file,
)

STEPS = 46_860
BATCH = 128
WARMUP = 5
MEASURED = 30


def model_binding(contract):
    """Keep the legacy G1 default; allow only the frozen local-stream extension."""
    variant = contract.get("model_variant", "degree_scale_ema999")
    expected = {"degree_scale_ema999": ("degree_initial_state.pt", 5246817),
                "degree_bond_local_ema999": ("degree_bond_local_ema999_initial.pt", 5871201)}
    if variant not in expected:
        raise ValueError("Unsupported fixed500K model variant")
    filename, parameters = expected[variant]
    if contract.get("initial_file", filename) != filename or contract.get("expected_parameters", parameters) != parameters:
        raise ValueError("Scale model/initialization/parameter binding changed")
    return variant, filename, parameters


def clock_projection():
    """Counterfactual clocks, not observed parameter shrinkage or tuned decay."""
    lrs = [FrozenEpochScheduler.learning_rate(e) for e in range(60)]
    return {str(rows): {"optimizer_steps": (rows // BATCH) * 60,
        "lr_sum": (rows // BATCH) * sum(lrs),
        "decay_only_factor": math.exp((rows // BATCH) * sum(math.log1p(-lr * .05) for lr in lrs))}
        for rows in (100_000, 500_000)}


def profile(inputs: Path, output: Path, contract: dict, *, source_identity: dict):
    import torch
    from torch_geometric.loader import DataLoader

    variant, initial_file, parameters = model_binding(contract)
    started = time.perf_counter()
    settings = configure_fp32_determinism(42)
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Qualification requires one isolated T4")
    root, manifest = find_cache(FIXED_500K_MANIFEST_SHA256)
    train_paths = []
    for shard in manifest["geometry_shards"]:
        if shard["role"] != "train":
            continue
        path = (root / shard["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != shard["sha256"]:
            raise ValueError("Fixed training shard identity changed")
        train_paths.append(path)
    graphs, _ = _load_datasets(tuple(train_paths))
    if len(graphs) != 500_000:
        raise ValueError("Qualification train role count changed")
    indices = torch.randperm(len(graphs), generator=torch.Generator().manual_seed(42))[:BATCH * (WARMUP + MEASURED)]
    loader = DataLoader(graphs, batch_size=BATCH, sampler=indices.tolist(), num_workers=4,
        persistent_workers=True, pin_memory=True, generator=torch.Generator().manual_seed(42))
    mean_value, std_value = _run_target_stats([], variant, inputs / "target_transform.json")
    mean, std = torch.tensor(mean_value, device="cuda"), torch.tensor(std_value, device="cuda")
    initial = inputs / initial_file
    if sha256_file(initial) != contract["initial_file_sha256"]:
        raise ValueError("Qualification initialization changed")
    fixture = next(iter(loader)).to("cuda")
    repeats = []
    for _ in range(2):
        configure_fp32_determinism(42)
        model, optimizer, scheduler, unused_ema = _make_training_state(initial, variant)
        if sum(p.numel() for p in model.parameters()) != parameters:
            raise ValueError("Qualified model parameter count changed")
        del unused_ema
        scheduler.step(0)
        ema_slow = ExponentialMovingAverage(model, .9999)
        ema_fast = ExponentialMovingAverage(model, .999)
        loss = _optimizer_step(model, optimizer, ema_slow, fixture, mean, std, check_finite=True)
        ema_fast.update(model)
        repeats.append({"loss": float(loss.cpu()), "live_state_sha256": _state_sha256(model)})
        del model, optimizer, scheduler, ema_slow, ema_fast
        torch.cuda.empty_cache()
    if repeats[0] != repeats[1]:
        raise RuntimeError("Deterministic optimizer-inclusive calibration failed")
    configure_fp32_determinism(42)
    model, optimizer, scheduler, unused_ema = _make_training_state(initial, variant)
    del unused_ema
    scheduler.step(0)
    ema_slow, ema_fast = ExponentialMovingAverage(model, .9999), ExponentialMovingAverage(model, .999)
    batches = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARMUP):
        batch = next(batches).to("cuda", non_blocking=True)
        _optimizer_step(model, optimizer, ema_slow, batch, mean, std, check_finite=True)
        ema_fast.update(model)
    torch.cuda.synchronize()
    began = time.perf_counter()
    for _ in range(MEASURED):
        batch = next(batches).to("cuda", non_blocking=True)
        _optimizer_step(model, optimizer, ema_slow, batch, mean, std, check_finite=True)
        ema_fast.update(model)
    torch.cuda.synchronize()
    optimizer_seconds = time.perf_counter() - began
    model.eval()
    began = time.perf_counter()
    with torch.no_grad():
        for _ in range(8):
            prediction = _forward(model, fixture)
            if not bool(torch.isfinite(prediction).all()):
                raise RuntimeError("Nonfinite train-role evaluation proxy")
    torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - began
    reserve = 1 - torch.cuda.max_memory_reserved() / torch.cuda.get_device_properties(0).total_memory
    # Sixty checkpoints each observe live and both filters on 50K dev rows.
    estimate = (STEPS * optimizer_seconds / MEASURED
        + 60 * 3 * 50_000 * evaluation_seconds / (8 * BATCH)) / 3600
    result = {"format": "molgap-g1-scale-qualification-v1", "status": "COMPLETE",
        "qualification_passed": reserve >= .15 and estimate <= contract["training_estimate_cap_hours"],
        "scientific_training_executed": False, "disposable_optimizer_updates": 2 + WARMUP + MEASURED,
        "model_inference_executed": True, "development_role_read": False,
        "official_validation_role_read": False, "test_dev_role_read": False,
        "test_challenge_role_read": False, "parameters": sum(p.numel() for p in model.parameters()),
        "precision": "fp32", "tf32_enabled": False, "physical_batch": BATCH,
        "train_role": [0, 500_000], "manifest_sha256": FIXED_500K_MANIFEST_SHA256,
        "fixture_sha256": _batch_sha256(fixture), "source_identity": source_identity,
        "runtime": build_runtime_manifest(settings), "deterministic_repeat": repeats,
        "optimizer_graphs_per_second": MEASURED * BATCH / optimizer_seconds,
        "estimated_equal_exposure_training_hours": estimate,
        "estimation_scope": "sampled optimizer plus train-role evaluation proxy; excludes validation size distribution, checkpoint IO and bootstrap; not a release certificate",
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "memory_reserve_fraction": reserve,
        "clock_counterfactual": clock_projection(), "worker_wall_seconds": time.perf_counter() - started}
    atomic_json(output / "qualification.json", result)
    return result
