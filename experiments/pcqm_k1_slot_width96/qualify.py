"""Frozen CPU initialization and retained-reference layer-nine ablation.

No optimizer, backward pass or training-role access. Parent process enforces
the wall budget even when a native inference call cannot be interrupted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INITIAL_SHA = "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd"
CHECKPOINT_SHA = "4a7dab43f50f5f3216b224374c213a01cc82cfb4160dcd6d54aab1ef614fce17"
PREDICTIONS_SHA = "857fe31475d6728c3c9b0612571a0397a67db93815714c20ac107e06e44d4e82"
REQUIRED_SOURCES = {
    "experiments/pcqm_k1_slot_width96/qualify.py",
    "src/molgap/k1_slot_width96.py", "src/molgap/qm9_neural_atom.py",
    "src/molgap/k1_screen_training.py", "src/molgap/training_reproducibility.py",
    "src/molgap/v4_runtime.py",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze():
    from molgap.training_reproducibility import sha256_file
    contract = read(OUT / "cpu_frozen.json")
    trajectory_path = OUT / "qualification/trajectory.json"
    trajectory = read(trajectory_path)
    if trajectory["record_mode"] != "prospective":
        raise RuntimeError("CPU execution requires the published prospective trajectory")
    if sha256_file(trajectory_path) != contract["trajectory_sha256"]:
        raise RuntimeError("Prospective trajectory changed after CPU freeze")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != contract["source_commit"] or trajectory["state_at_start"]["source_commit"] != commit:
        raise RuntimeError("Prospective CPU source commit changed")
    sources = contract["source_sha256"]
    if not REQUIRED_SOURCES <= sources.keys():
        raise RuntimeError("CPU freeze lacks required worker/model sources")
    for relative, expected in sources.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen source changed: {relative}")
    from molgap.k1_slot_width96 import CONFIG, EXPECTED_PARAMETER_COUNT, REFERENCE_PARAMETER_COUNT
    expected_models = {"config": CONFIG, "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
                       "reference_parameter_count": REFERENCE_PARAMETER_COUNT}
    if contract["model_declarations"] != expected_models:
        raise RuntimeError("Frozen model declarations changed")
    if contract["max_wall_seconds"] != 300:
        raise RuntimeError("CPU wall ceiling must remain 300 seconds")
    return contract, trajectory


def bound_path(contract, key, *, pinned=None):
    from molgap.training_reproducibility import sha256_file
    item = contract[key]
    path = Path(item["path"])
    if not path.is_absolute():
        path = ROOT / path
    if pinned is not None and item["sha256"] != pinned:
        raise RuntimeError(f"Accepted {key} identity changed")
    if sha256_file(path) != item["sha256"]:
        raise RuntimeError(f"Frozen input changed: {key}")
    return path


def worker():
    wall, cpu = time.perf_counter(), time.process_time()
    contract, trajectory = freeze()
    import numpy as np
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from molgap.k1_screen_training import _PackedGraphDatasetFactory, _forward, FORBIDDEN_MODEL_FIELDS
    from molgap.k1_slot_width96 import make_encoder as make_candidate
    from molgap.qm9_neural_atom import make_encoder
    from molgap.training_reproducibility import (
        atomic_json, atomic_torch_save, assert_finite_state_dict,
        configure_fp32_determinism, sha256_file,
    )
    from molgap.v4_runtime import state_dict_sha256

    def costs(start_wall, start_cpu):
        return {"wall_seconds": time.perf_counter() - start_wall,
                "process_cpu_seconds": time.process_time() - start_cpu,
                "cpu_threads": 4, "device_seconds": {"status": "not_applicable", "value": None},
                "queue_seconds": {"status": "not_applicable", "value": None}}

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    runtime = configure_fp32_determinism(42)
    baseline = make_encoder("neural_atom_k1")
    baseline_sha = state_dict_sha256(baseline.state_dict())
    if baseline_sha != INITIAL_SHA or sum(p.numel() for p in baseline.parameters()) != 3_658_817:
        raise RuntimeError("Accepted default192 initialization changed")
    configure_fp32_determinism(42)
    explicit = make_encoder("neural_atom_k1", latent_channels=64)
    if state_dict_sha256(explicit.state_dict()) != baseline_sha:
        raise RuntimeError("Default differs from explicit latent64 initialization")
    del explicit, baseline
    configure_fp32_determinism(42)
    candidate = make_candidate()
    state = {key: value.detach().cpu().clone() for key, value in candidate.state_dict().items()}
    assert_finite_state_dict(state, label="candidate initialization")
    digest = state_dict_sha256(state)
    output = OUT / "qualification_initial_state.pt"
    atomic_torch_save(output, state)
    restored = torch.load(output, map_location="cpu", weights_only=True)
    candidate.load_state_dict(restored, strict=True)
    if state_dict_sha256(candidate.state_dict()) != digest:
        raise RuntimeError("Candidate atomic roundtrip changed state")
    initialization_cost = costs(wall, cpu)
    qualification_report = {
        "schema": "molgap-k1-slot96-cpu-initialization-v1", "status": "CPU_INITIALIZATION_QUALIFIED",
        "trajectory_id": trajectory["trajectory_id"], "action_id": "A001",
        "candidate_parameters": 3_853_793, "reference_parameters": 3_658_817,
        "reference_default_state_sha256": baseline_sha, "reference_default_matches_explicit_latent64": True,
        "initial_state_sha256": digest, "initial_state_file_sha256": sha256_file(output),
        "finite": True, "saved_state_roundtrip": True, "runtime": runtime,
        "cost": initialization_cost, "data_access": False, "label_access": False,
        "gpu_qualification": "pending", "training_submission": "not_performed",
    }
    del state, restored

    inference_wall, inference_cpu = time.perf_counter(), time.process_time()
    checkpoint = bound_path(contract, "reference_checkpoint", pinned=CHECKPOINT_SHA)
    prediction_path = bound_path(contract, "reference_predictions", pinned=PREDICTIONS_SHA)
    transform = read(bound_path(contract, "target_transform"))
    # Reuse the accepted _target_stats asset without reopening training labels.
    from molgap.comparison_readiness import validate_target_transform_asset
    validate_target_transform_asset(transform)
    mean, std = transform["mean"], transform["std"]
    if (mean, std) != (5.3383002281188965, 1.275090217590332):
        raise RuntimeError("Accepted target transform differs from _target_stats")
    graphs = _PackedGraphDatasetFactory.load(bound_path(contract, "development_shard"))
    if len(graphs) != 50_000 or any(name in graphs._data for name in FORBIDDEN_MODEL_FIELDS):
        raise RuntimeError("Development shard size or pure2D fields changed")
    if not torch.equal(graphs._data.source_idx.view(-1).long(), torch.arange(100_000, 150_000)):
        raise RuntimeError("Development source membership/order changed")
    offsets = np.sort(np.random.default_rng(20261002).choice(50_000, 2048, replace=False))
    loader = DataLoader(Subset(graphs, offsets.tolist()), batch_size=64, shuffle=False,
                        drop_last=False, num_workers=0)
    fixture = next(iter(loader))
    candidate.eval()
    with torch.inference_mode():
        candidate_prediction = _forward(candidate, fixture)
    if candidate_prediction.shape != (64,) or not torch.isfinite(candidate_prediction).all():
        raise RuntimeError("Candidate real pure2D forward failed finite/shape qualification")
    if state_dict_sha256(candidate.state_dict()) != digest:
        raise RuntimeError("Candidate pure2D forward mutated frozen state")
    qualification_report.update(real_pure2d_forward_finite=True, fixture_rows=64,
                                data_access=True, label_access=False,
                                fixture_role="development-input-only-no-label-metric",
                                candidate_forward_cost_included_in="ablation inference cost")
    atomic_json(OUT / "qualification_result.json", qualification_report)
    del candidate, candidate_prediction, fixture
    retained = torch.load(prediction_path, map_location="cpu", weights_only=True)
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if selected["context"] != retained["context"]:
        raise RuntimeError("Checkpoint/prediction context mismatch")
    context = selected["context"]
    if (context.get("arm_id") != "reference" or context.get("family_name") != "neural_atom_k1"
            or context.get("source_commit") != "f21920ba5d9f3fe148cfa18803d53c0a5ec8f670"
            or context.get("run_reference") != "nvoid912/molgap-k1-v4-ssma-accuracy-100k-s42-v1"):
        raise RuntimeError("Accepted selected reference context changed")
    if (type(selected["epoch"]) is not int or selected["epoch"] != 40
            or type(selected["optimizer_step"]) is not int or selected["optimizer_step"] != 31_240
            or selected["weights"] != "live"):
        raise RuntimeError("Accepted selected reference epoch/step/weights changed")
    for field in ("source_idx", "target_eV", "prediction_eV"):
        if retained[field].numel() != 50_000 or not torch.isfinite(retained[field]).all():
            raise RuntimeError(f"Retained prediction shape/finite check failed: {field}")
    retained_source = retained["source_idx"].view(-1).long()
    retained_target = retained["target_eV"].view(-1).float()
    retained_prediction = retained["prediction_eV"].view(-1).float()
    if not torch.equal(retained_source, torch.arange(100_000, 150_000)):
        raise RuntimeError("Retained prediction source identity changed")
    model = make_encoder("neural_atom_k1").eval()
    model.load_state_dict(selected["model"], strict=True)
    model.requires_grad_(False)
    assert_finite_state_dict(model.state_dict(), label="retained reference")
    original_state_sha = state_dict_sha256(model.state_dict())

    def predict():
        targets, predictions, indices = [], [], []
        with torch.inference_mode():
            for batch in loader:
                if time.perf_counter() - wall >= 300:
                    raise TimeoutError("CPU wall ceiling reached")
                if any(name in batch for name in FORBIDDEN_MODEL_FIELDS):
                    raise RuntimeError("Geometry reached inference")
                prediction = _forward(model, batch) * std + mean
                target = batch.y.view(-1).float()
                if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                    raise RuntimeError("Nonfinite prediction/target")
                targets.append(target.cpu()); predictions.append(prediction.cpu())
                indices.append(batch.source_idx.view(-1).long().cpu())
        return tuple(torch.cat(parts).numpy() for parts in (targets, predictions, indices))

    target, original, source_idx = predict()
    if not np.array_equal(source_idx, retained_source[offsets].numpy()) or not np.array_equal(
            target, retained_target[offsets].numpy()):
        raise RuntimeError("Retained subset source/target exact reconstruction failed")
    replicate_delta = float(np.max(np.abs(original - retained_prediction[offsets].numpy())))
    if replicate_delta > 1e-4:
        raise RuntimeError(f"Original baseline reconstruction exceeds 1e-4 eV: {replicate_delta}")
    # Returning the incoming hidden tensor removes only layer-nine's mixer update.
    handle = model.neural_atom_mixers["9"].register_forward_hook(lambda module, args, output: args[0])
    try:
        ablated_target, zero9, ablated_source = predict()
    finally:
        handle.remove()
    if not np.array_equal(target, ablated_target) or not np.array_equal(source_idx, ablated_source):
        raise RuntimeError("Ablation row identities changed")
    if state_dict_sha256(model.state_dict()) != original_state_sha:
        raise RuntimeError("Frozen ablation mutated model weights")
    error_delta = np.abs(zero9.astype(np.float64) - target) - np.abs(original.astype(np.float64) - target)
    prediction_delta = zero9.astype(np.float64) - original
    rng = np.random.default_rng(42)
    bootstrap = np.asarray([error_delta[rng.integers(0, len(error_delta), len(error_delta))].mean()
                            for _ in range(1000)])
    rows_path = OUT / "ablation_rows.npz"
    temporary = rows_path.with_name("." + rows_path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, offsets=offsets, source_idx=source_idx, target_eV=target,
                            original_prediction_eV=original, zero9_prediction_eV=zero9,
                            per_row_error_delta_eV=error_delta, per_row_prediction_delta_eV=prediction_delta)
    os.replace(temporary, rows_path)
    delta_mae = float(error_delta.mean())
    atomic_json(OUT / "ablation_result.json", {
        "schema": "molgap-k1-slot96-zero-layer9-ablation-v1", "status": "CPU_ABLATION_COMPLETE",
        "trajectory_id": trajectory["trajectory_id"], "action_id": "A001", "rows": 2048,
        "subset_seed": 20261002, "batch_size": 64, "loader_workers": 0,
        "original_mae_eV": float(np.abs(original.astype(np.float64) - target).mean()),
        "zero9_mae_eV": float(np.abs(zero9.astype(np.float64) - target).mean()),
        "zero9_minus_original_mae_eV": delta_mae,
        "prediction_delta_eV": {"mean": float(prediction_delta.mean()), "std": float(prediction_delta.std()),
            "min": float(prediction_delta.min()), "max": float(prediction_delta.max()),
            "quantiles": np.quantile(prediction_delta, [0.025, 0.5, 0.975]).tolist()},
        "paired_row_bootstrap": {"replicates": 1000, "seed": 42,
            "mean_delta_ci95_eV": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "scope": "row uncertainty; not training stochasticity"},
        "original_replicate_max_delta_eV": replicate_delta, "target_source_exact": True,
        "selected_epoch": selected["epoch"], "selected_optimizer_step": selected["optimizer_step"],
        "selected_weights": selected["weights"], "reference_context": context,
        "all_weights_preserved": True, "layer3_layer6_preserved": True, "finite": True,
        "proceed_to_slot_capacity_screen": delta_mae >= 0.001,
        "gate_eV": 0.001, "interpretation": "Discriminator usefulness; not proof of a latent64 capacity bottleneck",
        "rows_sha256": sha256_file(rows_path), "inference_cost": costs(inference_wall, inference_cpu),
        "total_cost": costs(wall, cpu), "training_role_read": False,
        "official_validation_used": False, "test_dev_used": False, "test_challenge_used": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    started = time.perf_counter()
    freeze()
    for name in ("qualification_initial_state.pt", "qualification_result.json", "ablation_rows.npz",
                 "ablation_result.json", "cpu_failure.json"):
        if (OUT / name).exists():
            raise FileExistsError(f"Retained output exists; reconcile before retry: {name}")
    from molgap.training_reproducibility import atomic_json
    try:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                       cwd=ROOT, check=True, timeout=max(0, 300 - (time.perf_counter() - started)))
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        atomic_json(OUT / "cpu_failure.json", {"status": "CPU_HALTED", "error": str(error),
                    "wall_seconds": time.perf_counter() - started,
                    "scope": "qualification incomplete; no training authorized"})
        raise


if __name__ == "__main__":
    main()
