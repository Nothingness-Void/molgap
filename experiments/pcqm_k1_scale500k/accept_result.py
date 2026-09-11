"""No-inference acceptance for the frozen K1 500K paired result."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from molgap.pcqm_gap_data import sha256_file
from molgap.pcqm_k1_scale_runner import (
    EXPECTED_PARAMETERS,
    EPOCHS,
    FIXED_500K_DATASET,
    FIXED_500K_GEOMETRY_SHA256,
    LOADER_WORKERS,
    MIN_GAIN_EV,
    PRECISION,
    SCNET_REFERENCE_CACHE_SHA256,
)
from molgap.pcqm_k1_scale import VALIDATION_ROWS
from molgap.pcqm_k1_shadow_audit import paired_bootstrap
from molgap.screen_policy import validate_paired_screen_contract


def accept(root: Path, *, source_commit: str, cache_sha256: str) -> dict:
    import numpy as np
    import torch

    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    completion = json.loads((root / "completion_manifest.json").read_text(encoding="utf-8"))
    for key, expected in {
        "format": "molgap-pcqm-k1-scale500k-result-v4",
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "fixed_dataset": FIXED_500K_DATASET,
        "fixed_geometry_aggregate_sha256": FIXED_500K_GEOMETRY_SHA256,
        "scnet_aggregate_sha256": SCNET_REFERENCE_CACHE_SHA256,
        "minimum_gain_eV": MIN_GAIN_EV,
        "precision": PRECISION,
        "loader_workers_per_arm": LOADER_WORKERS,
        "model_inference_executed_by_acceptance": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "shadow_labels_read": False,
    }.items():
        if metrics.get(key) != expected or completion.get(key) != expected:
            raise RuntimeError(f"Result contract changed: {key}")
    results = metrics["results"]
    if set(results) != set(EXPECTED_PARAMETERS):
        raise RuntimeError("Scale arm set changed")
    comparability = validate_paired_screen_contract([value["contract"] for value in results.values()])
    if comparability != metrics["comparability"]:
        raise RuntimeError("Scale comparability changed")
    payloads = {}
    for arm, expected_parameters in EXPECTED_PARAMETERS.items():
        result = results[arm]
        if result["training"]["parameter_count"] != expected_parameters:
            raise RuntimeError(f"{arm} parameter count changed")
        if result["preflight"]["physical_batch_per_device"] != 128:
            raise RuntimeError(f"{arm} batch changed")
        if (
            result["preflight"].get("precision") != PRECISION
            or result["training"].get("precision") != PRECISION
            or result["training"].get("loader_workers") != LOADER_WORKERS
        ):
            raise RuntimeError(f"{arm} optimized runtime contract changed")
        if result["preflight"]["memory_reserve_fraction"] < 0.15:
            raise RuntimeError(f"{arm} memory reserve failed")
        if result["preflight"].get("finite_forward_backward") is not True:
            raise RuntimeError(f"{arm} preflight finiteness changed")
        training = result["training"]
        if training.get("epochs_completed") != EPOCHS or not 0 <= int(
            training.get("best_epoch", -1)
        ) < EPOCHS:
            raise RuntimeError(f"{arm} epoch completion changed")
        for key in ("validation_gap_mae_eV", "mean_epoch_seconds"):
            if not math.isfinite(float(training.get(key, float("nan")))):
                raise RuntimeError(f"{arm} non-finite training evidence: {key}")
        for filename, sha_key in (
            ("best_model.pt", "best_model_sha256"),
            ("last_checkpoint.pt", "checkpoint_sha256"),
        ):
            if sha256_file(root / arm / filename) != training.get(sha_key):
                raise RuntimeError(f"{arm} {filename} SHA changed")
        checkpoint = torch.load(
            root / arm / "last_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        for key, expected in {
            "epoch": EPOCHS - 1,
            "source_commit": source_commit,
            "cache_sha256": cache_sha256,
            "batch_size": 128,
            "precision": PRECISION,
            "loader_workers": LOADER_WORKERS,
            "seed": 42,
        }.items():
            if checkpoint.get(key) != expected:
                raise RuntimeError(f"{arm} checkpoint contract changed: {key}")
        trace = checkpoint.get("trace", [])
        if [row.get("epoch") for row in trace] != list(range(EPOCHS)):
            raise RuntimeError(f"{arm} checkpoint trace is incomplete")
        for row in trace:
            for key in (
                "train_normalized_mae",
                "validation_gap_mae_eV",
                "seconds",
                "learning_rate",
            ):
                if not math.isfinite(float(row.get(key, float("nan")))):
                    raise RuntimeError(f"{arm} non-finite trace value: {key}")
        if checkpoint.get("scheduler", {}).get("last_epoch") != EPOCHS:
            raise RuntimeError(f"{arm} scheduler completion changed")
        for state in checkpoint.get("model", {}).values():
            if torch.is_tensor(state) and not bool(torch.isfinite(state).all()):
                raise RuntimeError(f"{arm} model contains non-finite tensors")
        for state in checkpoint.get("optimizer", {}).get("state", {}).values():
            for value in state.values():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"{arm} optimizer contains non-finite tensors")
        payload = root / arm / "best_validation_payload.pt"
        if sha256_file(payload) != result["training"]["payload_sha256"]:
            raise RuntimeError(f"{arm} payload SHA changed")
        payloads[arm] = torch.load(payload, map_location="cpu", weights_only=False)
    target = payloads["full_gps"]["target_eV"].contiguous()
    source_idx = payloads["full_gps"]["source_idx"].contiguous()
    expected_source_idx = torch.arange(
        500_000, 500_000 + VALIDATION_ROWS, dtype=torch.long
    )
    if not torch.equal(source_idx, expected_source_idx):
        raise RuntimeError("Fixed development source indices changed")
    if target.numel() != VALIDATION_ROWS or not torch.equal(target, payloads["neural_atom_k1"]["target_eV"].contiguous()):
        raise RuntimeError("Scale validation alignment changed")
    for arm, payload in payloads.items():
        for key in ("target_eV", "prediction_eV"):
            value = payload.get(key)
            if value is None or value.numel() != VALIDATION_ROWS or not bool(
                torch.isfinite(value).all()
            ):
                raise RuntimeError(f"{arm} payload is incomplete or non-finite: {key}")
    if not torch.equal(
        source_idx, payloads["neural_atom_k1"]["source_idx"].contiguous()
    ):
        raise RuntimeError("Scale source-index alignment changed")
    errors = {
        arm: (value["prediction_eV"].contiguous() - target).abs()
        for arm, value in payloads.items()
    }
    mae = {arm: float(value.mean()) for arm, value in errors.items()}
    for arm, value in mae.items():
        if abs(value - metrics["validation_gap_mae_eV"][arm]) > 1e-9:
            raise RuntimeError(f"{arm} MAE changed")
    delta = (errors["neural_atom_k1"] - errors["full_gps"]).numpy().astype(np.float64)
    ci = paired_bootstrap(delta)
    gain = mae["full_gps"] - mae["neural_atom_k1"]
    passed = gain >= MIN_GAIN_EV and ci[1] < 0.0
    if list(ci) != metrics["paired_bootstrap_95_ci_k1_minus_full_eV"] or passed != metrics["scale_gate_passed"]:
        raise RuntimeError("Scale decision recomputation changed")
    for relative, expected in completion["artifact_sha256"].items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise RuntimeError(f"Artifact path escaped result root: {relative}") from error
        if sha256_file(path) != expected:
            raise RuntimeError(f"Artifact changed: {relative}")
    return {
        "format": "molgap-pcqm-k1-scale500k-acceptance-v3",
        "accepted": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "fixed_dataset": FIXED_500K_DATASET,
        "fixed_geometry_aggregate_sha256": FIXED_500K_GEOMETRY_SHA256,
        "scnet_aggregate_sha256": SCNET_REFERENCE_CACHE_SHA256,
        "recomputed_validation_gap_mae_eV": mae,
        "recomputed_paired_gain_full_minus_k1_eV": gain,
        "recomputed_paired_bootstrap_95_ci_k1_minus_full_eV": list(ci),
        "scale_gate_passed": passed,
        "historical_tail_batch_policy": "partial-final-batch-32-per-epoch",
        "eligible_as_v4_reference": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "shadow_labels_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = accept(args.root, source_commit=args.source_commit, cache_sha256=args.cache_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
