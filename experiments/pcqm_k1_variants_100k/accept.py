"""No-inference joint acceptance for the K1-v4 reference and candidates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from molgap.pcqm_k1_variants_runner import (
    DEVELOPMENT_ROWS,
    EPOCHS,
    FIXED_GEOMETRY_SHA256,
    FIXED_MANIFEST_SHA256,
    MINIMUM_GAIN_EV,
    SAMPLE_EXPOSURE,
    STOCHASTICITY_FLOOR_EV,
)
from molgap.screen_policy import (
    evaluate_reference_gain,
    validate_reference_screen_contract,
)
from molgap.training_reproducibility import atomic_json, sha256_file


EXPECTED_PARAMETERS = {
    "neural_atom_k1_v4": 3_658_817,
    "neural_atom_k1_g": 3_698_180,
    "neural_atom_k1_r": 3_739_841,
    "neural_atom_k4_cluster": 3_658_817,
}


def _load_arm(root: Path, mode: str) -> tuple[dict, dict]:
    arm_root = root / mode
    record = json.loads((arm_root / "arm_record.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (arm_root / "completion_manifest.json").read_text(encoding="utf-8")
    )
    if record.get("format") != "molgap-pcqm-k1-variant-arm-v1" or record.get("complete") is not True:
        raise RuntimeError(f"Incomplete arm record: {mode}")
    if completion.get("format") != "molgap-pcqm-k1-variant-completion-v1" or completion.get("complete") is not True:
        raise RuntimeError(f"Incomplete completion record: {mode}")
    for key, expected in {
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "fixed_geometry_sha256": FIXED_GEOMETRY_SHA256,
        "pure_2d": True,
        "geometry_attributes_removed_before_batching": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }.items():
        if record.get(key) != expected:
            raise RuntimeError(f"Arm contract changed for {mode}: {key}")
    training = record["training"]
    if training.get("parameter_count") != EXPECTED_PARAMETERS[mode]:
        raise RuntimeError(f"Parameter count changed: {mode}")
    if training.get("epochs_completed") != EPOCHS or training.get("sample_presentations") != SAMPLE_EXPOSURE:
        raise RuntimeError(f"Training exposure changed: {mode}")
    preflight = record["preflight"]
    if (
        preflight.get("mode") != mode
        or preflight.get("parameter_count") != EXPECTED_PARAMETERS[mode]
        or preflight.get("exact_k1_function_at_initialization") is not True
        or preflight.get("candidate_mechanism_trainable_after_two_steps") is not True
    ):
        raise RuntimeError(f"Architecture preflight changed: {mode}")
    for relative, expected in completion["artifact_sha256"].items():
        if sha256_file(arm_root / relative) != expected:
            raise RuntimeError(f"Artifact hash changed: {mode}/{relative}")
    payload_path = arm_root / "best_development_payload.pt"
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    target = payload["target_eV"].view(-1).float().contiguous()
    prediction = payload["prediction_eV"].view(-1).float().contiguous()
    source_idx = payload["source_idx"].view(-1).long().contiguous()
    if target.numel() != DEVELOPMENT_ROWS or prediction.shape != target.shape:
        raise RuntimeError(f"Development payload shape changed: {mode}")
    expected_idx = torch.arange(100_000, 150_000, dtype=torch.long)
    if not torch.equal(source_idx, expected_idx):
        raise RuntimeError(f"Development source order changed: {mode}")
    mae = float((prediction - target).abs().mean())
    if abs(mae - float(training["development_gap_mae_eV"])) > 1e-9:
        raise RuntimeError(f"Stored MAE changed: {mode}")
    if sha256_file(payload_path) != record["contract"]["result_artifact_sha256"]:
        raise RuntimeError(f"Result artifact identity changed: {mode}")
    return record, {
        "target": target,
        "prediction": prediction,
        "source_idx": source_idx,
        "mae": mae,
    }


def _bootstrap_upper(delta: np.ndarray, *, seed: int = 20260912) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    means = np.empty(5_000, dtype=np.float64)
    for start in range(0, len(means), 50):
        stop = min(start + 50, len(means))
        indices = rng.integers(0, delta.size, size=(stop - start, delta.size))
        means[start:stop] = delta[indices].mean(axis=1)
    interval = np.quantile(means, [0.025, 0.975]).tolist()
    return float(interval[1]), [float(value) for value in interval]


def accept(reference_root: Path, candidate_root: Path) -> dict:
    reference, reference_payload = _load_arm(
        reference_root, "neural_atom_k1_v4"
    )
    candidates = {}
    for mode in ("neural_atom_k1_g", "neural_atom_k1_r"):
        record, payload = _load_arm(candidate_root, mode)
        if not torch.equal(reference_payload["target"], payload["target"]):
            raise RuntimeError(f"Development targets differ: {mode}")
        if not torch.equal(reference_payload["source_idx"], payload["source_idx"]):
            raise RuntimeError(f"Development row order differs: {mode}")
        certificates = {
            reference["contract"]["runtime_certificate_id"]: reference["runtime_certificate"],
            record["contract"]["runtime_certificate_id"]: record["runtime_certificate"],
        }
        comparability = validate_reference_screen_contract(
            reference=reference["contract"],
            candidate=record["contract"],
            runtime_certificates=certificates,
        )
        delta = (
            (payload["prediction"] - payload["target"]).abs()
            - (reference_payload["prediction"] - reference_payload["target"]).abs()
        ).numpy().astype(np.float64)
        bootstrap_upper, bootstrap_interval = _bootstrap_upper(delta)
        gate = evaluate_reference_gain(
            reference_mae_eV=reference_payload["mae"],
            candidate_mae_eV=payload["mae"],
            stochasticity_floor_eV=STOCHASTICITY_FLOOR_EV,
            minimum_material_gain_eV=MINIMUM_GAIN_EV,
            paired_row_bootstrap_upper_eV=bootstrap_upper,
        )
        reserve = 1.0 - record["training"]["peak_reserved_mib"] / record["training"]["total_memory_mib"]
        gate["memory_reserve_fraction"] = reserve
        gate["memory_gate_passed"] = reserve >= 0.15
        gate["passed"] = bool(gate["passed"] and gate["memory_gate_passed"])
        candidates[mode] = {
            "record": record,
            "recomputed_development_gap_mae_eV": payload["mae"],
            "paired_error_delta_bootstrap_95_eV": bootstrap_interval,
            "comparability": comparability,
            "gate": gate,
        }
    passed = [name for name, value in candidates.items() if value["gate"]["passed"]]
    selected = min(
        passed,
        key=lambda name: candidates[name]["recomputed_development_gap_mae_eV"],
        default=None,
    )
    return {
        "format": "molgap-pcqm-k1-v4-variants-acceptance-v1",
        "accepted": True,
        "reference": reference,
        "reference_development_gap_mae_eV": reference_payload["mae"],
        "candidates": candidates,
        "selected_candidate": selected,
        "minimum_gain_eV": MINIMUM_GAIN_EV,
        "stochasticity_floor_eV": STOCHASTICITY_FLOOR_EV,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = accept(args.reference_root, args.candidate_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
