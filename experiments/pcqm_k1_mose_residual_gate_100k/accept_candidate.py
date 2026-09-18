"""No-inference acceptance for the selective MoSE residual V5 candidate."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from molgap.constants import REPO_ROOT
from molgap.pcqm_k1_variants_runner import (
    FEATURE_FINGERPRINT,
    MINIMUM_GAIN_EV,
    MOSE_RWSE_FEATURE_FINGERPRINT,
    STOCHASTICITY_FLOOR_EV,
)
from molgap.screen_policy import canonical_fingerprint, evaluate_reference_gain
from molgap.training_reproducibility import atomic_json, sha256_file


MODE = "neural_atom_k1_rwse_mose_residual_gate"
EXPECTED_PARAMETERS = {MODE: 3_673_394}
INTERVENTION_KEYS = {
    "run_id",
    "model_id",
    "architecture_fingerprint",
    "source_archive_sha256",
    "result_artifact_sha256",
    "platform_id",
    "accelerator",
    "runtime_certificate_id",
    "feature_fingerprint",
    "frozen_reference",
    "stochasticity_floor_eV",
    "minimum_material_gain_eV",
}


def _shared_module():
    path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("k1_shared_acceptance", path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    return shared


def accept(
    reference_root: Path,
    candidate_root: Path,
    *,
    source_commit: str,
    archive_sha256: str,
) -> dict:
    shared = _shared_module()
    reference, reference_payload = shared._load_arm(
        reference_root, "neural_atom_k1_v4"
    )
    candidate, candidate_payload = shared._load_arm(
        candidate_root,
        MODE,
        expected_parameters=EXPECTED_PARAMETERS,
        initialization_policy="nested-function",
    )
    if candidate.get("source_commit") != source_commit:
        raise RuntimeError("Candidate source commit mismatch")
    architecture = candidate.get("architecture", {})
    if (
        architecture.get("change")
        != "retain-rwse16-and-add-zero-initialized-node-gated-mose31-residual"
        or architecture.get("expected_parameters") != 3_673_394
        or architecture.get("geometry") is not False
        or architecture.get("teacher") is not False
    ):
        raise RuntimeError("Selective residual architecture identity changed")
    checks = candidate["preflight"]["mechanism_checks"]
    required_checks = {
        "rwse_channels": 16,
        "mose_channels": 31,
        "zero_initialized_residual_output": True,
        "exact_k1_output": True,
        "candidate_output_finite": True,
        "residual_trainable_after_two_steps": True,
        "gate_trainable_after_two_steps": True,
    }
    if any(checks.get(key) != value for key, value in required_checks.items()):
        raise RuntimeError("Selective residual mechanism verification missing")
    cache = candidate.get("mose_cache", {})
    if (
        cache.get("aggregate_sha256")
        != "5d949f90a35aea5001d2f6438c916f92cab45d6c5860ba6ccf5777aac1c897e3"
        or cache.get("feature_dim") != 31
    ):
        raise RuntimeError("Accepted MoSE cache identity changed")
    checkpoint_path = candidate_root / MODE / "last_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("source_commit") != source_commit
        or checkpoint.get("source_archive_sha256") != archive_sha256
        or checkpoint.get("epoch") != 39
    ):
        raise RuntimeError("Recoverable checkpoint identity changed")
    for name, digest in checkpoint["best_artifact_sha256"].items():
        if sha256_file(candidate_root / MODE / name) != digest:
            raise RuntimeError(f"Selected artifact mismatch: {name}")
    if not torch.equal(reference_payload["target"], candidate_payload["target"]):
        raise RuntimeError("Development targets differ")
    if not torch.equal(reference_payload["source_idx"], candidate_payload["source_idx"]):
        raise RuntimeError("Development row order differs")
    certificate = candidate["runtime_certificate"]
    required_runtime = {
        "status": "accepted",
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": 128,
        "tail_batch_policy": "drop_last",
        "calibration_checks_passed": True,
    }
    if any(certificate.get(key) != value for key, value in required_runtime.items()):
        raise RuntimeError("Runtime certificate is not V5 decision-grade")
    if canonical_fingerprint(certificate) != candidate["contract"]["runtime_certificate_id"]:
        raise RuntimeError("Runtime certificate identity changed")
    left = reference["contract"]
    right = candidate["contract"]
    mismatches = {
        key: {"reference": left.get(key), "candidate": right.get(key)}
        for key in sorted(set(left) | set(right))
        if key not in INTERVENTION_KEYS and left.get(key) != right.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Undeclared contract mismatch: {mismatches}")
    if left.get("feature_fingerprint") != FEATURE_FINGERPRINT:
        raise RuntimeError("Reference feature identity changed")
    if right.get("feature_fingerprint") != MOSE_RWSE_FEATURE_FINGERPRINT:
        raise RuntimeError("Candidate dual-feature identity changed")
    delta = (
        (candidate_payload["prediction"] - candidate_payload["target"]).abs()
        - (reference_payload["prediction"] - reference_payload["target"]).abs()
    ).numpy().astype(np.float64)
    bootstrap_upper, bootstrap_interval = shared._bootstrap_upper(delta)
    gate = evaluate_reference_gain(
        reference_mae_eV=reference_payload["mae"],
        candidate_mae_eV=candidate_payload["mae"],
        stochasticity_floor_eV=STOCHASTICITY_FLOOR_EV,
        minimum_material_gain_eV=MINIMUM_GAIN_EV,
        paired_row_bootstrap_upper_eV=bootstrap_upper,
    )
    reserve = 1.0 - candidate["training"]["peak_reserved_mib"] / candidate["training"]["total_memory_mib"]
    gate["memory_reserve_fraction"] = reserve
    gate["memory_gate_passed"] = reserve >= 0.15
    gate["passed"] = bool(gate["passed"] and gate["memory_gate_passed"])
    return {
        "format": "molgap-pcqm-k1-mose-residual-gate-acceptance-v1",
        "accepted": True,
        "scientific_gate_passed": gate["passed"],
        "selected_candidate": MODE if gate["passed"] else None,
        "reference_development_gap_mae_eV": reference_payload["mae"],
        "candidate_development_gap_mae_eV": candidate_payload["mae"],
        "paired_error_delta_bootstrap_95_eV": bootstrap_interval,
        "gate": gate,
        "candidate": candidate,
        "source_commit": source_commit,
        "source_archive_sha256": archive_sha256,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(
        args.output,
        accept(
            args.reference_root,
            args.candidate_root,
            source_commit=args.source_commit,
            archive_sha256=args.archive_sha256,
        ),
    )


if __name__ == "__main__":
    main()
