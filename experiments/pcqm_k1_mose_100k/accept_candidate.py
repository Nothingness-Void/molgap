"""No-inference acceptance for the K1-MoSE V5 candidate."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from molgap.constants import REPO_ROOT
from molgap.pcqm_k1_variants_runner import (
    FEATURE_FINGERPRINT,
    MINIMUM_GAIN_EV,
    MOSE_FEATURE_FINGERPRINT,
    STOCHASTICITY_FLOOR_EV,
)
from molgap.screen_policy import canonical_fingerprint, evaluate_reference_gain
from molgap.training_reproducibility import atomic_json, sha256_file


MODE = "neural_atom_k1_mose"
EXPECTED_PARAMETERS = {MODE: 3_661_697}
INITIALIZATION_POLICY = "feature-replacement-shared-k1-state"
CONTRACT_INTERVENTION_KEYS = {
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


def _check_runtime(record: dict) -> None:
    certificate = record["runtime_certificate"]
    contract = record["contract"]
    if (
        certificate.get("status") != "accepted"
        or certificate.get("precision") != "fp32"
        or certificate.get("tf32_enabled") is not False
        or certificate.get("deterministic_algorithms") is not True
        or certificate.get("physical_batch_per_device") != 128
        or certificate.get("tail_batch_policy") != "drop_last"
        or certificate.get("calibration_checks_passed") is not True
    ):
        raise RuntimeError("Runtime certificate is not V5 decision-grade")
    if canonical_fingerprint(certificate) != contract["runtime_certificate_id"]:
        raise RuntimeError("Runtime certificate identity changed")


def _check_declared_contract(reference: dict, candidate: dict) -> dict:
    reference_contract = reference["contract"]
    candidate_contract = candidate["contract"]
    mismatches = {}
    for key in sorted(set(reference_contract) | set(candidate_contract)):
        if key in CONTRACT_INTERVENTION_KEYS:
            continue
        if reference_contract.get(key) != candidate_contract.get(key):
            mismatches[key] = {
                "reference": reference_contract.get(key),
                "candidate": candidate_contract.get(key),
            }
    if mismatches:
        raise RuntimeError(f"Undeclared contract mismatch: {mismatches}")
    if reference_contract.get("feature_fingerprint") != FEATURE_FINGERPRINT:
        raise RuntimeError("Frozen reference feature identity changed")
    if candidate_contract.get("feature_fingerprint") != MOSE_FEATURE_FINGERPRINT:
        raise RuntimeError("Candidate MoSE feature identity changed")
    return {
        "decision_grade": True,
        "declared_intervention": "rwse16-to-mose31-replacement",
        "all_nonintervention_contract_fields_equal": True,
        "reference_runtime_certificate_accepted": True,
        "candidate_runtime_certificate_accepted": True,
    }


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
        initialization_policy=INITIALIZATION_POLICY,
    )
    if candidate.get("source_commit") != source_commit:
        raise RuntimeError("Candidate source commit mismatch")
    architecture = candidate.get("architecture", {})
    if (
        architecture.get("change") != "replace-rwse16-with-rooted-mose31"
        or architecture.get("structural_channels") != 31
        or architecture.get("geometry") is not False
        or architecture.get("teacher") is not False
    ):
        raise RuntimeError("MoSE architecture identity changed")
    cache = candidate.get("mose_cache", {})
    if (
        cache.get("aggregate_sha256")
        != "5d949f90a35aea5001d2f6438c916f92cab45d6c5860ba6ccf5777aac1c897e3"
        or cache.get("feature_dim") != 31
        or cache.get("feature_transform_at_training") != "log1p-float32"
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
    for key in ("optimizer", "scheduler", "rng_state", "best", "best_epoch", "best_artifact_sha256"):
        if key not in checkpoint:
            raise RuntimeError(f"Recoverable checkpoint field missing: {key}")
    for name, digest in checkpoint["best_artifact_sha256"].items():
        if sha256_file(candidate_root / MODE / name) != digest:
            raise RuntimeError(f"Selected artifact mismatch: {name}")
    if not torch.equal(reference_payload["target"], candidate_payload["target"]):
        raise RuntimeError("Development targets differ")
    if not torch.equal(reference_payload["source_idx"], candidate_payload["source_idx"]):
        raise RuntimeError("Development row order differs")
    _check_runtime(reference)
    _check_runtime(candidate)
    comparability = _check_declared_contract(reference, candidate)
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
        "format": "molgap-pcqm-k1-mose-acceptance-v1",
        "accepted": True,
        "scientific_gate_passed": gate["passed"],
        "selected_candidate": MODE if gate["passed"] else None,
        "reference_development_gap_mae_eV": reference_payload["mae"],
        "candidate_development_gap_mae_eV": candidate_payload["mae"],
        "paired_error_delta_bootstrap_95_eV": bootstrap_interval,
        "comparability": comparability,
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
