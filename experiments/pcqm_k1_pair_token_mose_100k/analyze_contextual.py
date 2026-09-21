"""Verify saved endpoints and classify the unplanned feature mismatch.

This is artifact-only analysis: no encoder construction, model inference, or
protected-role access. The frozen strict acceptance must continue to fail.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from molgap.constants import REPO_ROOT
from molgap.pcqm_k1_variants_runner import (
    FEATURE_FINGERPRINT,
    MOSE_RWSE_FEATURE_FINGERPRINT,
)
from molgap.screen_policy import REFERENCE_MATCH_FIELDS, validate_runtime_certificate
from molgap.training_reproducibility import atomic_json, sha256_file


MODE = "neural_atom_k1_pair_token_mose"
EXPECTED_PARAMETERS = {MODE: 3_696_193}
SOURCE_COMMIT = "246d89c94fa65752b2f54633630346e1c67effd7"
SOURCE_ARCHIVE_SHA256 = "6a5fd81792c71d9986ef6ee5937c74542e6c3e5ac74a12bd8a6bc4e8c72a671b"


def _shared_accept():
    path = REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py"
    spec = importlib.util.spec_from_file_location("k1_shared_artifact_reader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _paired_interval(candidate: dict, reference: dict, shared) -> dict:
    delta = (
        (candidate["prediction"] - candidate["target"]).abs()
        - (reference["prediction"] - reference["target"]).abs()
    ).numpy().astype(np.float64)
    upper, interval = shared._bootstrap_upper(delta)
    return {
        "candidate_minus_reference_mae_eV": float(delta.mean()),
        "paired_row_bootstrap_95_eV": interval,
        "paired_upper_eV": upper,
    }


def analyze(reference_root: Path, parent_root: Path, candidate_root: Path) -> dict:
    shared = _shared_accept()
    reference, ref_payload = shared._load_arm(reference_root, "neural_atom_k1_v4")
    parent, parent_payload = shared._load_arm(
        parent_root, "neural_atom_k1_pair_token",
        expected_parameters={"neural_atom_k1_pair_token": 3_681_665},
    )
    candidate, cand_payload = shared._load_arm(
        candidate_root, MODE, expected_parameters=EXPECTED_PARAMETERS
    )
    if candidate["source_commit"] != SOURCE_COMMIT:
        raise RuntimeError("Candidate source commit differs from frozen release")
    if candidate["contract"]["source_archive_sha256"] != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("Candidate source archive differs from frozen release")
    if candidate["training"]["best_epoch"] != 39:
        raise RuntimeError("Unexpected selected endpoint")
    for item in (parent_payload, cand_payload):
        if not torch.equal(ref_payload["target"], item["target"]):
            raise RuntimeError("Development targets are not identical")
        if not torch.equal(ref_payload["source_idx"], item["source_idx"]):
            raise RuntimeError("Development row order is not identical")
    for arm in (reference, parent, candidate):
        validate_runtime_certificate(arm["runtime_certificate"], arm["contract"])
    ref_contract = reference["contract"]
    cand_contract = candidate["contract"]
    mismatches = {
        field: {
            "reference": ref_contract[field],
            "candidate": cand_contract[field],
        }
        for field in REFERENCE_MATCH_FIELDS
        if ref_contract[field] != cand_contract[field]
    }
    if mismatches != {
        "feature_fingerprint": {
            "reference": FEATURE_FINGERPRINT,
            "candidate": MOSE_RWSE_FEATURE_FINGERPRINT,
        }
    }:
        raise RuntimeError(f"Unexpected contract mismatch: {mismatches}")
    checks = candidate["preflight"]["mechanism_checks"]
    required = (
        "exact_k1_output",
        "mose_residual_trainable_after_two_steps",
        "pair_token_trainable_after_two_steps",
        "mose_input_finite",
        "mose_input_nonnegative",
        "pairtoken_parent_unchanged",
    )
    if any(checks.get(key) is not True for key in required):
        raise RuntimeError("Mechanism preflight is incomplete")
    root = candidate_root / MODE
    checkpoint = torch.load(root / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 39 or checkpoint.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("Final checkpoint identity mismatch")
    for name, digest in checkpoint["best_artifact_sha256"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"Selected artifact hash mismatch: {name}")
    return {
        "format": "molgap-k1-pair-token-mose-contextual-endpoint-v1",
        "mechanical_artifacts_accepted": True,
        "strict_reference_screen_accepted": False,
        "comparison_class": "PAIRED_ENDPOINT",
        "strict_causal_claim_allowed": False,
        "blocker_codes": ["SCIENTIFIC_CONTRACT_MISMATCH"],
        "reason": "MoSE31 added a feature intervention that the frozen architecture-only prelaunch did not declare.",
        "mismatched_fields": mismatches,
        "frozen_prelaunch_feature_identity": FEATURE_FINGERPRINT,
        "observed_candidate_feature_identity": MOSE_RWSE_FEATURE_FINGERPRINT,
        "development_rows": len(cand_payload["target"]),
        "development_role_reused_for_selection": True,
        "reference_mae_eV": ref_payload["mae"],
        "pair_token_parent_mae_eV": parent_payload["mae"],
        "candidate_mae_eV": cand_payload["mae"],
        "candidate_vs_reference": _paired_interval(cand_payload, ref_payload, shared),
        "candidate_vs_pair_token_parent": _paired_interval(cand_payload, parent_payload, shared),
        "candidate_parameters": candidate["training"]["parameter_count"],
        "best_epoch": candidate["training"]["best_epoch"],
        "optimizer_steps": candidate["training"]["optimizer_steps"],
        "sample_presentations": candidate["training"]["sample_presentations"],
        "mean_graphs_per_second": candidate["training"]["mean_graphs_per_second"],
        "peak_reserved_mib": candidate["training"]["peak_reserved_mib"],
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "payload_sha256": sha256_file(root / "best_development_payload.pt"),
        "checkpoint_sha256": sha256_file(root / "last_checkpoint.pt"),
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "promotion_gate_evaluated": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, analyze(args.reference_root, args.parent_root, args.candidate_root))
