"""No-inference acceptance against the frozen K1 development payload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from molgap.pcqm_k1_sparse_pair_100k import (
    BASELINE_MAE_EV,
    DEVELOPMENT_ROWS,
    EPOCHS,
    MINIMUM_GAIN_EV,
    MODE,
    PARAMETERS,
    TRAIN_ROWS,
)
from molgap.router import paired_bootstrap_mean
from molgap.training_reproducibility import atomic_json, sha256_file


def _payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"source_idx", "target_eV", "prediction_eV"}
    if not required.issubset(payload):
        raise RuntimeError(f"Prediction payload is incomplete: {path}")
    return {key: payload[key].view(-1).contiguous() for key in required}


def accept(
    candidate_root: Path,
    baseline_payload: Path,
    source_commit: str,
    source_archive_sha256: str,
) -> dict:
    completion_path = candidate_root / "completion_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("complete") is not True:
        raise RuntimeError("Candidate completion is missing")
    if completion.get("source_commit") != source_commit:
        raise RuntimeError("Candidate source commit changed")
    if completion.get("source_archive_sha256") != source_archive_sha256:
        raise RuntimeError("Candidate source archive changed")
    training = completion["training"]
    if training["parameter_count"] != PARAMETERS:
        raise RuntimeError("Candidate parameter count changed")
    if training["epochs_completed"] != EPOCHS:
        raise RuntimeError("Candidate did not complete the frozen exposure")
    for name, digest in completion["artifact_sha256"].items():
        if sha256_file(candidate_root / name) != digest:
            raise RuntimeError(f"Candidate artifact changed: {name}")

    candidate = _payload(candidate_root / "best_development_payload.pt")
    baseline = _payload(baseline_payload)
    expected_source = torch.arange(
        TRAIN_ROWS, TRAIN_ROWS + DEVELOPMENT_ROWS, dtype=torch.long
    )
    if not torch.equal(candidate["source_idx"].long(), expected_source):
        raise RuntimeError("Candidate development identity changed")
    if not torch.equal(baseline["source_idx"].long(), expected_source):
        raise RuntimeError("Baseline development identity changed")
    if not torch.equal(candidate["target_eV"], baseline["target_eV"]):
        raise RuntimeError("Candidate and baseline targets are not identical")
    candidate_error = (
        candidate["prediction_eV"] - candidate["target_eV"]
    ).abs()
    baseline_error = (
        baseline["prediction_eV"] - baseline["target_eV"]
    ).abs()
    candidate_mae = float(candidate_error.mean())
    baseline_mae = float(baseline_error.mean())
    if abs(baseline_mae - BASELINE_MAE_EV) > 1e-6:
        raise RuntimeError(
            f"Frozen baseline scalar changed: {baseline_mae}"
        )
    if abs(candidate_mae - training["development_gap_mae_eV"]) > 1e-7:
        raise RuntimeError("Candidate scalar does not match saved predictions")
    delta = (
        candidate_error.double() - baseline_error.double()
    ).numpy().astype(np.float64)
    bootstrap = paired_bootstrap_mean(delta, n_bootstrap=10_000, seed=42)
    gain = baseline_mae - candidate_mae
    passed = (
        gain >= MINIMUM_GAIN_EV
        and bootstrap["ci95"] is not None
        and bootstrap["ci95"][1] < 0.0
    )
    return {
        "format": "molgap-k1-sparse-pair-100k-acceptance-v1",
        "accepted": True,
        "candidate": MODE,
        "candidate_development_mae_eV": candidate_mae,
        "baseline_k1_development_mae_eV": baseline_mae,
        "gain_vs_k1_eV": gain,
        "minimum_gain_eV": MINIMUM_GAIN_EV,
        "paired_candidate_minus_k1": bootstrap,
        "scientific_gate_passed": passed,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "candidate_completion_sha256": sha256_file(completion_path),
        "baseline_payload_sha256": sha256_file(baseline_payload),
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-payload", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(
        args.output,
        accept(
            args.candidate_root,
            args.baseline_payload,
            args.source_commit,
            args.source_archive_sha256,
        ),
    )
