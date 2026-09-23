"""Recompute the PairNorm versus frozen GPTrans development comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(candidate: Path, reference: Path) -> dict:
    candidate_manifest = json.loads((candidate / "stage_manifest.json").read_bytes())
    reference_manifest = json.loads((reference / "stage_manifest.json").read_bytes())
    if candidate_manifest["arm"] != "gptrans_pair_update_norm":
        raise ValueError("Wrong candidate arm")
    if reference_manifest["arm"] != "gptrans":
        raise ValueError("Wrong reference arm")
    if candidate_manifest["status"] != reference_manifest["status"] != "COMPLETE":
        raise ValueError("Comparison requires two completed arms")
    if candidate_manifest["contract"]["benchmark_id"] != reference_manifest["contract"]["benchmark_id"]:
        raise ValueError("Benchmark identity differs")
    for manifest in (candidate_manifest, reference_manifest):
        for role in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
            if manifest[role] is not False:
                raise ValueError(f"Protected role accessed: {role}")
    payloads = []
    for folder, manifest in ((candidate, candidate_manifest), (reference, reference_manifest)):
        path = folder / "best_predictions.pt"
        if sha256(path) != manifest["artifacts"]["best_predictions.pt"]:
            raise ValueError("Prediction hash differs from completion manifest")
        payloads.append(torch.load(path, map_location="cpu", weights_only=True))
    candidate_payload, reference_payload = payloads
    expected_rows = torch.arange(500000, 550000)
    if not torch.equal(candidate_payload["source_idx"], expected_rows):
        raise ValueError("Candidate rows differ from frozen development role")
    if not torch.equal(reference_payload["source_idx"], expected_rows):
        raise ValueError("Reference rows differ from frozen development role")
    if not torch.equal(candidate_payload["target"], reference_payload["target"]):
        raise ValueError("Development targets differ")
    for payload in payloads:
        if not all(bool(torch.isfinite(payload[key]).all()) for key in ("prediction", "target")):
            raise ValueError("Predictions or targets are nonfinite")
    candidate_error = (candidate_payload["prediction"] - candidate_payload["target"]).abs().double().numpy()
    reference_error = (reference_payload["prediction"] - reference_payload["target"]).abs().double().numpy()
    improvement = reference_error - candidate_error
    if abs(float(candidate_error.mean()) - candidate_manifest["best_development_mae_eV"]) > 1e-6:
        raise ValueError("Candidate MAE differs from completion manifest")
    if abs(float(reference_error.mean()) - reference_manifest["best_development_mae_eV"]) > 1e-6:
        raise ValueError("Reference MAE differs from completion manifest")
    rng = np.random.default_rng(20260923)
    bootstrap = np.empty(10000, dtype=np.float64)
    for start in range(0, len(bootstrap), 100):
        rows = rng.integers(0, len(improvement), size=(100, len(improvement)))
        bootstrap[start:start + 100] = improvement[rows].mean(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "format": "molgap-pair-norm-500k-paired-comparison-v1",
        "candidate_manifest_sha256": sha256(candidate / "stage_manifest.json"),
        "reference_manifest_sha256": sha256(reference / "stage_manifest.json"),
        "candidate_predictions_sha256": sha256(candidate / "best_predictions.pt"),
        "reference_predictions_sha256": sha256(reference / "best_predictions.pt"),
        "rows": len(improvement),
        "source_idx_and_target_alignment": True,
        "candidate_mae_eV": float(candidate_error.mean()),
        "reference_mae_eV": float(reference_error.mean()),
        "reference_minus_candidate_mae_eV": float(improvement.mean()),
        "bootstrap_seed": 20260923,
        "bootstrap_replicates": len(bootstrap),
        "row_paired_bootstrap_95pct_eV": interval.tolist(),
        "nomination_floor_eV": 0.003,
        "nominated_under_frozen_point_gate": bool(improvement.mean() >= 0.003),
        "uncertainty_scope": "development rows only; does not measure training stochasticity",
        "protected_roles": "untouched",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.candidate, args.reference)
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if args.output:
        args.output.write_bytes(encoded)
    print(encoded.decode(), end="")
