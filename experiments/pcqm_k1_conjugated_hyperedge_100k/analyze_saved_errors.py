"""Exploratory row-aligned component attribution from saved development payloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from molgap.k1_conjugated_sidecar import accept_sidecar
from molgap.training_reproducibility import atomic_json, sha256_file


MODES = (
    "neural_atom_k1_conjugated_oneshot",
    "neural_atom_k1_conjugated_persistent",
)
SOURCE_COMMIT = "b2340edd93bff46172959bdc2f9771da51fad3d6"


def payload(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    source = value["source_idx"].view(-1).long().contiguous()
    target = value["target_eV"].view(-1).float().contiguous()
    prediction = value["prediction_eV"].view(-1).float().contiguous()
    if (
        not torch.equal(source, torch.arange(100_000, 150_000))
        or target.shape != prediction.shape
        or not torch.isfinite(target).all()
        or not torch.isfinite(prediction).all()
    ):
        raise RuntimeError(f"Unaligned/nonfinite development payload: {path}")
    return {"source": source, "target": target, "prediction": prediction}


def describe(base_error: torch.Tensor, candidate_error: torch.Tensor, mask: torch.Tensor) -> dict:
    before = base_error[mask]
    after = candidate_error[mask]
    delta = before - after
    wins = delta > 0
    losses = delta < 0
    return {
        "rows": int(mask.sum()),
        "reference_mae_eV": float(before.mean()) if before.numel() else None,
        "candidate_mae_eV": float(after.mean()) if after.numel() else None,
        "candidate_gain_eV": float(delta.mean()) if delta.numel() else None,
        "win_fraction": float(wins.float().mean()) if delta.numel() else None,
        "mean_winning_margin_eV": float(delta[wins].mean()) if bool(wins.any()) else None,
        "mean_losing_margin_eV": float((-delta[losses]).mean()) if bool(losses.any()) else None,
    }


def analyze(reference_root: Path, candidate_root: Path, sidecar_root: Path) -> dict:
    sidecar_acceptance = accept_sidecar(
        sidecar_root, expected_source_commit=SOURCE_COMMIT
    )
    sidecar = json.loads((sidecar_root / "manifest.json").read_text(encoding="utf-8"))
    counts = []
    for shard in sidecar["shards"]:
        if shard["role"] != "development":
            continue
        rows = torch.load(sidecar_root / shard["file"], map_location="cpu", weights_only=False)
        counts.extend(int(row["component_count"]) for row in rows)
    count_tensor = torch.tensor(counts, dtype=torch.long)
    if count_tensor.numel() != 50_000:
        raise RuntimeError("Development sidecar coverage changed")
    masks = {
        "no_conjugated_component": count_tensor == 0,
        "one_conjugated_component": count_tensor == 1,
        "multiple_conjugated_components": count_tensor >= 2,
    }
    reference_path = reference_root / "neural_atom_k1_v4" / "best_development_payload.pt"
    reference = payload(reference_path)
    base_error = (reference["prediction"] - reference["target"]).abs()
    results = {}
    for mode in MODES:
        path = candidate_root / mode / "best_development_payload.pt"
        candidate = payload(path)
        if not torch.equal(candidate["target"], reference["target"]):
            raise RuntimeError(f"Development targets changed for {mode}")
        error = (candidate["prediction"] - candidate["target"]).abs()
        results[mode] = {
            "payload_sha256": sha256_file(path),
            "all_rows": describe(base_error, error, torch.ones(50_000, dtype=torch.bool)),
            "by_conjugated_component_count": {
                name: describe(base_error, error, mask) for name, mask in masks.items()
            },
        }
    return {
        "format": "molgap-k1-conjugated-saved-error-attribution-v1",
        "source_commit": SOURCE_COMMIT,
        "sidecar_aggregate_sha256": sidecar_acceptance["aggregate_sha256"],
        "reference_payload_sha256": sha256_file(reference_path),
        "exploratory_same_development_split": True,
        "independent_holdout": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "candidates": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, analyze(args.reference_root, args.candidate_root, args.sidecar_root))
