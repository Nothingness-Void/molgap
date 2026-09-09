"""No-inference acceptance for the matched QM9 GAPE-lite screen."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from molgap.qm9_gape import MIN_GAIN_EV, sha256_file
from molgap.screen_policy import validate_paired_screen_contract


def accept(root: Path, *, source_commit: str, cache_sha256: str) -> dict:
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (root / "completion_manifest.json").read_text(encoding="utf-8")
    )
    required = {
        "format": "molgap-qm9-gape-lite-screen-v1",
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "required_gain_eV": MIN_GAIN_EV,
        "model_inference_executed_by_acceptance": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    for key, value in required.items():
        if metrics.get(key) != value or completion.get(key) != value:
            raise RuntimeError(f"Result contract changed for {key}")
    results = metrics["results"]
    if set(results) != {"baseline", "shuffled_control", "matched_gape"}:
        raise RuntimeError("Unexpected GAPE-lite arm set")
    comparability = validate_paired_screen_contract(
        [results[name]["contract"] for name in results]
    )
    if comparability != metrics["comparability"]:
        raise RuntimeError("Stored comparability report changed")
    recomputed = {}
    artifact_hashes = completion["artifact_sha256"]
    for name, result in results.items():
        training = result["training"]
        payload_path = root / name / "best_validation_payload.pt"
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        value = float((payload["prediction_eV"] - payload["target_eV"]).abs().mean())
        if abs(value - training["validation_gap_mae_eV"]) > 1e-9:
            raise RuntimeError(f"Validation metric changed for {name}")
        if sha256_file(payload_path) != training["payload_sha256"]:
            raise RuntimeError(f"Payload hash changed for {name}")
        recomputed[name] = value
    for relative, expected in artifact_hashes.items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"Artifact hash changed: {relative}")
    gain_baseline = recomputed["baseline"] - recomputed["matched_gape"]
    gain_control = recomputed["shuffled_control"] - recomputed["matched_gape"]
    nominated = gain_baseline >= MIN_GAIN_EV and gain_control >= MIN_GAIN_EV
    if nominated != metrics["pcqm_transfer_nominated"]:
        raise RuntimeError("Stored GAPE-lite decision changed")
    return {
        "format": "molgap-qm9-gape-lite-acceptance-v1",
        "accepted": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "comparability": comparability,
        "recomputed_validation_gap_mae_eV": recomputed,
        "candidate_gain_vs_baseline_eV": gain_baseline,
        "candidate_gain_vs_equal_compute_control_eV": gain_control,
        "pcqm_transfer_nominated": nominated,
        "model_inference_executed": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = accept(
        args.root,
        source_commit=args.source_commit,
        cache_sha256=args.cache_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()

