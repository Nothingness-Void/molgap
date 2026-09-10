"""No-inference acceptance for the QM9 cardinality-channel screen."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from molgap.qm9_cardinality import (
    MAX_EPOCH_TIME_RATIO,
    MAX_PARAMETER_COUNT,
    MIN_GAIN_VS_BASELINE_EV,
    MIN_GAIN_VS_SIZE_CONTROL_EV,
)
from molgap.qm9_gape import sha256_file
from molgap.screen_policy import validate_paired_screen_contract


def accept(root: Path, *, source_commit: str, cache_sha256: str) -> dict:
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (root / "completion_manifest.json").read_text(encoding="utf-8")
    )
    required = {
        "format": "molgap-qm9-cardinality-channel-screen-v1",
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "required_gain_vs_baseline_eV": MIN_GAIN_VS_BASELINE_EV,
        "required_gain_vs_size_control_eV": MIN_GAIN_VS_SIZE_CONTROL_EV,
        "max_epoch_time_ratio": MAX_EPOCH_TIME_RATIO,
        "model_inference_executed_by_acceptance": False,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    for key, value in required.items():
        if metrics.get(key) != value or completion.get(key) != value:
            raise RuntimeError(f"Result contract changed for {key}")
    results = metrics["results"]
    if set(results) != {"baseline", "size_control", "cpa"}:
        raise RuntimeError("Unexpected cardinality arm set")
    comparability = validate_paired_screen_contract(
        [results[name]["contract"] for name in results]
    )
    if comparability != metrics["comparability"]:
        raise RuntimeError("Stored comparability report changed")
    preflight = metrics["preflight"]
    if (
        preflight.get("accepted") is not True
        or preflight.get("max_hops") != 3
        or preflight.get("channel_layers") != [3, 6, 9]
        or preflight.get("chain_support_exact") is not True
        or preflight.get("test_role_read") is not False
    ):
        raise RuntimeError("Remote architecture preflight changed")
    recomputed = {}
    for name, result in results.items():
        training = result["training"]
        if training["parameter_count"] > MAX_PARAMETER_COUNT:
            raise RuntimeError(f"Parameter ceiling failed for {name}")
        payload_path = root / name / "best_validation_payload.pt"
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        value = float((payload["prediction_eV"] - payload["target_eV"]).abs().mean())
        if abs(value - training["validation_gap_mae_eV"]) > 1e-9:
            raise RuntimeError(f"Validation metric changed for {name}")
        if sha256_file(payload_path) != training["payload_sha256"]:
            raise RuntimeError(f"Payload hash changed for {name}")
        recomputed[name] = value
    for relative, expected in completion["artifact_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"Artifact hash changed: {relative}")
    gain_baseline = recomputed["baseline"] - recomputed["cpa"]
    gain_size = recomputed["size_control"] - recomputed["cpa"]
    epoch_ratio = (
        results["cpa"]["training"]["mean_epoch_seconds"]
        / results["baseline"]["training"]["mean_epoch_seconds"]
    )
    nominated = (
        gain_baseline >= MIN_GAIN_VS_BASELINE_EV
        and gain_size >= MIN_GAIN_VS_SIZE_CONTROL_EV
        and epoch_ratio <= MAX_EPOCH_TIME_RATIO
        and results["cpa"]["training"]["memory_reserve_fraction"] >= 0.15
    )
    if nominated != metrics["pcqm_transfer_nominated"]:
        raise RuntimeError("Stored cardinality decision changed")
    return {
        "format": "molgap-qm9-cardinality-channel-acceptance-v1",
        "accepted": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha256,
        "comparability": comparability,
        "recomputed_validation_gap_mae_eV": recomputed,
        "cpa_gain_vs_baseline_eV": gain_baseline,
        "cpa_gain_vs_size_control_eV": gain_size,
        "cpa_epoch_time_ratio": epoch_ratio,
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
