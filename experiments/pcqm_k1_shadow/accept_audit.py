"""Recompute the completed K1 shadow decision without model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from molgap.pcqm_k1_shadow_audit import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_SHADOW_CACHE_SHA256,
    MAX_INFERENCE_TIME_RATIO,
    MIN_MEMORY_RESERVE,
    SHADOW_ROWS,
    paired_bootstrap,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import numpy as np
    import torch

    metrics_path = args.root / "metrics.json"
    completion_path = args.root / "completion_manifest.json"
    payload_path = args.root / "audit_payload.pt"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    target = payload["target_eV"].view(-1).float()
    full = payload["full_gps_prediction_eV"].view(-1).float()
    k1 = payload["neural_atom_k1_prediction_eV"].view(-1).float()
    delta = ((k1 - target).abs() - (full - target).abs()).numpy().astype(np.float64)
    ci_low, ci_high = paired_bootstrap(delta)
    full_mae = float((full - target).abs().mean())
    k1_mae = float((k1 - target).abs().mean())
    time_ratio = float(metrics["k1_inference_time_ratio"])
    reserve = [
        float(metrics["inference"][name]["memory_reserve_fraction"])
        for name in ("full_gps", "neural_atom_k1")
    ]
    passed = (
        k1_mae < full_mae
        and ci_high < 0.0
        and time_ratio <= MAX_INFERENCE_TIME_RATIO
        and min(reserve) >= MIN_MEMORY_RESERVE
    )
    artifact_checks = {
        name: sha256_file(args.root / name) == expected
        for name, expected in completion["artifact_sha256"].items()
    }
    checks = {
        "complete": metrics.get("complete") is True,
        "source_commit": metrics.get("source_commit") == args.source_commit,
        "cache": metrics.get("cache_aggregate_sha256")
        == EXPECTED_SHADOW_CACHE_SHA256,
        "rows": target.numel() == full.numel() == k1.numel() == SHADOW_ROWS,
        "finite": bool(torch.isfinite(target).all())
        and bool(torch.isfinite(full).all())
        and bool(torch.isfinite(k1).all()),
        "label_access": metrics.get("shadow_labels_read") is True
        and metrics.get("shadow_label_values_accessed") == SHADOW_ROWS
        and payload.get("shadow_label_values_accessed") == SHADOW_ROWS,
        "no_training": metrics.get("training_executed") is False,
        "official_validation_sealed": metrics.get("official_validation_role_read")
        is False,
        "test_dev_sealed": metrics.get("test_dev_role_read") is False,
        "bootstrap_contract": metrics.get("bootstrap_seed") == BOOTSTRAP_SEED
        and metrics.get("bootstrap_replicates") == BOOTSTRAP_REPLICATES,
        "mae_match": abs(metrics["shadow_gap_mae_eV"]["full_gps"] - full_mae) < 1e-9
        and abs(metrics["shadow_gap_mae_eV"]["neural_atom_k1"] - k1_mae) < 1e-9,
        "ci_match": np.allclose(
            metrics["paired_bootstrap_95_ci_eV"], [ci_low, ci_high], atol=1e-12
        ),
        "decision_match": metrics.get("k1_shadow_passed") is passed,
        "artifacts": all(artifact_checks.values()),
    }
    result = {
        "format": "molgap-pcqm-k1-shadow-audit-acceptance-v1",
        "accepted": all(checks.values()),
        "checks": checks,
        "artifact_checks": artifact_checks,
        "recomputed_shadow_gap_mae_eV": {
            "full_gps": full_mae,
            "neural_atom_k1": k1_mae,
        },
        "recomputed_paired_delta_k1_minus_full_eV": k1_mae - full_mae,
        "recomputed_paired_bootstrap_95_ci_eV": [ci_low, ci_high],
        "k1_shadow_passed": passed,
        "model_inference_executed_by_acceptance": False,
        "training_executed_by_acceptance": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    output = args.output or args.root / "acceptance.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
