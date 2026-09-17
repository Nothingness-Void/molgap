"""Mechanically accept a downloaded same-allocation repeatability result."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_CACHE = "bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    payload = json.loads((args.root / "selection.json").read_text(encoding="utf-8"))
    required = {
        "format": "molgap-pcqm-scratch-repeatability-s42-v1",
        "complete": True,
        "source_commit": "c2302c7d433dfec73d293cdbd71ee46aa9e4cae5",
        "seed": 42,
        "batch_size": 48,
        "epochs": 60,
        "parameter_count": 3_665_809,
        "geometry_cache_aggregate_sha256": EXPECTED_CACHE,
        "architecture_ranking_performed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    errors = [
        f"{key}: {payload.get(key)!r} != {expected!r}"
        for key, expected in required.items()
        if payload.get(key) != expected
    ]
    metric_keys = (
        "repeat_a_best_40_gap_mae_eV",
        "repeat_b_best_40_gap_mae_eV",
        "absolute_best_40_drift_eV",
        "repeat_a_best_60_gap_mae_eV",
        "repeat_b_best_60_gap_mae_eV",
        "absolute_best_60_drift_eV",
    )
    errors.extend(
        f"non-finite metric: {key}"
        for key in metric_keys
        if not math.isfinite(float(payload.get(key, float("nan"))))
    )
    initial_hash = payload.get("initial_encoder_sha256")
    if not isinstance(initial_hash, str) or len(initial_hash) != 64:
        errors.append("invalid shared initialization SHA-256")
    result = {"accepted": not errors, "errors": errors, "selection": payload}
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
