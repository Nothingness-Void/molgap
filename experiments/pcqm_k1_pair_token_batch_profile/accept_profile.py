"""Mechanical, no-inference acceptance for the PairToken batch profile."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from molgap.pcqm_k1_pair_token_batch_profile import (
    BATCH_SIZES,
    FORMAT,
    MIN_MEMORY_RESERVE_FRACTION,
)
from molgap.pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
from molgap.training_reproducibility import atomic_json, sha256_file


def accept(root: Path) -> dict:
    root = root.resolve()
    completion_path = root / "completion_manifest.json"
    payload = json.loads(completion_path.read_text(encoding="utf-8"))
    failures = []
    expected = {
        "format": FORMAT,
        "status": "complete",
        "profiling_only": True,
        "scientific_result_produced": False,
        "training_checkpoint_produced": False,
        "model_bundle_produced": False,
        "scientific_contract_changed": False,
        "validation_executed": False,
        "development_role_read": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "train_role_read": True,
        "cache_manifest_sha256": FIXED_500K_MANIFEST_SHA256,
        "train_rows": 500_000,
        "batch_sizes": list(BATCH_SIZES),
        "minimum_memory_reserve_fraction": MIN_MEMORY_RESERVE_FRACTION,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            failures.append(f"{key}: expected {value!r}, got {payload.get(key)!r}")
    rows = payload.get("results", [])
    if len(rows) != len(BATCH_SIZES) * 2:
        failures.append(f"expected {len(BATCH_SIZES) * 2} result rows, got {len(rows)}")
    seen = set()
    for row in rows:
        key = (row.get("batch_size"), row.get("cohort"))
        seen.add(key)
        if row.get("status") not in {"complete", "oom"}:
            failures.append(f"invalid status for {key}: {row.get('status')}")
        for name in (
            "graphs_per_second_compute",
            "graphs_per_second_end_to_end",
        ):
            value = row.get(name)
            if row.get("status") == "complete" and (
                not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            ):
                failures.append(f"invalid {name} for {key}: {value}")
        reserve = row.get("memory", {}).get("reserve_fraction_from_peak_reserved")
        if not isinstance(reserve, (int, float)) or not math.isfinite(reserve):
            failures.append(f"invalid memory reserve for {key}: {reserve}")
    expected_seen = {
        (batch, cohort)
        for batch in BATCH_SIZES
        for cohort in ("representative", "graph_size_tail")
    }
    if seen != expected_seen:
        failures.append(f"batch/cohort grid changed: {sorted(seen)}")
    for name, expected_sha in payload.get("artifact_sha256", {}).items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected_sha:
            failures.append(f"artifact hash mismatch: {name}")
    result = {
        "accepted": not failures,
        "v5_outcome": {
            "execution_status": "COMPLETE" if not failures else "FAILED",
            "artifact_status": "ACCEPTED" if not failures else "REJECTED",
            "comparison_status": "NOT_APPLICABLE",
            "scientific_status": "NOT_APPLICABLE",
            "transfer_status": "NOT_READY",
            "budget_decision": "PROFILING_ONLY",
            "full_handoff_status": "NONE",
        },
        "v5_profiling_coverage": "PARTIAL",
        "v5_missing_stage_evidence": [
            "validation_or_train-role-evaluation_timing",
            "real_model_checkpoint_hash_archive_timing",
            "repeated_order-randomized_batch_measurements",
        ],
        "model_inference_executed": False,
        "scientific_result_accepted": False,
        "scientific_contract_change_authorized": False,
        "failures": failures,
        "source_commit": payload.get("source_commit"),
        "slurm_job_id": payload.get("slurm_job_id"),
        "recommendations_for_future_contract_study_only": payload.get(
            "recommendations_for_future_contract_study_only"
        ),
        "completion_manifest_sha256": sha256_file(completion_path),
    }
    atomic_json(root / "acceptance.json", result)
    if failures:
        raise RuntimeError("; ".join(failures))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(accept(args.root), indent=2))


if __name__ == "__main__":
    main()
