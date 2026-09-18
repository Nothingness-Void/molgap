"""Mechanical no-inference acceptance for the GPTrans path profile."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from molgap.pcqm_gptrans_shortest_path_profile import (
    BATCH_SIZE,
    COHORTS,
    FORMAT,
    MAX_EQUIVALENCE_DELTA,
    REPEATS,
)
from molgap.pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
from molgap.training_reproducibility import atomic_json, sha256_file


def accept(root: Path) -> dict:
    root = root.resolve()
    completion = root / "completion_manifest.json"
    payload = json.loads(completion.read_text(encoding="utf-8"))
    failures = []
    expected = {
        "format": FORMAT,
        "status": "complete",
        "profiling_only": True,
        "scientific_result_produced": False,
        "training_checkpoint_produced": False,
        "model_bundle_produced": False,
        "scientific_contract_changed": False,
        "development_role_read": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "train_role_read": True,
        "cache_manifest_sha256": FIXED_500K_MANIFEST_SHA256,
        "physical_batch_per_device": BATCH_SIZE,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            failures.append(f"{key}: expected {value!r}, got {payload.get(key)!r}")
    rows = payload.get("results", [])
    if [row.get("cohort") for row in rows] != list(COHORTS):
        failures.append("Cohort order or identity changed")
    representative_complete = False
    for row in rows:
        if row.get("status") == "oom":
            continue
        if row.get("status") != "complete":
            failures.append(f"Invalid cohort status: {row.get('status')}")
            continue
        if row.get("cohort") == "representative":
            representative_complete = True
        if len(row.get("repeats", [])) != REPEATS:
            failures.append(f"Repeat count changed for {row.get('cohort')}")
        for repeat in row.get("repeats", []):
            for variant in ("dynamic", "cached"):
                value = repeat.get(variant, {}).get("graphs_per_second")
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    failures.append(f"Invalid throughput: {row.get('cohort')}/{variant}")
            equivalence = repeat.get("equivalence", {})
            for name in (
                "maximum_loss_delta",
                "maximum_model_delta",
                "maximum_ema_delta",
            ):
                value = equivalence.get(name)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    failures.append(f"Invalid equivalence delta: {name}")
            if equivalence.get("maximum_allowed_delta") != MAX_EQUIVALENCE_DELTA:
                failures.append("Equivalence tolerance changed")
    if not representative_complete:
        failures.append("Representative cohort did not complete")
    for name, expected_sha in payload.get("artifact_sha256", {}).items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected_sha:
            failures.append(f"Artifact hash mismatch: {name}")
    result = {
        "accepted": not failures,
        "v5_outcome": {
            "execution_status": "COMPLETE" if not failures else "FAILED",
            "artifact_status": "ACCEPTED" if not failures else "REJECTED",
            "comparison_status": "NOT_APPLICABLE",
            "scientific_status": "NOT_APPLICABLE",
            "transfer_status": "NOT_READY",
            "budget_decision": "PROFILING_ONLY",
            "full_handoff_status": "NONE"
        },
        "model_inference_executed": False,
        "scientific_result_accepted": False,
        "scientific_contract_change_authorized": False,
        "exact_implementation_followup_eligible": payload.get(
            "exact_cached_shortest_path_followup_eligible"
        ),
        "failures": failures,
        "source_commit": payload.get("source_commit"),
        "slurm_job_id": payload.get("slurm_job_id"),
        "completion_manifest_sha256": sha256_file(completion),
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
