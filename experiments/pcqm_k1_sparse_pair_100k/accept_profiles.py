"""Select a training platform from two accepted no-science profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.training_reproducibility import atomic_json, sha256_file


def accept(kaggle: Path, kunshan: Path) -> dict:
    records = {
        "kaggle": json.loads(kaggle.read_text(encoding="utf-8")),
        "kunshan": json.loads(kunshan.read_text(encoding="utf-8")),
    }
    for name, record in records.items():
        if (
            record.get("format") != "molgap-k1-sparse-pair-100k-profile-v1"
            or record.get("status") != "accepted"
        ):
            raise RuntimeError(f"{name} profile is not accepted")
        if record["architecture"]["candidate_parameters"] != 3_681_233:
            raise RuntimeError(f"{name} parameter identity changed")
        if not record["architecture"]["exact_nested_initial_predictions"]:
            raise RuntimeError(f"{name} candidate is not nested in K1")
        if record["profile"]["memory_reserve_fraction"] < 0.15:
            raise RuntimeError(f"{name} memory reserve failed")
    invariant_keys = (
        "benchmark_id",
        "fixed_manifest_sha256",
        "fixed_geometry_aggregate_sha256",
        "source_archive_sha256",
        "source_commit",
        "target_stats",
    )
    for key in invariant_keys:
        if records["kaggle"].get(key) != records["kunshan"].get(key):
            raise RuntimeError(f"Cross-platform profile mismatch: {key}")
    platforms = {
        name: record["runtime_certificate"]["platform_id"]
        for name, record in records.items()
    }
    if platforms["kaggle"] == platforms["kunshan"]:
        raise RuntimeError("Profiles do not represent distinct platforms")
    throughput = {
        name: record["profile"]["graphs_per_second"]
        for name, record in records.items()
    }
    winner = max(throughput, key=throughput.get)
    loser = "kunshan" if winner == "kaggle" else "kaggle"
    return {
        "format": "molgap-k1-sparse-pair-profile-selection-v1",
        "accepted": True,
        "selected_platform": platforms[winner],
        "winner_label": winner,
        "graphs_per_second": throughput,
        "speed_ratio": throughput[winner] / throughput[loser],
        "profile_sha256": {
            "kaggle": sha256_file(kaggle),
            "kunshan": sha256_file(kunshan),
        },
        "selection_rule": "higher-identical-contract-bs128-fp32-throughput",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--kunshan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, accept(args.kaggle, args.kunshan))
