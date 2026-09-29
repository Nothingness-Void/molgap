"""No-Torch acceptance of Kaggle2 real-input initialization-scale output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from molgap.gptrans_initial_scale_preflight import (
    INITIAL_MODEL_SHA256,
    INITIAL_STATE_SHA256,
    MANIFEST_SHA256,
)


ENERGY_KEYS = ("atom_squared_sum", "degree_squared_sum", "cross_sum")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify(output: Path, manifest_path: Path, *, source_commit: str, source_archive_sha256: str) -> dict:
    if digest(manifest_path) != MANIFEST_SHA256:
        raise ValueError("accepted fixed manifest changed")
    records = [item for item in read(manifest_path)["geometry_shards"] if item["role"] == "train"]
    if len(records) != 2:
        raise ValueError("expected two fixed train shards")
    summary = read(output / "summary.json")
    if (summary.get("format") != "molgap-gptrans-initial-scale-preflight-v1"
            or summary.get("status") != "COMPLETE"
            or summary.get("manifest_sha256") != MANIFEST_SHA256
            or summary.get("initial_state_sha256") != INITIAL_STATE_SHA256
            or summary.get("initial_model_sha256") != INITIAL_MODEL_SHA256
            or summary.get("sample_per_shard") != 256
            or summary.get("labels_read") is not False
            or summary.get("training_executed") is not False
            or summary.get("official_evaluation_read") is not False
            or len(summary.get("shards", [])) != 2):
        raise ValueError("summary identity, role or completeness failed")
    totals = {key: 0.0 for key in ENERGY_KEYS}
    nodes = 0
    for ordinal, record in enumerate(records):
        item = read(output / f"train_shard_{ordinal}.json")
        if item != summary["shards"][ordinal]:
            raise ValueError("shard output differs from final summary")
        start = int(record["source_idx_min"])
        size = int(record["source_idx_max"]) - start + 1
        indices = [start + index * size // 256 for index in range(256)]
        expected_idx_sha = hashlib.sha256(
            json.dumps(indices, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        if (item.get("shard_ordinal") != ordinal
                or item.get("shard_sha256") != record["sha256"]
                or item.get("sampled_rows") != 256
                or item.get("source_idx_first") != indices[0]
                or item.get("source_idx_last") != indices[-1]
                or item.get("source_indices_sha256") != expected_idx_sha
                or type(item.get("nodes")) is not int or item["nodes"] < 256
                or type(item.get("degree_max")) is not int or not 0 <= item["degree_max"] <= 511):
            raise ValueError("sample, source-row or shard identity failed")
        energy = item.get("energy")
        if (not isinstance(energy, dict) or set(energy) != set(ENERGY_KEYS)
                or any(not isinstance(energy[key], (int, float)) or not math.isfinite(energy[key])
                       for key in ENERGY_KEYS)
                or energy["atom_squared_sum"] <= 0 or energy["degree_squared_sum"] <= 0):
            raise ValueError("invalid component energies")
        for key in totals:
            totals[key] += energy[key]
        nodes += item["nodes"]
    observed = summary.get("input_energy", {})
    expected = {
        "atom_mean_square": totals["atom_squared_sum"] / (nodes * 256),
        "degree_mean_square": totals["degree_squared_sum"] / (nodes * 256),
        "atom_degree_cross_mean": totals["cross_sum"] / (nodes * 256),
        "sampled_nodes": nodes,
    }
    expected["degree_fraction_without_cross"] = (
        expected["degree_mean_square"] /
        (expected["atom_mean_square"] + expected["degree_mean_square"])
    )
    expected["observed_total_mean_square"] = (
        expected["atom_mean_square"] + expected["degree_mean_square"]
        + 2 * expected["atom_degree_cross_mean"]
    )
    for key, value in expected.items():
        actual = observed.get(key)
        if (not isinstance(actual, (int, float)) or not math.isfinite(actual)
                or not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f"aggregate energy mismatch: {key}")
    if not 0 < observed["degree_fraction_without_cross"] < 1:
        raise ValueError("invalid degree fraction")
    cost = read(output / "native_cost.json")
    if (cost.get("format") != "molgap-gptrans-initial-scale-preflight-cost-v1"
            or cost.get("gpu_used") is not False
            or cost.get("sampled_train_rows") != 512
            or cost.get("source_commit") != source_commit
            or cost.get("source_archive_sha256") != source_archive_sha256
            or not isinstance(cost.get("wall_seconds"), (int, float))
            or not 0 < cost["wall_seconds"] < 4 * 3600):
        raise ValueError("source or native-cost receipt failed")
    return {
        "accepted": True,
        "sampled_train_rows": 512,
        "sampled_nodes": nodes,
        "degree_fraction_without_cross": observed["degree_fraction_without_cross"],
        "atom_degree_cross_mean": observed["atom_degree_cross_mean"],
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "output_sha256": {name: digest(output / name) for name in (
            "summary.json", "train_shard_0.json", "train_shard_1.json", "native_cost.json")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.manifest,
                            source_commit=args.source_commit,
                            source_archive_sha256=args.source_archive_sha256),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
