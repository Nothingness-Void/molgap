"""Independent, no-Torch acceptance of Kaggle2 P0 retrieved output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
SOURCE_COMMIT = "8c50a668c4afb9fa3d1e2398287b69e652131cc1"
SOURCE_ARCHIVE_SHA256 = "0f6e0b2f5cda04db0b022432d550a40223b76338026a4564683672fba482056e"
COUNT_KEYS = {
    "connected_nonbond_pairs", "capped_nonbond_pairs", "nontrivial_bond_pairs",
    "same_distance_multiple_signature_pairs", "distances_with_multiple_signatures",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify(output: Path, manifest_path: Path) -> dict:
    if digest(manifest_path) != MANIFEST_SHA256:
        raise ValueError("accepted manifest identity changed")
    manifest = read(manifest_path)
    if manifest.get("identity", {}).get("name") != "ogb-train-100k":
        raise ValueError("wrong fixed dataset")
    records = [row for row in manifest["geometry_shards"] if row["role"] == "train"]
    if len(records) != 2:
        raise ValueError("expected exactly two fixed train shards")
    summary = read(output / "summary.json")
    if (summary.get("format") != "molgap-gptrans-real-path-preflight-v1"
            or summary.get("status") != "COMPLETE"
            or summary.get("manifest_sha256") != MANIFEST_SHA256
            or summary.get("sample_per_shard") != 256
            or summary.get("maximum_distance") != 20
            or summary.get("labels_read") is not False
            or summary.get("checkpoint_loaded") is not False
            or summary.get("official_evaluation_read") is not False
            or len(summary.get("shards", [])) != 2):
        raise ValueError("preflight summary identity/role/completeness failed")
    totals = {name: 0 for name in COUNT_KEYS}
    for ordinal, record in enumerate(records):
        result = read(output / f"train_shard_{ordinal}.json")
        if result != summary["shards"][ordinal]:
            raise ValueError("per-shard output differs from complete summary")
        start = int(record["source_idx_min"])
        size = int(record["source_idx_max"]) - start + 1
        positions = [(index * size) // 256 for index in range(256)]
        indices = [start + position for position in positions]
        index_hash = hashlib.sha256(
            json.dumps(indices, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        if (result.get("shard_ordinal") != ordinal
                or result.get("shard_sha256") != record["sha256"]
                or result.get("sampled_rows") != 256
                or result.get("source_idx_first") != indices[0]
                or result.get("source_idx_last") != indices[-1]
                or result.get("source_indices_sha256") != index_hash):
            raise ValueError("sample/source-row identity failed")
        counts = result.get("counts", {})
        if set(counts) != COUNT_KEYS or any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("missing or invalid count")
        if (counts["capped_nonbond_pairs"] > counts["connected_nonbond_pairs"]
                or counts["nontrivial_bond_pairs"] > counts["capped_nonbond_pairs"]
                or counts["same_distance_multiple_signature_pairs"] > counts["capped_nonbond_pairs"]):
            raise ValueError("inconsistent path counts")
        for key in COUNT_KEYS:
            totals[key] += counts[key]
    cost = read(output / "native_cost.json")
    if (cost.get("format") != "molgap-gptrans-real-path-preflight-cost-v1"
            or cost.get("gpu_used") is not False
            or cost.get("sampled_train_rows") != 512
            or cost.get("source_commit") != SOURCE_COMMIT
            or cost.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
            or not isinstance(cost.get("wall_seconds"), (int, float))
            or not 0 < cost["wall_seconds"] < 4 * 3600):
        raise ValueError("source or native-cost receipt failed")
    return {
        "accepted": True,
        "sampled_train_rows": 512,
        "manifest_sha256": MANIFEST_SHA256,
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "counts": totals,
        "same_distance_multiple_signature_fraction": (
            totals["same_distance_multiple_signature_pairs"] / totals["capped_nonbond_pairs"]
        ),
        "output_sha256": {name: digest(output / name) for name in (
            "summary.json", "train_shard_0.json", "train_shard_1.json", "native_cost.json",
        )},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
