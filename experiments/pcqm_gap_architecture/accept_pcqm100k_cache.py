"""No-runtime acceptance for the downloaded official-PCQM 100K cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_sha256(indices) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def accept(root: Path, expected_source_commit: str | None = None) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_path = root / manifest["split_file"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    errors = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(manifest.get("format") == "molgap-pcqm-gap100k-cache-v1", "format")
    require(manifest.get("complete") is True, "complete")
    require(manifest.get("source_dataset") == "piero0/pcqm4mv2", "source dataset")
    if expected_source_commit is not None:
        require(manifest.get("source_commit") == expected_source_commit, "source commit")
    require(manifest.get("official_train_rows_read") == 3_378_606, "train rows")
    require(manifest.get("official_validation_role_read") is False, "official validation read")
    require(manifest.get("test_dev_role_read") is False, "test-dev read")
    require(manifest.get("gpu_used") is False, "GPU used")
    require(manifest.get("failed_graphs") == 0, "graph failures")
    require(manifest.get("train_graphs") == 100_000, "train graph count")
    require(manifest.get("validation_graphs") == 10_000, "validation graph count")
    require(manifest.get("atom_feature_dim") == 9, "atom feature dim")
    require(manifest.get("bond_feature_dim") == 3, "bond feature dim")
    require(manifest.get("rwse_dim") == 16, "RWSE dim")
    require(sha256_file(split_path) == manifest.get("split_file_sha256"), "split SHA")
    train = split.get("train", [])
    validation = split.get("validation", [])
    require(len(train) == 100_000, "train split count")
    require(len(validation) == 10_000, "validation split count")
    require(set(train).isdisjoint(validation), "split overlap")
    require(min(train + validation, default=-1) >= 0, "negative index")
    require(max(train + validation, default=3_378_606) < 3_378_606, "non-train index")
    require(index_sha256(train) == manifest.get("train_index_sha256"), "train index SHA")
    require(
        index_sha256(validation) == manifest.get("validation_index_sha256"),
        "validation index SHA",
    )
    aggregate = hashlib.sha256()
    shard_graphs = {"train": 0, "validation": 0}
    for shard in manifest.get("shards", []):
        path = root / shard["file"]
        require(path.is_file(), f"missing shard {path.name}")
        if path.is_file():
            require(sha256_file(path) == shard.get("sha256"), f"shard SHA {path.name}")
        role = shard.get("role")
        require(role in shard_graphs, f"shard role {role}")
        if role in shard_graphs:
            shard_graphs[role] += int(shard.get("graph_count", 0))
        aggregate.update(
            f"{role}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
    require(shard_graphs == {"train": 100_000, "validation": 10_000}, "shard graph totals")
    require(aggregate.hexdigest() == manifest.get("aggregate_sha256"), "aggregate SHA")
    result = {
        "format": "molgap-pcqm-gap100k-cache-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": manifest.get("source_commit"),
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "train_index_sha256": manifest.get("train_index_sha256"),
        "validation_index_sha256": manifest.get("validation_index_sha256"),
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root, args.expected_source_commit)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

