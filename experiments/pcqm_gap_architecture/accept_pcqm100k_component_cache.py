"""No-model acceptance for the Kaggle conjugated-component cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


GEOMETRY_SHA = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path, source_commit: str) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, label: str) -> None:
        if not condition:
            errors.append(label)

    expected = {
        "format": "molgap-pcqm-gap100k-conjugated-component-cache-v1",
        "complete": True,
        "source_commit": source_commit,
        "parent_geometry_cache_aggregate_sha256": GEOMETRY_SHA,
        "component_rule": "connected_components_of_ogb_conjugated_bonds",
        "local_component_order": "ascending_minimum_atom_index",
        "nonmember_component_id": -1,
        "feature_channels": 8,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "gpu_used": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in expected.items():
        require(manifest.get(key) == value, key)
    require(len(manifest.get("shards", [])) == 22, "shard count")
    aggregate = hashlib.sha256()
    role_counts = {"train": 0, "validation": 0}
    for shard in manifest.get("shards", []):
        path = root / shard["file"]
        require(path.is_file(), f"missing {shard['file']}")
        if path.is_file():
            require(sha256_file(path) == shard.get("sha256"), f"hash {shard['file']}")
        role_counts[shard["role"]] += int(shard["graph_count"])
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
    require(role_counts == {"train": 100_000, "validation": 10_000}, "roles")
    require(aggregate.hexdigest() == manifest.get("aggregate_sha256"), "aggregate")
    result = {
        "format": "molgap-pcqm-gap100k-conjugated-cache-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": source_commit,
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "role_counts": role_counts,
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
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root, args.source_commit)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
