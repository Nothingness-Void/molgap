"""No-model acceptance for the deterministic conjugated-component cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PARENT_SHA = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path, source_commit: str) -> dict:
    import torch

    from molgap.pcqm_conjugated_cache import FEATURE_NAMES, with_conjugated_components

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "format": "molgap-pcqm-gap100k-conjugated-component-cache-v1",
        "complete": True,
        "source_commit": source_commit,
        "parent_geometry_cache_aggregate_sha256": PARENT_SHA,
        "component_rule": "connected_components_of_ogb_conjugated_bonds",
        "local_component_order": "ascending_minimum_atom_index",
        "nonmember_component_id": -1,
        "feature_names": list(FEATURE_NAMES),
        "feature_channels": 8,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "gpu_used": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"conjugated cache contract changed: {key}")
    if len(manifest.get("shards", [])) != 22:
        raise RuntimeError("expected 22 cache shards")
    aggregate = hashlib.sha256()
    role_counts = {"train": 0, "validation": 0}
    checked_graphs = 0
    for shard in manifest["shards"]:
        path = root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"conjugated shard hash changed: {path.name}")
        aggregate.update(
            f"{shard['role']}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
        graphs = torch.load(path, map_location="cpu", weights_only=False)
        if len(graphs) != int(shard["graph_count"]):
            raise RuntimeError(f"conjugated graph count changed: {path.name}")
        role_counts[shard["role"]] += len(graphs)
        for graph in graphs:
            expected_id = graph.conjugated_component_id.clone()
            expected_count = graph.conjugated_component_count.clone()
            expected_features = graph.conjugated_features.clone()
            with_conjugated_components(graph)
            if not torch.equal(graph.conjugated_component_id, expected_id):
                raise RuntimeError("component id recomputation mismatch")
            if not torch.equal(graph.conjugated_component_count, expected_count):
                raise RuntimeError("component count recomputation mismatch")
            if not torch.equal(graph.conjugated_features, expected_features):
                raise RuntimeError("component feature recomputation mismatch")
            if not torch.isfinite(expected_features).all():
                raise RuntimeError("nonfinite conjugated descriptor")
            checked_graphs += 1
    if role_counts != {"train": 100_000, "validation": 10_000}:
        raise RuntimeError("conjugated role counts changed")
    if aggregate.hexdigest() != manifest.get("aggregate_sha256"):
        raise RuntimeError("conjugated aggregate hash changed")
    if manifest["totals"]["train"]["components"] <= 0:
        raise RuntimeError("training cache contains no conjugated components")
    return {
        "format": "molgap-pcqm-gap100k-conjugated-cache-acceptance-v1",
        "accepted": True,
        "source_commit": source_commit,
        "parent_geometry_cache_aggregate_sha256": PARENT_SHA,
        "aggregate_sha256": manifest["aggregate_sha256"],
        "checked_graphs": checked_graphs,
        "role_counts": role_counts,
        "totals": manifest["totals"],
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = accept(args.root, args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
