"""No-inference, no-label acceptance for the K1 shadow graph cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from molgap.pcqm_gap_data import sha256_file
from molgap.pcqm_shadow import (
    FROZEN_CANDIDATE,
    GEOMETRY_CACHE_SHA256,
    PARENT_GRAPH_CACHE_SHA256,
    RWSE_DIM,
    SHADOW_ROWS,
    SHADOW_SPLIT_SEED,
    index_sha256,
)


def accept(root: Path, *, source_commit: str) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "format": "molgap-pcqm-k1-shadow-cache-v1",
        "complete": True,
        "source_commit": source_commit,
        "frozen_candidate": FROZEN_CANDIDATE,
        "candidate_selected_before_shadow_labels": True,
        "csv_columns_read": ["idx", "smiles"],
        "parent_graph_cache_aggregate_sha256": PARENT_GRAPH_CACHE_SHA256,
        "parent_geometry_cache_aggregate_sha256": GEOMETRY_CACHE_SHA256,
        "shadow_graphs": SHADOW_ROWS,
        "atom_feature_dim": 9,
        "bond_feature_dim": 3,
        "rwse_dim": RWSE_DIM,
        "shadow_labels_read": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "model_inference_executed": False,
        "gpu_used": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Shadow cache contract changed for {key}")
    split_path = root / manifest["split_file"]
    failures_path = root / manifest["failures_file"]
    if sha256_file(split_path) != manifest["split_file_sha256"]:
        raise RuntimeError("Shadow split hash changed")
    if sha256_file(failures_path) != manifest["failures_file_sha256"]:
        raise RuntimeError("Shadow failure-ledger hash changed")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("split_seed") != SHADOW_SPLIT_SEED:
        raise RuntimeError("Shadow split seed changed")
    effective = [int(value) for value in split["effective_shadow"]]
    if len(effective) != SHADOW_ROWS or len(set(effective)) != SHADOW_ROWS:
        raise RuntimeError("Shadow role count/uniqueness changed")
    if index_sha256(effective) != manifest["effective_shadow_index_sha256"]:
        raise RuntimeError("Shadow effective index identity changed")
    aggregate = hashlib.sha256()
    observed = []
    for shard in manifest["shards"]:
        path = root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"Shadow shard hash changed: {path.name}")
        aggregate.update(
            f"shadow\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
        graphs = torch.load(path, map_location="cpu", weights_only=False)
        if len(graphs) != int(shard["graph_count"]):
            raise RuntimeError(f"Shadow shard count changed: {path.name}")
        for graph in graphs:
            if "y" in graph:
                raise RuntimeError("Shadow graph contains a target")
            if graph.x.ndim != 2 or graph.x.shape[1] != 9:
                raise RuntimeError("Shadow atom shape changed")
            if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != 3:
                raise RuntimeError("Shadow bond shape changed")
            if tuple(graph.random_walk_pe.shape) != (graph.num_nodes, RWSE_DIM):
                raise RuntimeError("Shadow RWSE shape changed")
            if not torch.isfinite(graph.random_walk_pe).all():
                raise RuntimeError("Shadow RWSE became non-finite")
            observed.append(int(graph.row_index.view(-1)[0]))
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("Shadow aggregate hash changed")
    if observed != effective:
        raise RuntimeError("Shadow graph row order changed")
    return {
        "format": "molgap-pcqm-k1-shadow-cache-acceptance-v1",
        "accepted": True,
        "source_commit": source_commit,
        "aggregate_sha256": manifest["aggregate_sha256"],
        "effective_shadow_index_sha256": manifest["effective_shadow_index_sha256"],
        "shadow_graphs": len(observed),
        "failure_count": manifest["failure_count"],
        "reserve_rows_consumed": manifest["reserve_rows_consumed"],
        "shadow_labels_read": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = accept(args.root, source_commit=args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
