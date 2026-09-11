"""No-model acceptance for the K1 500K pure-2D graph cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from molgap.pcqm_gap_data import sha256_file
from molgap.pcqm_k1_scale import (
    ROLE_ROWS_READ,
    SCALE_TRAIN_ROWS,
    SCNET_REFERENCE_CACHE_SHA256,
    TRAIN_SHA256,
    UNSANITIZED_OGB_INDEX_SHA256,
    UNSANITIZED_OGB_SOURCE_INDICES,
    VALIDATION_ROWS,
    VALIDATION_SHA256,
    index_sha256,
)


def accept(root: Path, *, source_commit: str | None = None) -> dict:
    import torch

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    split_path = root / manifest["split_file"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    errors = []

    def require(value, name):
        if not value:
            errors.append(name)

    require(manifest.get("format") == "molgap-pcqm-k1-scale500k-cache-v4", "format")
    require(manifest.get("complete") is True, "complete")
    if source_commit:
        require(manifest.get("source_commit") == source_commit, "source_commit")
    require(manifest.get("train_graphs") == SCALE_TRAIN_ROWS, "train_rows")
    require(manifest.get("validation_graphs") == VALIDATION_ROWS, "validation_rows")
    require(manifest.get("official_train_rows_read") == ROLE_ROWS_READ, "rows_read")
    require(len(split.get("train", [])) == SCALE_TRAIN_ROWS, "split_train_rows")
    require(len(split.get("validation", [])) == VALIDATION_ROWS, "split_validation_rows")
    require(index_sha256(split.get("train", [])) == manifest.get("train_index_sha256"), "train_sha")
    require(index_sha256(split.get("validation", [])) == manifest.get("validation_index_sha256"), "validation_sha")
    require(manifest.get("train_index_sha256") == TRAIN_SHA256, "scnet_train_sha")
    require(manifest.get("validation_index_sha256") == VALIDATION_SHA256, "scnet_validation_sha")
    require(split.get("train") == list(range(SCALE_TRAIN_ROWS)), "scnet_train_rows")
    require(
        split.get("validation") == list(range(SCALE_TRAIN_ROWS, ROLE_ROWS_READ)),
        "scnet_validation_rows",
    )
    require(
        manifest.get("scnet_reference_cache_aggregate_sha256")
        == SCNET_REFERENCE_CACHE_SHA256,
        "scnet_reference_cache",
    )
    require(set(split.get("train", [])).isdisjoint(split.get("validation", [])), "role_overlap")
    require(sha256_file(split_path) == manifest.get("split_file_sha256"), "split_file_sha")
    for key in ("official_validation_role_read", "test_dev_role_read", "shadow_labels_read"):
        require(manifest.get(key) is False, key)
    require(manifest.get("gpu_used") is False, "gpu_used")
    require(manifest.get("atom_feature_dim") == 9, "atom_dim")
    require(manifest.get("bond_feature_dim") == 3, "bond_dim")
    require(manifest.get("rwse_dim") == 16, "rwse_dim")
    require(manifest.get("failed_graph_attempts") == 0, "failed_graph_attempts")
    require(manifest.get("unresolved_graphs") == 0, "unresolved_graphs")
    failures_path = root / manifest.get("failures_file", "")
    require(failures_path.is_file(), "failures_file")
    if failures_path.is_file():
        require(sha256_file(failures_path) == manifest.get("failures_file_sha256"), "failures_sha")
        failures = json.loads(failures_path.read_text(encoding="utf-8"))
        require(failures.get("attempts") == [], "failures_empty")
    fallback_path = root / manifest.get("parser_fallback_file", "")
    require(fallback_path.is_file(), "parser_fallback_file")
    if fallback_path.is_file():
        require(
            sha256_file(fallback_path) == manifest.get("parser_fallback_file_sha256"),
            "parser_fallback_sha",
        )
        fallback = json.loads(fallback_path.read_text(encoding="utf-8"))
        fallback_indices = [int(item["row_index"]) for item in fallback.get("entries", [])]
        require(tuple(fallback_indices) == UNSANITIZED_OGB_SOURCE_INDICES, "parser_fallback_rows")
        require(index_sha256(fallback_indices) == UNSANITIZED_OGB_INDEX_SHA256, "parser_fallback_index_sha")
        require(manifest.get("parser_fallback_count") == len(fallback_indices), "parser_fallback_count")
        require(
            manifest.get("parser_fallback_index_sha256") == UNSANITIZED_OGB_INDEX_SHA256,
            "manifest_parser_fallback_sha",
        )

    aggregate = hashlib.sha256()
    counts = {"train": 0, "validation": 0}
    seen_rows = {"train": [], "validation": []}
    for shard in manifest.get("shards", []):
        path = root / shard["file"]
        require(path.is_file(), f"missing:{path.name}")
        if not path.is_file():
            continue
        require(sha256_file(path) == shard.get("sha256"), f"sha:{path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(len(payload) == shard.get("graph_count"), f"count:{path.name}")
        role = shard.get("role")
        require(role in counts, f"role:{path.name}")
        if role not in counts:
            continue
        counts[role] += len(payload)
        for graph in payload:
            require(graph.x.ndim == 2 and graph.x.shape[1] == 9, "graph_atom_shape")
            require(graph.edge_attr.ndim == 2 and graph.edge_attr.shape[1] == 3, "graph_bond_shape")
            require(tuple(graph.random_walk_pe.shape) == (graph.num_nodes, 16), "graph_rwse_shape")
            require(bool(torch.isfinite(graph.random_walk_pe).all()), "graph_rwse_finite")
            require(bool(torch.isfinite(graph.y).all()), "graph_target_finite")
            seen_rows[role].append(int(graph.row_id.view(-1)[0]))
        aggregate.update(f"{role}\t{shard['file']}\t{shard['sha256']}\n".encode("ascii"))
    require(counts == {"train": SCALE_TRAIN_ROWS, "validation": VALIDATION_ROWS}, "graph_counts")
    require(seen_rows["train"] == split.get("train"), "train_row_alignment")
    require(seen_rows["validation"] == split.get("validation"), "validation_row_alignment")
    require(aggregate.hexdigest() == manifest.get("aggregate_sha256"), "aggregate_sha")
    result = {
        "format": "molgap-pcqm-k1-scale500k-cache-acceptance-v3",
        "accepted": not errors,
        "errors": sorted(set(errors)),
        "source_commit": manifest.get("source_commit"),
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "train_index_sha256": manifest.get("train_index_sha256"),
        "validation_index_sha256": manifest.get("validation_index_sha256"),
        "train_graphs": counts["train"],
        "validation_graphs": counts["validation"],
        "parser_fallback_count": manifest.get("parser_fallback_count"),
        "parser_fallback_index_sha256": manifest.get("parser_fallback_index_sha256"),
        "scnet_reference_cache_aggregate_sha256": manifest.get(
            "scnet_reference_cache_aggregate_sha256"
        ),
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "shadow_labels_read": False,
    }
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root, source_commit=args.source_commit)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
