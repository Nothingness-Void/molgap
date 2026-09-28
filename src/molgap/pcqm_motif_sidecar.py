"""CPU-only, hash-bound motif partitions over the accepted fixed PCQM cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .pcqm_motif_partition import RULES, derive_motif_partition
from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file


FORMAT = "molgap-pcqm-fixed100k-motif-partition-sidecar-v1"
ACCEPTANCE_FORMAT = "molgap-pcqm-fixed100k-motif-partition-acceptance-v1"


def _rules_sha256() -> str:
    import inspect

    payload = json.dumps({
        "rules": RULES,
        "implementation": inspect.getsource(derive_motif_partition),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _aggregate(shards: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in shards:
        digest.update(
            f"{item['role']}\t{item['parent_file']}\t{item['file']}\t{item['sha256']}\n".encode()
        )
    return digest.hexdigest()


def _fixed_cache():
    from .pcqm_k1_variants_runner import find_fixed_cache

    return find_fixed_cache()


def _dataset(path: Path):
    from .pcqm_k1_variants_runner import _PackedGraphDatasetFactory

    payload = _PackedGraphDatasetFactory.load(path)
    # The packed cache also carries Gap targets. Remove their storage before
    # materializing any graph so this topology-only job cannot read labels.
    if "y" in payload._data:
        del payload._data["y"]
        payload.slices.pop("y", None)
    return payload


def _row_stats(rows: list[dict]) -> dict:
    counts = [int(row["motif_count"]) for row in rows]
    return {
        "rows": len(rows),
        "multi_motif_rows": sum(count > 1 for count in counts),
        "three_plus_motif_rows": sum(count > 2 for count in counts),
        "motif_total": sum(counts),
        "motif_max": max(counts, default=0),
        "inter_motif_undirected_bonds": sum(
            row["motif_edge_index"].shape[1] // 2 for row in rows
        ),
    }


def build_sidecar(output: Path, *, source_commit: str) -> dict:
    """Persist independently retrievable row-aligned shards, never Gap labels."""
    from .pcqm_k1_variants_runner import (
        DEVELOPMENT_ROWS, FIXED_GEOMETRY_SHA256, FIXED_MANIFEST_SHA256,
        TRAIN_ROWS,
    )

    if len(source_commit) != 40:
        raise ValueError("Full source commit required")
    fixed_root, fixed = _fixed_cache()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("A finalized motif sidecar cannot be overwritten")
    shards: list[dict] = []
    role_counts = {"train": 0, "development": 0}
    next_index = {"train": 0, "development": TRAIN_ROWS}
    for parent in fixed["geometry_shards"]:
        role = parent["role"]
        parent_path = fixed_root / parent["file"]
        if role not in role_counts or sha256_file(parent_path) != parent["sha256"]:
            raise RuntimeError("Unaccepted fixed graph shard")
        rows = []
        for graph in _dataset(parent_path):
            source_idx = int(graph.source_idx.reshape(-1)[0])
            if source_idx != next_index[role]:
                raise RuntimeError("Fixed source-index order changed")
            rows.append({
                "source_idx": source_idx,
                "node_count": int(graph.num_nodes),
                **derive_motif_partition(graph),
            })
            next_index[role] += 1
            role_counts[role] += 1
        if len(rows) != parent["rows"]:
            raise RuntimeError("Fixed graph shard row count changed")
        name = Path(parent["file"]).stem + "-motif-partition.pt"
        path = output / name
        atomic_torch_save(path, rows)
        shards.append({
            "role": role, "parent_file": parent["file"],
            "parent_sha256": parent["sha256"], "file": name,
            "sha256": sha256_file(path), "rows": len(rows),
            "source_idx_min": rows[0]["source_idx"],
            "source_idx_max": rows[-1]["source_idx"],
            "statistics": _row_stats(rows),
        })
        atomic_json(output / "progress.json", {
            "format": FORMAT, "complete": False, "source_commit": source_commit,
            "completed_shards": shards, "role_counts": role_counts,
            "gap_labels_read": False, "model_inference_executed": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False, "test_challenge_role_read": False,
        })
        print(f"motif sidecar {role} {role_counts[role]}", flush=True)
    if role_counts != {"train": TRAIN_ROWS, "development": DEVELOPMENT_ROWS}:
        raise RuntimeError("Motif sidecar is incomplete")
    manifest = {
        "format": FORMAT, "complete": True, "source_commit": source_commit,
        "rules_sha256": _rules_sha256(),
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "fixed_geometry_sha256": FIXED_GEOMETRY_SHA256,
        "role_counts": role_counts, "shards": shards,
        "aggregate_sha256": _aggregate(shards),
        "derived_from_ogb_2d_only": True,
        "gap_labels_read": False, "model_inference_executed": False,
        "gpu_used": False, "official_validation_role_read": False,
        "test_dev_role_read": False, "test_challenge_role_read": False,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def accept_sidecar(root: Path, *, expected_source_commit: str) -> dict:
    """Independently recompute every row against its accepted parent graph."""
    import torch

    from .pcqm_k1_variants_runner import (
        DEVELOPMENT_ROWS, FIXED_GEOMETRY_SHA256, FIXED_MANIFEST_SHA256,
        TRAIN_ROWS,
    )

    fixed_root, fixed = _fixed_cache()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("format") != FORMAT or manifest.get("complete") is not True
        or manifest.get("source_commit") != expected_source_commit
        or manifest.get("rules_sha256") != _rules_sha256()
        or manifest.get("fixed_manifest_sha256") != FIXED_MANIFEST_SHA256
        or manifest.get("fixed_geometry_sha256") != FIXED_GEOMETRY_SHA256
        or manifest.get("role_counts") != {"train": TRAIN_ROWS, "development": DEVELOPMENT_ROWS}
        or manifest.get("derived_from_ogb_2d_only") is not True
        or any(manifest.get(flag) is not False for flag in (
            "gap_labels_read", "model_inference_executed", "gpu_used",
            "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read",
        ))
        or len(manifest.get("shards", [])) != len(fixed["geometry_shards"])
    ):
        raise RuntimeError("Motif sidecar identity/role seal invalid")
    next_index = {"train": 0, "development": TRAIN_ROWS}
    for parent, item in zip(fixed["geometry_shards"], manifest["shards"], strict=True):
        path = (root / item["file"]).resolve()
        parent_path = fixed_root / parent["file"]
        if (
            not path.is_relative_to(root.resolve())
            or item["role"] != parent["role"]
            or item["parent_file"] != parent["file"]
            or item["parent_sha256"] != parent["sha256"]
            or item["rows"] != parent["rows"]
            or item["source_idx_min"] != next_index[parent["role"]]
            or sha256_file(parent_path) != parent["sha256"]
            or sha256_file(path) != item["sha256"]
        ):
            raise RuntimeError("Motif sidecar shard hash/parent identity invalid")
        rows = torch.load(path, map_location="cpu", weights_only=False)
        graphs = _dataset(parent_path)
        if len(rows) != len(graphs) or len(rows) != item["rows"]:
            raise RuntimeError("Motif sidecar row count invalid")
        for row, graph in zip(rows, graphs, strict=True):
            source_idx = next_index[parent["role"]]
            expected = derive_motif_partition(graph)
            if (
                row["source_idx"] != source_idx
                or int(graph.source_idx.reshape(-1)[0]) != source_idx
                or row["node_count"] != int(graph.num_nodes)
                or row.keys() != {"source_idx", "node_count", *expected.keys()}
                or any(
                    not torch.equal(row[key], value) if isinstance(value, torch.Tensor)
                    else row[key] != value
                    for key, value in expected.items()
                )
            ):
                raise RuntimeError(f"Motif derivation changed at {source_idx}")
            next_index[parent["role"]] += 1
        if (
            item["source_idx_max"] != next_index[parent["role"]] - 1
            or item["statistics"] != _row_stats(rows)
        ):
            raise RuntimeError("Motif shard statistics/endpoint invalid")
    if (
        next_index != {"train": TRAIN_ROWS, "development": TRAIN_ROWS + DEVELOPMENT_ROWS}
        or manifest.get("aggregate_sha256") != _aggregate(manifest["shards"])
    ):
        raise RuntimeError("Motif sidecar coverage/aggregate invalid")
    acceptance = {
        "format": ACCEPTANCE_FORMAT, "accepted": True,
        "source_commit": expected_source_commit,
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "aggregate_sha256": manifest["aggregate_sha256"],
        "rows_recomputed": TRAIN_ROWS + DEVELOPMENT_ROWS,
        "gap_labels_read": False, "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False, "test_challenge_role_read": False,
    }
    atomic_json(root / "acceptance.json", acceptance)
    return acceptance
