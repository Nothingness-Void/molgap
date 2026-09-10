"""Immutable pure-2D cache for the frozen K1 500K scale bridge."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .pcqm_gap_data import (
    RWSE_DIM,
    _atomic_torch_save,
    _make_graph,
    atomic_json,
    fixed_screen_split,
    read_official_train_prefix,
    sha256_file,
)
from .pcqm_k1_scale import SCALE_TRAIN_ROWS, VALIDATION_ROWS, build_scale_split
from .pcqm_shadow import frozen_shadow_split


SHARD_SIZE = 5_000


def _frozen_scale_split() -> dict:
    base = fixed_screen_split()
    used = set(base["train"].tolist()) | set(base["validation"].tolist())
    shadow = frozen_shadow_split(used)
    return build_scale_split(
        {key: value.tolist() if hasattr(value, "tolist") else value for key, value in base.items()},
        {
            "effective_shadow": shadow["shadow"].tolist(),
            "reserve": shadow["reserve"].tolist(),
        },
    )


def build_scale_cache(source_csv: Path, output: Path, *, source_commit: str) -> dict:
    """Build resumable role shards without touching official validation/test roles."""
    import torch

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is True:
            return manifest
        raise RuntimeError("Incomplete final scale-cache manifest exists")

    frame = read_official_train_prefix(source_csv)
    split = _frozen_scale_split()
    atomic_json(output / "split.json", split)
    progress_path = output / "progress.json"
    progress = {"next": {"train": 0, "validation": 0}, "shards": []}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("source_commit") != source_commit:
            raise RuntimeError("Scale-cache resume source changed")

    shards = list(progress.get("shards", []))
    ranges = progress.get(
        "feature_ranges",
        {
            "atom_feature_min": [None] * 9,
            "atom_feature_max": [None] * 9,
            "bond_feature_min": [None] * 3,
            "bond_feature_max": [None] * 3,
        },
    )
    bondless = int(progress.get("bondless_graphs", 0))
    for role in ("train", "validation"):
        positions = split[role]
        offset = int(progress.get("next", {}).get(role, 0))
        while offset < len(positions):
            stop = min(offset + SHARD_SIZE, len(positions))
            graphs = []
            for position in positions[offset:stop]:
                graph = _make_graph(frame.iloc[int(position)])
                graph.row_id = graph.row_index.clone()
                if graph.edge_attr.shape[0] == 0:
                    bondless += 1
                for stem, values in (("atom", graph.x), ("bond", graph.edge_attr)):
                    if values.shape[0] == 0:
                        continue
                    low = values.min(dim=0).values.tolist()
                    high = values.max(dim=0).values.tolist()
                    for suffix, observed, reducer in (
                        ("min", low, min), ("max", high, max)
                    ):
                        key = f"{stem}_feature_{suffix}"
                        ranges[key] = [
                            int(value) if old is None else reducer(old, int(value))
                            for old, value in zip(ranges[key], observed)
                        ]
                graphs.append(graph)
            path = output / f"{role}-{offset // SHARD_SIZE:04d}.pt"
            _atomic_torch_save(path, graphs)
            record = {
                "role": role,
                "file": path.name,
                "source_start": offset,
                "source_stop": stop,
                "graph_count": len(graphs),
                "sha256": sha256_file(path),
            }
            shards = [item for item in shards if item["file"] != path.name] + [record]
            offset = stop
            progress.setdefault("next", {})[role] = offset
            progress.update(
                {
                    "format": "molgap-pcqm-k1-scale500k-progress-v1",
                    "source_commit": source_commit,
                    "split_train_sha256": split["train_sha256"],
                    "split_validation_sha256": split["validation_sha256"],
                    "shards": shards,
                    "feature_ranges": ranges,
                    "bondless_graphs": bondless,
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                    "shadow_labels_read": False,
                }
            )
            atomic_json(progress_path, progress)
            del graphs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    shards.sort(key=lambda item: (item["role"], item["file"]))
    aggregate = hashlib.sha256()
    for item in shards:
        aggregate.update(
            f"{item['role']}\t{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    counts = {
        role: sum(item["graph_count"] for item in shards if item["role"] == role)
        for role in ("train", "validation")
    }
    if counts != {"train": SCALE_TRAIN_ROWS, "validation": VALIDATION_ROWS}:
        raise RuntimeError(f"Incomplete scale cache: {counts}")
    manifest = {
        "format": "molgap-pcqm-k1-scale500k-cache-v1",
        "complete": True,
        "source_dataset": "piero0/pcqm4mv2",
        "source_commit": source_commit,
        "official_train_rows_read": len(frame),
        "train_graphs": counts["train"],
        "validation_graphs": counts["validation"],
        "train_index_sha256": split["train_sha256"],
        "validation_index_sha256": split["validation_sha256"],
        "added_train_sha256": split["added_train_sha256"],
        "split_file": "split.json",
        "split_file_sha256": sha256_file(output / "split.json"),
        "atom_feature_dim": 9,
        "bond_feature_dim": 3,
        "rwse_dim": RWSE_DIM,
        "feature_ranges": ranges,
        "bondless_graphs": bondless,
        "shards": shards,
        "aggregate_sha256": aggregate.hexdigest(),
        "gpu_used": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "shadow_labels_read": False,
    }
    atomic_json(manifest_path, manifest)
    return manifest
