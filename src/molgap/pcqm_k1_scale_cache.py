"""Immutable pure-2D cache for the SCNet-matched K1 500K benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .pcqm_gap_data import (
    RWSE_DIM,
    _atomic_torch_save,
    _make_graph,
    _open_csv,
    atomic_json,
    sha256_file,
)
from .pcqm_k1_scale import (
    ROLE_ROWS_READ,
    SCALE_TRAIN_ROWS,
    VALIDATION_ROWS,
    build_scale_split,
)


SHARD_SIZE = 5_000


def _read_frozen_roles(source_csv: Path):
    import numpy as np
    import pandas as pd

    with _open_csv(source_csv) as handle:
        frame = pd.read_csv(
            handle,
            nrows=ROLE_ROWS_READ,
            usecols=["idx", "smiles", "homolumogap"],
        )
    if len(frame) != ROLE_ROWS_READ:
        raise RuntimeError(f"Expected {ROLE_ROWS_READ} PCQM role rows, found {len(frame)}")
    expected = np.arange(ROLE_ROWS_READ, dtype=np.int64)
    observed = frame["idx"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(observed, expected):
        raise RuntimeError("SCNet-matched PCQM source indices changed")
    targets = pd.to_numeric(frame["homolumogap"], errors="coerce").to_numpy()
    if not np.isfinite(targets).all() or frame["smiles"].isna().any():
        raise RuntimeError("SCNet-matched PCQM role rows are incomplete")
    return frame


def build_scale_cache(source_csv: Path, output: Path, *, source_commit: str) -> dict:
    """Build resumable role shards without reading official validation/test roles."""
    import torch

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is True:
            return manifest
        raise RuntimeError("Incomplete final scale-cache manifest exists")

    frame = _read_frozen_roles(source_csv)
    split = build_scale_split()
    atomic_json(output / "split.json", split)
    failures_path = output / "failures.json"
    progress_path = output / "progress.json"
    progress = {
        "format": "molgap-pcqm-k1-scale500k-progress-v3",
        "source_commit": source_commit,
        "next": {"train": 0, "validation": 0},
        "shards": [],
        "feature_ranges": {
            "atom_feature_min": [None] * 9,
            "atom_feature_max": [None] * 9,
            "bond_feature_min": [None] * 3,
            "bond_feature_max": [None] * 3,
        },
        "bondless_graphs": 0,
        "failures": [],
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("format") != "molgap-pcqm-k1-scale500k-progress-v3"
            or progress.get("source_commit") != source_commit
        ):
            raise RuntimeError("Scale-cache resume source changed")

    shards = list(progress.get("shards", []))
    ranges = progress["feature_ranges"]
    bondless = int(progress.get("bondless_graphs", 0))
    failures = list(progress.get("failures", []))
    for role in ("train", "validation"):
        positions = split[role]
        offset = int(progress["next"].get(role, 0))
        while offset < len(positions):
            stop = min(offset + SHARD_SIZE, len(positions))
            graphs = []
            for slot, position in enumerate(positions[offset:stop], start=offset):
                row = frame.iloc[int(position)]
                try:
                    graph = _make_graph(row)
                except Exception as error:
                    failures.append(
                        {
                            "role": role,
                            "slot": slot,
                            "row_index": int(row.idx),
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    atomic_json(
                        failures_path,
                        {
                            "format": "molgap-pcqm-k1-scale500k-failures-v2",
                            "attempts": failures,
                        },
                    )
                    raise RuntimeError(
                        "SCNet-matched 500K/50K role contains an unparseable graph"
                    ) from error
                graph.row_id = graph.row_index.clone()
                if graph.edge_attr.shape[0] == 0:
                    bondless += 1
                for stem, values in (("atom", graph.x), ("bond", graph.edge_attr)):
                    if values.shape[0] == 0:
                        continue
                    low = values.min(dim=0).values.tolist()
                    high = values.max(dim=0).values.tolist()
                    for suffix, observed, reducer in (("min", low, min), ("max", high, max)):
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
            progress.update(
                {
                    "next": {**progress["next"], role: offset},
                    "shards": shards,
                    "feature_ranges": ranges,
                    "bondless_graphs": bondless,
                    "failures": failures,
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                }
            )
            atomic_json(progress_path, progress)
            atomic_json(
                failures_path,
                {
                    "format": "molgap-pcqm-k1-scale500k-failures-v2",
                    "attempts": failures,
                },
            )
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
    if failures:
        raise RuntimeError("SCNet-matched scale cache has unresolved graph failures")
    manifest = {
        "format": "molgap-pcqm-k1-scale500k-cache-v3",
        "complete": True,
        "source_dataset": "piero0/pcqm4mv2",
        "source_commit": source_commit,
        "official_train_rows_read": ROLE_ROWS_READ,
        "train_graphs": counts["train"],
        "validation_graphs": counts["validation"],
        "train_index_sha256": split["train_sha256"],
        "validation_index_sha256": split["validation_sha256"],
        "split_file": "split.json",
        "split_file_sha256": sha256_file(output / "split.json"),
        "failures_file": failures_path.name,
        "failures_file_sha256": sha256_file(failures_path),
        "failed_graph_attempts": 0,
        "unresolved_graphs": 0,
        "atom_feature_dim": 9,
        "bond_feature_dim": 3,
        "rwse_dim": RWSE_DIM,
        "feature_ranges": ranges,
        "bondless_graphs": bondless,
        "shards": shards,
        "aggregate_sha256": aggregate.hexdigest(),
        "scnet_reference_cache_aggregate_sha256": split[
            "scnet_reference_cache_aggregate_sha256"
        ],
        "gpu_used": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "shadow_labels_read": False,
    }
    atomic_json(manifest_path, manifest)
    return manifest
