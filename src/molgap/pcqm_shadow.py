"""Label-sealed official-train-derived PCQM shadow graph cache."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .pcqm_gap_data import OFFICIAL_TRAIN_ROWS, RWSE_DIM, _open_csv, sha256_file


SHADOW_ROWS = 10_000
SHADOW_RESERVE_ROWS = 256
SHADOW_SPLIT_SEED = 2_026_091_042
PARENT_GRAPH_CACHE_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
)
GEOMETRY_CACHE_SHA256 = (
    "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
)
FROZEN_CANDIDATE = "neural_atom_k1"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def index_sha256(indices) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _used_selection_indices() -> set[int]:
    import torch

    from .pcqm_local_global_runner import find_geometry_cache

    root, manifest = find_geometry_cache()
    if manifest.get("aggregate_sha256") != GEOMETRY_CACHE_SHA256:
        raise RuntimeError("Geometry cache identity changed")
    used = set()
    for shard in manifest["shards"]:
        graphs = torch.load(root / shard["file"], map_location="cpu", weights_only=False)
        for graph in graphs:
            values = graph.row_index.view(-1).tolist()
            if len(values) != 1:
                raise RuntimeError("Parent row identity changed")
            used.add(int(values[0]))
    if len(used) != 110_000:
        raise RuntimeError(f"Expected 110000 distinct parent rows, found {len(used)}")
    return used


def frozen_shadow_split(used: set[int]) -> dict:
    import numpy as np

    if any(index < 0 or index >= OFFICIAL_TRAIN_ROWS for index in used):
        raise RuntimeError("Parent role contains an index outside official train")
    available = np.setdiff1d(
        np.arange(OFFICIAL_TRAIN_ROWS, dtype=np.int64),
        np.fromiter(sorted(used), dtype=np.int64),
        assume_unique=True,
    )
    generator = np.random.default_rng(SHADOW_SPLIT_SEED)
    selected = generator.choice(
        available,
        size=SHADOW_ROWS + SHADOW_RESERVE_ROWS,
        replace=False,
    )
    shadow = np.sort(selected[:SHADOW_ROWS]).astype(np.int64)
    reserve = selected[SHADOW_ROWS:].astype(np.int64)
    if np.intersect1d(shadow, reserve).size or any(int(x) in used for x in selected):
        raise RuntimeError("Shadow split overlaps a sealed role")
    return {
        "shadow": shadow,
        "reserve": reserve,
        "shadow_sha256": index_sha256(shadow),
        "reserve_sha256": index_sha256(reserve),
    }


def read_official_train_smiles(path: Path):
    import numpy as np
    import pandas as pd

    with _open_csv(path) as handle:
        frame = pd.read_csv(
            handle,
            nrows=OFFICIAL_TRAIN_ROWS,
            usecols=["idx", "smiles"],
        )
    if list(frame.columns) != ["idx", "smiles"]:
        raise RuntimeError(f"Unexpected label-sealed columns: {list(frame.columns)}")
    if len(frame) != OFFICIAL_TRAIN_ROWS:
        raise RuntimeError("Official train prefix row count changed")
    indices = frame["idx"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(indices, np.arange(OFFICIAL_TRAIN_ROWS, dtype=np.int64)):
        raise RuntimeError("Official train index prefix changed")
    if frame["smiles"].isna().any():
        raise RuntimeError("Official train contains missing SMILES")
    return frame


def _make_unlabeled_graph(row):
    import torch
    from ogb.utils.mol import smiles2graph
    from torch_geometric.data import Data
    from torch_geometric.transforms import AddRandomWalkPE

    payload = smiles2graph(str(row.smiles))
    graph = Data(
        x=torch.as_tensor(payload["node_feat"], dtype=torch.long),
        edge_index=torch.as_tensor(payload["edge_index"], dtype=torch.long),
        edge_attr=torch.as_tensor(payload["edge_feat"], dtype=torch.long),
        row_index=torch.tensor([int(row.idx)], dtype=torch.long),
    )
    graph = AddRandomWalkPE(walk_length=RWSE_DIM, attr_name="random_walk_pe")(graph)
    if "y" in graph or tuple(graph.x.shape[1:]) != (9,):
        raise RuntimeError("Unlabeled atom contract changed")
    if tuple(graph.edge_attr.shape[1:]) != (3,):
        raise RuntimeError("Unlabeled bond contract changed")
    if tuple(graph.random_walk_pe.shape) != (graph.num_nodes, RWSE_DIM):
        raise RuntimeError("Shadow RWSE shape changed")
    if not torch.isfinite(graph.random_walk_pe).all():
        raise RuntimeError("Shadow RWSE contains non-finite values")
    return graph


def build_shadow_cache(
    source_csv: Path,
    output_dir: Path,
    *,
    source_dataset: str,
    source_commit: str,
    shard_size: int = 1_000,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is True:
            return manifest
        raise RuntimeError("Incomplete shadow manifest exists")

    used = _used_selection_indices()
    split = frozen_shadow_split(used)
    frame = read_official_train_smiles(source_csv)
    progress_path = output_dir / "progress.json"
    failures_path = output_dir / "failures.json"
    split_path = output_dir / "split.json"
    progress = {
        "next_offset": 0,
        "reserve_cursor": 0,
        "effective_indices": [],
        "failures": [],
        "shards": [],
    }
    if progress_path.exists():
        progress.update(json.loads(progress_path.read_text(encoding="utf-8")))
        if progress.get("source_commit") != source_commit:
            raise RuntimeError("Shadow progress source identity changed")
    offset = int(progress["next_offset"])
    while offset < SHADOW_ROWS:
        stop = min(offset + shard_size, SHADOW_ROWS)
        graphs = []
        for slot, initial_index in enumerate(split["shadow"][offset:stop], start=offset):
            row_index = int(initial_index)
            original_error = None
            while True:
                row = frame.iloc[row_index]
                try:
                    graph = _make_unlabeled_graph(row)
                    break
                except Exception as error:
                    progress["failures"].append(
                        {
                            "slot": slot,
                            "row_index": row_index,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    original_error = error
                    cursor = int(progress["reserve_cursor"])
                    if cursor >= SHADOW_RESERVE_ROWS:
                        raise RuntimeError("Shadow reserve exhausted") from error
                    row_index = int(split["reserve"][cursor])
                    progress["reserve_cursor"] = cursor + 1
            if row_index != int(initial_index) and original_error is None:
                raise RuntimeError("Shadow replacement accounting changed")
            progress["effective_indices"].append(row_index)
            graphs.append(graph)
        part = output_dir / f"shadow_part_{offset // shard_size:03d}.pt"
        _atomic_torch_save(part, graphs)
        progress["shards"].append(
            {
                "role": "shadow",
                "file": part.name,
                "graph_count": len(graphs),
                "sha256": sha256_file(part),
            }
        )
        offset = stop
        progress.update(
            {
                "format": "molgap-pcqm-k1-shadow-progress-v1",
                "source_commit": source_commit,
                "next_offset": offset,
                "shadow_labels_read": False,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
                "gpu_used": False,
            }
        )
        _atomic_json(progress_path, progress)
        _atomic_json(
            failures_path,
            {
                "format": "molgap-pcqm-k1-shadow-failures-v1",
                "failures": progress["failures"],
            },
        )

    effective = [int(value) for value in progress["effective_indices"]]
    if len(effective) != SHADOW_ROWS or len(set(effective)) != SHADOW_ROWS:
        raise RuntimeError("Shadow effective role is incomplete or duplicated")
    if any(value in used for value in effective):
        raise RuntimeError("Effective shadow overlaps parent roles")
    split_payload = {
        "format": "molgap-pcqm-k1-shadow-split-v1",
        "split_seed": SHADOW_SPLIT_SEED,
        "initial_shadow": [int(value) for value in split["shadow"]],
        "reserve": [int(value) for value in split["reserve"]],
        "effective_shadow": effective,
        "initial_shadow_sha256": split["shadow_sha256"],
        "reserve_sha256": split["reserve_sha256"],
        "effective_shadow_sha256": index_sha256(effective),
        "parent_role_count": len(used),
        "parent_role_index_sha256": index_sha256(sorted(used)),
        "shadow_labels_read": False,
    }
    _atomic_json(split_path, split_payload)
    aggregate = hashlib.sha256()
    for shard in progress["shards"]:
        aggregate.update(
            f"shadow\t{shard['file']}\t{shard['sha256']}\n".encode("ascii")
        )
    manifest = {
        "format": "molgap-pcqm-k1-shadow-cache-v1",
        "complete": True,
        "source_dataset": source_dataset,
        "source_commit": source_commit,
        "frozen_candidate": FROZEN_CANDIDATE,
        "candidate_selected_before_shadow_labels": True,
        "official_train_rows_read": OFFICIAL_TRAIN_ROWS,
        "csv_columns_read": ["idx", "smiles"],
        "parent_graph_cache_aggregate_sha256": PARENT_GRAPH_CACHE_SHA256,
        "parent_geometry_cache_aggregate_sha256": GEOMETRY_CACHE_SHA256,
        "shadow_graphs": SHADOW_ROWS,
        "split_file": split_path.name,
        "split_file_sha256": sha256_file(split_path),
        "effective_shadow_index_sha256": split_payload["effective_shadow_sha256"],
        "failures_file": failures_path.name,
        "failures_file_sha256": sha256_file(failures_path),
        "failure_count": len(progress["failures"]),
        "reserve_rows_consumed": int(progress["reserve_cursor"]),
        "atom_feature_dim": 9,
        "bond_feature_dim": 3,
        "rwse_dim": RWSE_DIM,
        "shards": progress["shards"],
        "aggregate_sha256": aggregate.hexdigest(),
        "shadow_labels_read": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "model_inference_executed": False,
        "gpu_used": False,
    }
    _atomic_json(manifest_path, manifest)
    return manifest
