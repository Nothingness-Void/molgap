"""Immutable official-train-derived PCQM Gap architecture-screen cache."""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path


OFFICIAL_TRAIN_ROWS = 3_378_606
SCREEN_TRAIN_ROWS = 100_000
SCREEN_VALIDATION_ROWS = 10_000
SCREEN_SPLIT_SEED = 42
RWSE_DIM = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_sha256(indices) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
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


def _open_csv(path: Path):
    if not zipfile.is_zipfile(path):
        return path.open("rb")
    archive = zipfile.ZipFile(path)
    members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if len(members) != 1:
        archive.close()
        raise RuntimeError(f"Expected one CSV member in {path}, found {members}")
    handle = archive.open(members[0], "r")

    class _ArchiveHandle:
        def __enter__(self):
            return handle

        def __exit__(self, exc_type, exc, traceback):
            handle.close()
            archive.close()

    return _ArchiveHandle()


def read_official_train_prefix(path: Path):
    """Read exactly the official training prefix and no later role."""
    import numpy as np
    import pandas as pd

    with _open_csv(path) as handle:
        frame = pd.read_csv(
            handle,
            nrows=OFFICIAL_TRAIN_ROWS,
            usecols=["idx", "smiles", "homolumogap"],
        )
    if len(frame) != OFFICIAL_TRAIN_ROWS:
        raise RuntimeError(
            f"Official training prefix has {len(frame)} rows, expected "
            f"{OFFICIAL_TRAIN_ROWS}"
        )
    indices = frame["idx"].to_numpy(dtype=np.int64, copy=False)
    expected = np.arange(OFFICIAL_TRAIN_ROWS, dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise RuntimeError("PCQM index column is not the official contiguous prefix")
    targets = pd.to_numeric(frame["homolumogap"], errors="coerce").to_numpy()
    if not np.isfinite(targets).all():
        raise RuntimeError("Official PCQM training prefix contains non-finite targets")
    if frame["smiles"].isna().any():
        raise RuntimeError("Official PCQM training prefix contains missing SMILES")
    return frame


def fixed_screen_split(total_rows: int = OFFICIAL_TRAIN_ROWS) -> dict:
    import numpy as np

    if total_rows != OFFICIAL_TRAIN_ROWS:
        raise ValueError("The PCQM screen is pinned to the official training row count")
    generator = np.random.default_rng(SCREEN_SPLIT_SEED)
    selected = generator.choice(
        total_rows,
        size=SCREEN_TRAIN_ROWS + SCREEN_VALIDATION_ROWS,
        replace=False,
    )
    train = np.sort(selected[:SCREEN_TRAIN_ROWS]).astype(np.int64)
    validation = np.sort(selected[SCREEN_TRAIN_ROWS:]).astype(np.int64)
    if np.intersect1d(train, validation).size:
        raise RuntimeError("PCQM screen split roles overlap")
    return {
        "seed": SCREEN_SPLIT_SEED,
        "train": train,
        "validation": validation,
        "train_sha256": index_sha256(train),
        "validation_sha256": index_sha256(validation),
    }


def _selected_rows_sha256(frame, split: dict) -> str:
    digest = hashlib.sha256()
    for role in ("train", "validation"):
        for position in split[role]:
            row = frame.iloc[int(position)]
            line = (
                f"{role}\t{int(row.idx)}\t{row.smiles}\t"
                f"{float(row.homolumogap):.17g}\n"
            )
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _make_graph(row):
    import torch
    from ogb.utils.mol import smiles2graph
    from torch_geometric.data import Data
    from torch_geometric.transforms import AddRandomWalkPE

    payload = smiles2graph(str(row.smiles))
    graph = Data(
        x=torch.as_tensor(payload["node_feat"], dtype=torch.long),
        edge_index=torch.as_tensor(payload["edge_index"], dtype=torch.long),
        edge_attr=torch.as_tensor(payload["edge_feat"], dtype=torch.long),
        y=torch.tensor([float(row.homolumogap)], dtype=torch.float32),
        row_index=torch.tensor([int(row.idx)], dtype=torch.long),
    )
    graph = AddRandomWalkPE(
        walk_length=RWSE_DIM,
        attr_name="random_walk_pe",
    )(graph)
    if graph.x.ndim != 2 or graph.x.shape[1] != 9:
        raise RuntimeError(f"Unexpected atom feature shape: {tuple(graph.x.shape)}")
    if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != 3:
        raise RuntimeError(
            f"Unexpected bond feature shape: {tuple(graph.edge_attr.shape)}"
        )
    if tuple(graph.random_walk_pe.shape) != (graph.num_nodes, RWSE_DIM):
        raise RuntimeError("Unexpected RWSE shape")
    if not torch.isfinite(graph.random_walk_pe).all():
        raise RuntimeError("RWSE contains non-finite values")
    return graph


def build_pcqm_gap_screen_cache(
    source_csv: Path,
    output_dir: Path,
    *,
    source_dataset: str,
    source_commit: str,
    shard_size: int = 5_000,
) -> dict:
    """Build resumable role-sharded PyG graphs from the official train prefix."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is True:
            return manifest
        raise RuntimeError("Incomplete final manifest exists; inspect before resuming")

    frame = read_official_train_prefix(source_csv)
    split = fixed_screen_split(len(frame))
    split_payload = {
        "format": "molgap-pcqm-gap100k-split-v1",
        "official_train_rows": OFFICIAL_TRAIN_ROWS,
        "seed": SCREEN_SPLIT_SEED,
        "train": [int(value) for value in split["train"]],
        "validation": [int(value) for value in split["validation"]],
        "train_sha256": split["train_sha256"],
        "validation_sha256": split["validation_sha256"],
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    split_path = output_dir / "split.json"
    atomic_json(split_path, split_payload)

    failures_path = output_dir / "failures.json"
    progress_path = output_dir / "progress.json"
    failures = []
    completed_shards = []
    start_offsets = {"train": 0, "validation": 0}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        failures = progress.get("failures", [])
        completed_shards = progress.get("shards", [])
        start_offsets.update(progress.get("next_offset", {}))

    ranges = {
        "atom_feature_min": [None] * 9,
        "atom_feature_max": [None] * 9,
        "bond_feature_min": [None] * 3,
        "bond_feature_max": [None] * 3,
    }
    for role in ("train", "validation"):
        positions = split[role]
        offset = int(start_offsets[role])
        while offset < len(positions):
            stop = min(offset + shard_size, len(positions))
            graphs = []
            for position in positions[offset:stop]:
                row = frame.iloc[int(position)]
                try:
                    graph = _make_graph(row)
                except Exception as error:
                    failures.append(
                        {
                            "role": role,
                            "row_index": int(row.idx),
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    continue
                for key, values in (
                    ("atom", graph.x),
                    ("bond", graph.edge_attr),
                ):
                    minimum = values.min(dim=0).values.tolist()
                    maximum = values.max(dim=0).values.tolist()
                    min_key = f"{key}_feature_min"
                    max_key = f"{key}_feature_max"
                    ranges[min_key] = [
                        int(value) if old is None else min(old, int(value))
                        for old, value in zip(ranges[min_key], minimum)
                    ]
                    ranges[max_key] = [
                        int(value) if old is None else max(old, int(value))
                        for old, value in zip(ranges[max_key], maximum)
                    ]
                graphs.append(graph)
            part_number = offset // shard_size
            part_path = output_dir / f"{role}_part_{part_number:03d}.pt"
            _atomic_torch_save(part_path, graphs)
            record = {
                "role": role,
                "file": part_path.name,
                "source_start": offset,
                "source_stop": stop,
                "graph_count": len(graphs),
                "sha256": sha256_file(part_path),
            }
            completed_shards = [
                item for item in completed_shards if item["file"] != part_path.name
            ]
            completed_shards.append(record)
            offset = stop
            start_offsets[role] = offset
            atomic_json(failures_path, {"failures": failures})
            atomic_json(
                progress_path,
                {
                    "format": "molgap-pcqm-gap100k-progress-v1",
                    "source_commit": source_commit,
                    "next_offset": start_offsets,
                    "shards": completed_shards,
                    "failures": failures,
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                },
            )
            del graphs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    completed_shards.sort(key=lambda item: (item["role"], item["file"]))
    aggregate = hashlib.sha256()
    for item in completed_shards:
        aggregate.update(
            f"{item['role']}\t{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    role_counts = {
        role: sum(
            item["graph_count"] for item in completed_shards if item["role"] == role
        )
        for role in ("train", "validation")
    }
    manifest = {
        "format": "molgap-pcqm-gap100k-cache-v1",
        "complete": True,
        "source_dataset": source_dataset,
        "source_file": source_csv.name,
        "source_commit": source_commit,
        "official_train_rows_read": OFFICIAL_TRAIN_ROWS,
        "selected_rows_sha256": _selected_rows_sha256(frame, split),
        "split_file": split_path.name,
        "split_file_sha256": sha256_file(split_path),
        "split_seed": SCREEN_SPLIT_SEED,
        "train_index_sha256": split["train_sha256"],
        "validation_index_sha256": split["validation_sha256"],
        "train_graphs": role_counts["train"],
        "validation_graphs": role_counts["validation"],
        "failed_graphs": len(failures),
        "atom_feature_dim": 9,
        "bond_feature_dim": 3,
        "rwse_dim": RWSE_DIM,
        "feature_ranges": ranges,
        "shards": completed_shards,
        "aggregate_sha256": aggregate.hexdigest(),
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "gpu_used": False,
    }
    atomic_json(manifest_path, manifest)
    return manifest

