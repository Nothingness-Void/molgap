"""Extract an explicit split from a source-indexed PyG graph cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    split = pd.read_csv(args.split_csv)
    required = {"source_idx", "split", "homo", "lumo", "gap"}
    if not required.issubset(split.columns):
        raise ValueError(f"Split CSV is missing {sorted(required - set(split.columns))}")
    if split.source_idx.duplicated().any():
        raise ValueError("Split CSV contains duplicate source_idx values")

    graphs = torch.load(args.graphs, map_location="cpu", weights_only=False)
    graph_map = {
        int(graph.source_idx.view(-1)[0]): graph
        for graph in graphs
    }
    if len(graph_map) != len(graphs):
        raise ValueError("Input graph cache contains duplicate source_idx values")
    requested = split.source_idx.to_numpy(dtype=np.int64)
    missing = [int(value) for value in requested if int(value) not in graph_map]
    if missing:
        raise ValueError(f"Input graph cache is missing {len(missing)} split rows")
    selected = [graph_map[int(value)] for value in requested]
    labels = np.stack(
        [graph.y.view(-1).numpy() for graph in selected]
    ).astype(np.float64)
    expected = split[["homo", "lumo", "gap"]].to_numpy(dtype=np.float64)
    max_label_difference = float(np.max(np.abs(labels - expected)))
    if max_label_difference > 1e-5:
        raise ValueError(
            f"Graph/CSV labels differ by up to {max_label_difference:.6g} eV"
        )
    if not np.isfinite(labels).all():
        raise ValueError("Selected graph labels are non-finite")

    atomic_torch(selected, args.output)
    report = {
        "input_graphs": str(args.graphs),
        "input_graph_rows": int(len(graphs)),
        "split_csv": str(args.split_csv),
        "split_csv_sha256": sha256(args.split_csv),
        "output": str(args.output),
        "output_rows": int(len(selected)),
        "output_sha256": sha256(args.output),
        "split_rows": {
            role: int(split.split.eq(role).sum())
            for role in ("train", "validation", "test")
        },
        "source_idx_unique": len(set(requested.tolist())) == len(requested),
        "max_label_difference_eV": max_label_difference,
        "finite_labels": True,
    }
    atomic_json(report, args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
