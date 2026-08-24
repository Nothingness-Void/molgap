"""Durable positional-encoding caches for structural graph models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import torch
from torch_geometric.transforms import AddRandomWalkPE


RWSE_ATTRIBUTE = "random_walk_pe"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def add_random_walk_pe(graph, walk_length: int = 16):
    """Return one graph with finite ``random_walk_pe`` node features."""
    if walk_length <= 0:
        raise ValueError("walk_length must be positive")
    existing = getattr(graph, RWSE_ATTRIBUTE, None)
    if existing is not None:
        if existing.shape != (graph.num_nodes, walk_length):
            raise ValueError("Existing random_walk_pe has an incompatible shape")
        if not torch.isfinite(existing).all():
            raise ValueError("Existing random_walk_pe contains non-finite values")
        return graph

    graph = AddRandomWalkPE(
        walk_length=walk_length,
        attr_name=RWSE_ATTRIBUTE,
    )(graph)
    encoding = getattr(graph, RWSE_ATTRIBUTE)
    if encoding.shape != (graph.num_nodes, walk_length):
        raise RuntimeError("Random-walk transform returned an unexpected shape")
    if not torch.isfinite(encoding).all():
        raise RuntimeError("Random-walk transform returned non-finite values")
    return graph


def _source_idx(graph) -> int:
    value = getattr(graph, "source_idx", None)
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError("Every graph must contain one tensor source_idx")
    return int(value.view(-1)[0])


def _valid_part(path: Path, sidecar: Path, contract: dict) -> dict | None:
    if not path.is_file() or not sidecar.is_file():
        return None
    report = json.loads(sidecar.read_text(encoding="utf-8"))
    if any(report.get(key) != value for key, value in contract.items()):
        return None
    if report.get("sha256") != sha256(path):
        return None
    return report


def _iter_ranges(rows: int, shard_size: int) -> Iterable[tuple[int, int, int]]:
    for part, start in enumerate(range(0, rows, shard_size)):
        yield part, start, min(start + shard_size, rows)


def build_rwse_graph_cache(
    input_path: Path,
    output_path: Path,
    progress_dir: Path,
    *,
    walk_length: int = 16,
    shard_size: int = 10_000,
    max_graphs: int | None = None,
) -> dict:
    """Build resumable RWSE shards and one accepted monolithic graph cache."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    progress_dir = Path(progress_dir)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    input_hash = sha256(input_path)
    graphs = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("Input graph cache must be a non-empty list")
    rows = len(graphs) if max_graphs is None else min(len(graphs), int(max_graphs))
    if rows <= 0:
        raise ValueError("max_graphs selected no rows")
    graphs = graphs[:rows]
    source_indices = [_source_idx(graph) for graph in graphs]
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("Input graph cache contains duplicate source_idx values")

    progress_dir.mkdir(parents=True, exist_ok=True)
    part_reports = []
    for part, start, end in _iter_ranges(rows, shard_size):
        part_path = progress_dir / f"graphs_{start:07d}_{end:07d}.pt"
        sidecar = part_path.with_suffix(".json")
        contract = {
            "format": "molgap-rwse-part-v1",
            "input_sha256": input_hash,
            "walk_length": int(walk_length),
            "start": start,
            "end": end,
            "rows": end - start,
            "first_source_idx": source_indices[start],
            "last_source_idx": source_indices[end - 1],
        }
        report = _valid_part(part_path, sidecar, contract)
        if report is None:
            encoded = [
                add_random_walk_pe(graph, walk_length=walk_length)
                for graph in graphs[start:end]
            ]
            atomic_torch_save(encoded, part_path)
            report = {
                **contract,
                "bytes": part_path.stat().st_size,
                "sha256": sha256(part_path),
            }
            atomic_json(report, sidecar)
        part_reports.append(report)
        print(f"RWSE shard {part + 1}: {start:,}:{end:,}", flush=True)

    merged = []
    for report in part_reports:
        part_path = progress_dir / f"graphs_{report['start']:07d}_{report['end']:07d}.pt"
        encoded = torch.load(part_path, map_location="cpu", weights_only=False)
        if len(encoded) != report["rows"]:
            raise RuntimeError(f"RWSE shard row mismatch: {part_path}")
        merged.extend(encoded)
    if len(merged) != rows:
        raise RuntimeError("RWSE parts do not cover the selected graph rows")
    atomic_torch_save(merged, output_path)
    manifest = {
        "format": "molgap-rwse-cache-v1",
        "complete": True,
        "input_path": str(input_path),
        "input_sha256": input_hash,
        "output_path": str(output_path),
        "output_sha256": sha256(output_path),
        "output_bytes": output_path.stat().st_size,
        "walk_length": int(walk_length),
        "rows": rows,
        "first_source_idx": source_indices[0],
        "last_source_idx": source_indices[-1],
        "parts": part_reports,
    }
    atomic_json(manifest, output_path.with_suffix(".manifest.json"))
    return manifest
