"""Deterministic MoSE counts and cache attachment for fixed PCQM screens.

The pattern order is derived from the official ICLR 2025 MoSE repository at
commit ``1cf3ea117e83e5db9f96d754743707e3dc23cbad``.  Its ``all_5vertex``
basis contains the 30 connected, unlabelled patterns on two through five
vertices.  The published 31-channel molecular setup adds an anchored six-cycle.

Only graph adjacency is used.  Atom labels, bond labels, targets, geometry and
protected roles cannot influence these counts.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable


MOSE_DIM = 31
MOSE_UPSTREAM_COMMIT = "1cf3ea117e83e5db9f96d754743707e3dc23cbad"

# Node zero is the distinguished/root vertex, matching the official basis.
_ALL5_EDGES = (
    ((0, 1),),
    ((0, 2), (1, 2)),
    ((0, 1), (0, 2), (1, 2)),
    ((0, 3), (1, 3), (2, 3)),
    ((0, 2), (0, 3), (1, 3)),
    ((0, 2), (0, 3), (1, 3), (2, 3)),
    ((0, 2), (0, 3), (1, 2), (1, 3)),
    ((0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    ((0, 4), (1, 4), (2, 4), (3, 4)),
    ((0, 3), (0, 4), (1, 4), (2, 4)),
    ((0, 3), (0, 4), (1, 4), (2, 4), (3, 4)),
    ((0, 3), (0, 4), (1, 3), (1, 4), (2, 4)),
    ((0, 3), (0, 4), (1, 3), (2, 4), (3, 4)),
    ((0, 3), (0, 4), (1, 3), (1, 4), (2, 4), (3, 4)),
    ((0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)),
    ((0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)),
    ((0, 2), (0, 4), (1, 3), (1, 4)),
    ((0, 2), (0, 4), (1, 3), (1, 4), (2, 4)),
    ((0, 2), (0, 4), (1, 3), (1, 4), (2, 4), (3, 4)),
    ((0, 2), (0, 3), (1, 3), (1, 4), (2, 4)),
    ((0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4)),
    ((0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4), (3, 4)),
    ((0, 2), (0, 3), (0, 4), (1, 4), (2, 3), (2, 4)),
    ((0, 2), (0, 3), (0, 4), (1, 4), (2, 3), (2, 4), (3, 4)),
    ((0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)),
    ((0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 4)),
    ((0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)),
    ((0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)),
    ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)),
)
_C6_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5))
MOSE_PATTERNS = _ALL5_EDGES + (_C6_EDGES,)


def pattern_fingerprint() -> str:
    payload = json.dumps(MOSE_PATTERNS, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _pattern_plan(edges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], ...]:
    nodes = 1 + max(max(edge) for edge in edges)
    adjacency = [set() for _ in range(nodes)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    order = [0]
    remaining = set(range(1, nodes))
    while remaining:
        chosen = max(
            remaining,
            key=lambda node: (
                len(adjacency[node].intersection(order)),
                len(adjacency[node]),
                -node,
            ),
        )
        if not adjacency[chosen].intersection(order):
            raise RuntimeError("MoSE pattern is not connected")
        order.append(chosen)
        remaining.remove(chosen)
    return tuple(tuple(sorted(neighbors)) for neighbors in adjacency), tuple(order)


_PATTERN_PLANS = tuple(_pattern_plan(edges) for edges in MOSE_PATTERNS)


def rooted_homomorphism_counts(
    num_nodes: int,
    edge_index: Iterable[Iterable[int]],
) -> list[list[int]]:
    """Return one exact 31-channel rooted homomorphism vector per host node."""
    if num_nodes <= 0:
        raise ValueError("Host graph must contain at least one node")
    rows = [tuple(int(value) for value in row) for row in edge_index]
    if len(rows) != 2 or len(rows[0]) != len(rows[1]):
        raise ValueError("edge_index must be a 2xE sequence")
    host_adjacency = [0] * num_nodes
    for left, right in zip(rows[0], rows[1]):
        if not 0 <= left < num_nodes or not 0 <= right < num_nodes:
            raise ValueError("edge_index contains an out-of-range node")
        if left == right:
            continue
        host_adjacency[left] |= 1 << right
        host_adjacency[right] |= 1 << left

    output = [[0] * MOSE_DIM for _ in range(num_nodes)]
    all_hosts = (1 << num_nodes) - 1
    for channel, (pattern_adjacency, order) in enumerate(_PATTERN_PLANS):
        pattern_nodes = len(pattern_adjacency)
        mapping = [-1] * pattern_nodes

        def visit(position: int) -> int:
            if position == pattern_nodes:
                return 1
            pattern_node = order[position]
            candidates = all_hosts
            for neighbor in pattern_adjacency[pattern_node]:
                mapped = mapping[neighbor]
                if mapped >= 0:
                    candidates &= host_adjacency[mapped]
            total = 0
            while candidates:
                bit = candidates & -candidates
                mapping[pattern_node] = bit.bit_length() - 1
                total += visit(position + 1)
                candidates ^= bit
            mapping[pattern_node] = -1
            return total

        for root in range(num_nodes):
            mapping[0] = root
            output[root][channel] = visit(1)
    return output


def _count_graph(arguments):
    source_idx, num_nodes, edge_index = arguments
    return source_idx, rooted_homomorphism_counts(num_nodes, edge_index)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_mose_cache(output_root: Path, workers: int = 4) -> dict:
    """Build an independently retrievable cache from the accepted fixed graphs."""
    import torch

    from .pcqm_k1_variants_runner import (
        FIXED_GEOMETRY_SHA256,
        FIXED_MANIFEST_SHA256,
        _PackedGraphDatasetFactory,
        find_fixed_cache,
    )

    fixed_root, fixed_manifest = find_fixed_cache()
    output_root.mkdir(parents=True, exist_ok=True)
    parts = []
    aggregate = hashlib.sha256()
    total_rows = 0
    total_nodes = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for part_index, item in enumerate(fixed_manifest["geometry_shards"]):
            dataset = _PackedGraphDatasetFactory.load(fixed_root / item["file"])
            jobs = (
                (
                    int(graph.source_idx.view(-1)[0]),
                    int(graph.num_nodes),
                    graph.edge_index.detach().cpu().tolist(),
                )
                for graph in dataset
            )
            source_indices = []
            pointers = [0]
            count_blocks = []
            for source_idx, counts in executor.map(_count_graph, jobs, chunksize=32):
                tensor = torch.tensor(counts, dtype=torch.int32)
                if tensor.ndim != 2 or tensor.shape[1] != MOSE_DIM:
                    raise RuntimeError("MoSE counter returned an invalid shape")
                source_indices.append(source_idx)
                pointers.append(pointers[-1] + tensor.shape[0])
                count_blocks.append(tensor)
            payload = {
                "format": "molgap-pcqm-mose-part-v1",
                "role": item["role"],
                "source_idx": torch.tensor(source_indices, dtype=torch.int64),
                "node_ptr": torch.tensor(pointers, dtype=torch.int64),
                "counts": torch.cat(count_blocks, dim=0),
                "pattern_sha256": pattern_fingerprint(),
            }
            filename = f"mose_{part_index:03d}_{item['role']}.pt"
            temporary = output_root / (filename + ".tmp")
            final = output_root / filename
            torch.save(payload, temporary)
            os.replace(temporary, final)
            digest = _sha256(final)
            row = {
                "file": filename,
                "role": item["role"],
                "rows": len(source_indices),
                "nodes": int(pointers[-1]),
                "source_start": source_indices[0],
                "source_stop": source_indices[-1] + 1,
                "sha256": digest,
                "fixed_graph_file": item["file"],
                "fixed_graph_sha256": item["sha256"],
            }
            parts.append(row)
            aggregate.update(
                f"{filename}\t{digest}\t{row['source_start']}\t{row['source_stop']}\n".encode("ascii")
            )
            total_rows += len(source_indices)
            total_nodes += pointers[-1]
            _atomic_json(
                output_root / "progress.json",
                {"complete": False, "parts": parts, "rows": total_rows, "nodes": total_nodes},
            )
    manifest = {
        "format": "molgap-pcqm-mose-cache-v1",
        "complete": True,
        "feature_dim": MOSE_DIM,
        "feature_transform_at_training": "log1p-float32",
        "pattern_sha256": pattern_fingerprint(),
        "upstream_repository": "https://github.com/teriolx/graph-encoding-GT",
        "upstream_commit": MOSE_UPSTREAM_COMMIT,
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "fixed_geometry_sha256": FIXED_GEOMETRY_SHA256,
        "rows": total_rows,
        "nodes": total_nodes,
        "parts": parts,
        "aggregate_sha256": aggregate.hexdigest(),
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    _atomic_json(output_root / "mose_manifest.json", manifest)
    return manifest


def find_mose_cache() -> tuple[Path, dict]:
    candidates = []
    explicit = os.environ.get("MOLGAP_MOSE_CACHE_ROOT")
    roots = [Path(explicit)] if explicit else [Path("/kaggle/input")]
    for root in roots:
        paths = [root / "mose_manifest.json"] if (root / "mose_manifest.json").is_file() else root.rglob("mose_manifest.json")
        for path in paths:
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("format") == "molgap-pcqm-mose-cache-v1":
                candidates.append((path.parent, manifest))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one accepted MoSE cache, found {candidates}")
    return candidates[0]


class MoSEGraphDataset:
    """Replace RWSE with aligned MoSE counts without mutating fixed graphs."""

    def __init__(self, base, payload):
        self.base = base
        self.payload = payload
        self._data = base._data
        if len(base) != int(payload["source_idx"].numel()):
            raise RuntimeError("MoSE/base part row mismatch")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        import copy
        import torch

        graph = copy.copy(self.base[index])
        expected = int(self.payload["source_idx"][index])
        actual = int(graph.source_idx.view(-1)[0])
        if actual != expected:
            raise RuntimeError("MoSE/base source index mismatch")
        start = int(self.payload["node_ptr"][index])
        stop = int(self.payload["node_ptr"][index + 1])
        counts = self.payload["counts"][start:stop]
        if counts.shape != (graph.num_nodes, MOSE_DIM):
            raise RuntimeError("MoSE/base node alignment mismatch")
        graph.random_walk_pe = torch.log1p(counts.float())
        return graph


def attach_mose_roles(roles, fixed_manifest: dict):
    import torch
    from torch.utils.data import ConcatDataset

    from .pcqm_k1_variants_runner import FIXED_GEOMETRY_SHA256, FIXED_MANIFEST_SHA256

    root, manifest = find_mose_cache()
    required = {
        "complete": True,
        "feature_dim": MOSE_DIM,
        "pattern_sha256": pattern_fingerprint(),
        "upstream_commit": MOSE_UPSTREAM_COMMIT,
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "fixed_geometry_sha256": FIXED_GEOMETRY_SHA256,
        "rows": 150_000,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"MoSE manifest mismatch: {key}")
    if len(manifest.get("parts", [])) != len(fixed_manifest["geometry_shards"]):
        raise RuntimeError("MoSE part inventory differs from fixed graph inventory")
    wrapped = {"train": [], "development": []}
    role_offsets = {"train": 0, "development": 0}
    aggregate = hashlib.sha256()
    for fixed_item, mose_item in zip(fixed_manifest["geometry_shards"], manifest["parts"]):
        if (
            mose_item["role"] != fixed_item["role"]
            or mose_item["fixed_graph_file"] != fixed_item["file"]
            or mose_item["fixed_graph_sha256"] != fixed_item["sha256"]
        ):
            raise RuntimeError("MoSE part no longer aligns with fixed graph part")
        path = root / mose_item["file"]
        if _sha256(path) != mose_item["sha256"]:
            raise RuntimeError("MoSE part hash changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "molgap-pcqm-mose-part-v1":
            raise RuntimeError("MoSE part format changed")
        role = mose_item["role"]
        base = roles[role].datasets[role_offsets[role]]
        role_offsets[role] += 1
        wrapped[role].append(MoSEGraphDataset(base, payload))
        aggregate.update(
            f"{mose_item['file']}\t{mose_item['sha256']}\t{mose_item['source_start']}\t{mose_item['source_stop']}\n".encode("ascii")
        )
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError("MoSE aggregate hash changed")
    return {role: ConcatDataset(parts) for role, parts in wrapped.items()}, manifest

