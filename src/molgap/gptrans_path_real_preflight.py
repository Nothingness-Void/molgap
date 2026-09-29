"""Label-free path-information check on the accepted fixed PCQM graph cache.

This is an input preflight, not a GPTrans accuracy or official-preprocessing
reproduction. The pinned author Cython parity check is a separate experiment.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path


MANIFEST_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
SAMPLE_PER_SHARD = 256
MAX_DISTANCE = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def graph_path_counts(node_count: int, edges: list[tuple[int, int, tuple[int, ...]]]) -> dict:
    """Count one deterministic shortest-path signature per undirected pair.

    A path tie may resolve differently in the author's Floyd-Warshall code;
    this deliberately tests available input information, not byte parity.
    """
    if node_count < 1:
        raise ValueError("empty graph")
    neighbors: list[dict[int, tuple[int, ...]]] = [dict() for _ in range(node_count)]
    for source, target, bond in edges:
        if not (0 <= source < node_count and 0 <= target < node_count) or source == target:
            raise ValueError("invalid bond endpoint")
        if not bond:
            raise ValueError("empty bond category")
        previous = neighbors[source].get(target)
        if previous is not None and previous != bond:
            raise ValueError("conflicting directed bond features")
        neighbors[source][target] = bond
    for source, row in enumerate(neighbors):
        for target, bond in row.items():
            if neighbors[target].get(source) != bond:
                raise ValueError("asymmetric bond graph")

    by_distance: dict[int, dict[tuple[tuple[int, ...], ...], int]] = {}
    connected_nonbond_pairs = 0
    capped_nonbond_pairs = 0
    nontrivial_bond_pairs = 0
    for source in range(node_count):
        distances = {source: 0}
        signatures: dict[int, tuple[tuple[int, ...], ...]] = {source: ()}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for target in sorted(neighbors[current]):
                if target not in distances:
                    distances[target] = distances[current] + 1
                    signatures[target] = signatures[current] + (neighbors[current][target],)
                    queue.append(target)
        for target in range(source + 1, node_count):
            distance = distances.get(target)
            if distance is None or distance < 2:
                continue
            connected_nonbond_pairs += 1
            if distance > MAX_DISTANCE:
                continue
            capped_nonbond_pairs += 1
            signature = signatures[target]
            by_distance.setdefault(distance, {})[signature] = (
                by_distance.setdefault(distance, {}).get(signature, 0) + 1
            )
            if any(bond[0] != 0 for bond in signature):
                nontrivial_bond_pairs += 1
    discriminated_pairs = sum(
        sum(counts.values()) for counts in by_distance.values() if len(counts) > 1
    )
    return {
        "connected_nonbond_pairs": connected_nonbond_pairs,
        "capped_nonbond_pairs": capped_nonbond_pairs,
        "nontrivial_bond_pairs": nontrivial_bond_pairs,
        "same_distance_multiple_signature_pairs": discriminated_pairs,
        "distances_with_multiple_signatures": sum(len(counts) > 1 for counts in by_distance.values()),
    }


def _sample_positions(length: int) -> list[int]:
    if length < SAMPLE_PER_SHARD:
        raise ValueError("fixed shard is too short")
    return [(index * length) // SAMPLE_PER_SHARD for index in range(SAMPLE_PER_SHARD)]


def run(dataset_root: Path, output: Path) -> dict:
    import torch
    from torch_geometric.data import InMemoryDataset

    manifest_path = dataset_root / "manifest.json"
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("fixed 100K manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "molgap-pcqm4mv2-kaggle-fixed-subset-v1":
        raise RuntimeError("fixed manifest format changed")
    if manifest.get("identity", {}).get("name") != "ogb-train-100k":
        raise RuntimeError("not the accepted 100K dataset")
    records = [item for item in manifest["geometry_shards"] if item["role"] == "train"]
    if len(records) != 2:
        raise RuntimeError("expected two fixed train shards")

    class PackedGraphs(InMemoryDataset):
        def __init__(self, path: Path):
            super().__init__(root=None)
            self.data, self.slices = torch.load(path, map_location="cpu", weights_only=False)

    summary = {
        "format": "molgap-gptrans-real-path-preflight-v1",
        "status": "RUNNING",
        "manifest_sha256": MANIFEST_SHA256,
        "sample_per_shard": SAMPLE_PER_SHARD,
        "maximum_distance": MAX_DISTANCE,
        "labels_read": False,
        "checkpoint_loaded": False,
        "official_evaluation_read": False,
        "shards": [],
    }
    atomic_json(output / "summary.json", summary)
    for ordinal, record in enumerate(records):
        path = (dataset_root / record["file"]).resolve()
        if not path.is_relative_to(dataset_root.resolve()):
            raise RuntimeError("shard escaped fixed dataset root")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError("fixed train shard identity changed")
        graphs = PackedGraphs(path)
        positions = _sample_positions(len(graphs))
        totals = {key: 0 for key in (
            "connected_nonbond_pairs", "capped_nonbond_pairs", "nontrivial_bond_pairs",
            "same_distance_multiple_signature_pairs", "distances_with_multiple_signatures",
        )}
        source_indices = []
        for position in positions:
            graph = graphs[position]
            source_idx = getattr(graph, "source_idx", getattr(graph, "row_index", None))
            if source_idx is None:
                raise RuntimeError("fixed graph lacks source identity")
            observed = int(source_idx.reshape(-1)[0])
            expected = int(record["source_idx_min"]) + position
            if observed != expected:
                raise RuntimeError("fixed graph source row changed")
            source_indices.append(observed)
            edge_index = graph.edge_index.tolist()
            edge_attr = graph.edge_attr.tolist()
            edges = [
                (int(source), int(target), tuple(int(v) for v in category))
                for source, target, category in zip(edge_index[0], edge_index[1], edge_attr, strict=True)
            ]
            counts = graph_path_counts(int(graph.x.shape[0]), edges)
            for key, value in counts.items():
                totals[key] += value
        item = {
            "shard_ordinal": ordinal,
            "shard_sha256": record["sha256"],
            "sampled_rows": len(positions),
            "source_indices_sha256": hashlib.sha256(
                json.dumps(source_indices, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "source_idx_first": source_indices[0],
            "source_idx_last": source_indices[-1],
            "counts": totals,
        }
        atomic_json(output / f"train_shard_{ordinal}.json", item)
        summary["shards"].append(item)
        atomic_json(output / "summary.json", summary)
        del graphs
    summary["status"] = "COMPLETE"
    atomic_json(output / "summary.json", summary)
    return summary
