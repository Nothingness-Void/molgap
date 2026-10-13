"""CPU-only common-chordless-cycle inputs derived from accepted PCQM graphs.

No labels, geometry, model construction or platform operations are used here.
Independent NetworkX enumeration verifies every derived graph before publication.
"""
from __future__ import annotations

from dataclasses import replace
from bisect import bisect_right
import json
from pathlib import Path
import time

from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file

FORMAT = "molgap-common-cycle-sidecar-v1"
MAX_CYCLE = 18
FIXED_100K_SHA = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
FIXED_500K_SHA = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"


def adjacency(nodes, edges):
    if type(nodes) is not int or nodes < 1:
        raise ValueError("Invalid graph node count")
    result = [set() for _ in range(nodes)]
    for source, target in edges:
        if (type(source) is not int or type(target) is not int or
                not 0 <= source < nodes or not 0 <= target < nodes or source == target):
            raise ValueError("Expected finite simple-graph edges")
        result[source].add(target)
        result[target].add(source)
    return result


def chordless_cycles(neighbors, maximum=MAX_CYCLE):
    """Enumerate induced cycles once, with bounded paths and no node-ID features."""
    if type(maximum) is not int or not 3 <= maximum <= MAX_CYCLE:
        raise ValueError("Unsupported cycle cap")
    cycles = set()
    for root in range(len(neighbors)):
        def visit(path):
            last = path[-1]
            for node in sorted(neighbors[last]):
                if node <= root or node in path or len(path) >= maximum:
                    continue
                # Only the root may close the path; any other old neighbor is a chord.
                if any(node in neighbors[old] for old in path[1:-1]):
                    continue
                extended = (*path, node)
                if node in neighbors[root] and len(extended) >= 3:
                    if extended[1] < node:
                        cycles.add(tuple(sorted(extended)))
                else:
                    visit(extended)
        for first in sorted(neighbors[root]):
            if first > root:
                visit((root, first))
    return tuple(sorted(cycles))


def common_cycle_pairs(nodes, edges, *, independent_check=False):
    """Return ordered (source,target,category) for common-cycle pairs only."""
    neighbors = adjacency(nodes, edges)
    cycles = chordless_cycles(neighbors)
    if independent_check:
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(range(nodes))
        graph.add_edges_from((i, j) for i, row in enumerate(neighbors) for j in row if i < j)
        reference = tuple(sorted({tuple(sorted(cycle)) for cycle in
                                  nx.chordless_cycles(graph, length_bound=MAX_CYCLE)}))
        if cycles != reference:
            raise ValueError("Independent chordless-cycle enumeration differs")
    pairs = {(i, j) for cycle in cycles for i in cycle for j in cycle if i != j}
    return tuple((i, j, 1 if j in neighbors[i] else 2) for i, j in sorted(pairs))


def _confined(root, relative):
    from .experiment_preflight import _under
    return _under(root, relative)


def build_sidecar(selections, output, *, source_commit, source_archive_sha256,
                  maximum_wall_seconds=7200):
    """Stream accepted shards, independently rederive every row, retain chunks.

    selections contains (root, manifest_sha256, roles). Only those topology
    roles are deserialized; a500K mount never permits reading its train shards.
    """
    import torch
    from .packed_graph_dataset import PackedGraphDataset
    began = time.monotonic()
    if torch.cuda.is_available():
        raise ValueError("Cycle derivation requires a CPU-only allocation")
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    binding = {"format": FORMAT, "maximum_cycle": MAX_CYCLE,
               "source_commit": source_commit, "source_archive_sha256": source_archive_sha256,
               "selections": [{"manifest_sha256": sha, "roles": list(roles)}
                              for _, sha, roles in selections]}
    binding_path = output / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_bytes()) != binding:
        raise ValueError("Cannot reuse outputs from another source/input selection")
    atomic_json(binding_path, binding)
    chunks, total_rows, total_pairs = [], 0, 0
    for root, expected_sha, roles in selections:
        root = Path(root).absolute()
        manifest_path = root / "manifest.json"
        if sha256_file(manifest_path) != expected_sha or expected_sha not in (FIXED_100K_SHA, FIXED_500K_SHA):
            raise ValueError("Unaccepted fixed data identity")
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("status") != "complete" or manifest.get("source", {}).get("external_data_used") is not False:
            raise ValueError("Fixed asset is incomplete or external")
        permitted = ("train", "development") if expected_sha == FIXED_100K_SHA else ("development",)
        if tuple(roles) != permitted:
            raise ValueError("Cycle preparation role scope changed")
        from .pcqm_fixed_records import graph_records
        records = graph_records(manifest, expected_sha)
        for role in roles:
            declaration = manifest["roles"][role]
            cursor = declaration["source_idx_start"]
            for record in records:
                if record["role"] != role:
                    continue
                if time.monotonic() - began > maximum_wall_seconds:
                    raise TimeoutError("CPU budget exhausted; preserve independently retained chunks")
                origin = _confined(root, record["file"])
                if sha256_file(origin) != record["sha256"] or record["source_idx_min"] != cursor:
                    raise ValueError("Accepted topology bytes/order changed")
                dataset = PackedGraphDataset(origin, mmap=True)
                if len(dataset) != record["rows"]:
                    raise ValueError("Source shard count changed")
                ptr, pairs, categories, nodes = [0], [], [], []
                for offset in range(len(dataset)):
                    if time.monotonic() - began > maximum_wall_seconds:
                        raise TimeoutError("CPU budget exhausted during shard; retained earlier chunks remain valid")
                    graph = dataset[offset]
                    source_idx = int(graph.source_idx.view(-1)[0])
                    if source_idx != cursor + offset:
                        raise ValueError("Decoded source identity changed")
                    # Labels are carried by the trusted packed source but never inspected.
                    count = int(graph.x.shape[0])
                    derived = common_cycle_pairs(count, graph.edge_index.t().tolist(), independent_check=True)
                    pairs.extend((i, j) for i, j, _ in derived)
                    categories.extend(category for _, _, category in derived)
                    nodes.append(count)
                    ptr.append(len(pairs))
                payload = {"source_idx": torch.arange(cursor, cursor + len(dataset)),
                           "nodes": torch.tensor(nodes, dtype=torch.int64),
                           "ptr": torch.tensor(ptr, dtype=torch.int64),
                           "pair_index": torch.tensor(pairs, dtype=torch.int64).reshape(-1, 2).t().contiguous(),
                           "category": torch.tensor(categories, dtype=torch.uint8)}
                relative = f"chunks/cycle_{cursor:06d}_{cursor + len(dataset):06d}.pt"
                target = output / relative
                if target.exists():
                    retained = torch.load(target, map_location="cpu", weights_only=True)
                    if set(retained) != set(payload) or any(not torch.equal(retained[k], v) for k, v in payload.items()):
                        raise ValueError("Retained chunk differs from source rederivation")
                else:
                    atomic_torch_save(target, payload)
                chunks.append({"file": relative, "sha256": sha256_file(target), "rows": len(dataset),
                               "source_start": cursor, "source_stop": cursor + len(dataset),
                               "pairs": len(pairs), "source_payload_sha256": record["sha256"],
                               "source_manifest_sha256": expected_sha, "role": role,
                               "independently_rederived": True})
                cursor += len(dataset)
                total_rows += len(dataset)
                total_pairs += len(pairs)
                atomic_json(output / "progress.json", {**binding, "status": "partial", "rows": total_rows,
                    "chunks": chunks, "wall_seconds": time.monotonic() - began})
                print(json.dumps({"rows": total_rows, "chunks": len(chunks)}), flush=True)
                del dataset, payload
            if cursor != declaration["source_idx_stop"]:
                raise ValueError("Source role is incomplete")
    result = {**binding, "status": "complete", "rows": total_rows, "pairs": total_pairs,
              "chunks": chunks, "wall_seconds": time.monotonic() - began,
              "labels_used": False, "model_executed": False, "geometry_used": False,
              "gpu_used": False, "independent_algorithm": "networkx.chordless_cycles"}
    atomic_json(output / "manifest.json", result)
    return result


class CycleDataset:
    """Borrow base graph tensors, attach accepted sparse cycle indices only."""
    transform = None

    def __init__(self, dataset, chunks):
        self.dataset, self.chunks = dataset, chunks
        self.starts = [start for start, _, _ in chunks]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        from copy import copy
        source = self.dataset[index]
        row = int(source.source_idx.view(-1)[0])
        location = bisect_right(self.starts, row) - 1
        if location >= 0:
            start, stop, payload = self.chunks[location]
            if start <= row < stop:
                local = row - start
                if int(payload["nodes"][local]) != source.num_nodes:
                    raise ValueError("Sidecar node count changed")
                left, right = (int(payload["ptr"][local + i]) for i in (0, 1))
                result = copy(source)
                result.cycle_pair_index = payload["pair_index"][:, left:right]
                result.cycle_pair_category = payload["category"][left:right].long()
                return result
        raise ValueError("Missing immutable cycle row")


def validate_chunk(payload, record):
    """Fail closed on decoded tensor schema, bounds, symmetry and ordering."""
    import torch
    required = {"source_idx", "nodes", "ptr", "pair_index", "category"}
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("Cycle chunk schema changed")
    rows, pairs = record["rows"], record["pairs"]
    if record["source_stop"] - record["source_start"] != rows:
        raise ValueError("Cycle chunk row declaration changed")
    for key, shape, dtype in (("source_idx", (rows,), torch.int64),
            ("nodes", (rows,), torch.int64), ("ptr", (rows + 1,), torch.int64),
            ("pair_index", (2, pairs), torch.int64), ("category", (pairs,), torch.uint8)):
        value = payload[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError("Cycle tensor shape/dtype changed: " + key)
    if (not torch.equal(payload["source_idx"], torch.arange(record["source_start"], record["source_stop"]))
            or (payload["nodes"] < 1).any() or int(payload["ptr"][0]) != 0
            or int(payload["ptr"][-1]) != pairs or (payload["ptr"][1:] < payload["ptr"][:-1]).any()
            or ((payload["category"] < 1) | (payload["category"] > 2)).any()):
        raise ValueError("Cycle tensor values changed")
    for index in range(rows):
        left, right = (int(payload["ptr"][index + offset]) for offset in (0, 1))
        relations = payload["pair_index"][:, left:right].t().tolist()
        categories = payload["category"][left:right].tolist()
        count = int(payload["nodes"][index])
        if (relations != sorted(relations) or len({tuple(p) for p in relations}) != len(relations)
                or any(i == j or not 0 <= i < count or not 0 <= j < count for i, j in relations)):
            raise ValueError("Cycle relation indices changed")
        ordered = {(i, j, c) for (i, j), c in zip(relations, categories)}
        if ordered != {(j, i, c) for i, j, c in ordered}:
            raise ValueError("Cycle relation symmetry changed")
    return payload


def attach_sidecar(root, selection, data):
    import torch
    path = _confined(Path(root), selection.manifest)
    if sha256_file(path) != selection.manifest_sha256:
        raise ValueError("Cycle sidecar manifest changed")
    manifest = json.loads(path.read_bytes())
    if (manifest.get("format") != FORMAT or manifest.get("status") != "complete" or
            manifest.get("maximum_cycle") != MAX_CYCLE):
        raise ValueError("Unaccepted cycle sidecar")
    def wrap(dataset, start, rows):
        selected, cursor = [], start
        for record in manifest["chunks"]:
            if not start <= record["source_start"] < start + rows:
                continue
            if (record["source_start"] != cursor or record["source_manifest_sha256"] != data.manifest_sha256 or
                    record["independently_rederived"] is not True):
                raise ValueError("Cycle sidecar selection/order/source changed")
            chunk = _confined(path.parent, record["file"])
            if sha256_file(chunk) != record["sha256"]:
                raise ValueError("Cycle chunk SHA changed")
            payload = torch.load(chunk, map_location="cpu", weights_only=True, mmap=True)
            validate_chunk(payload, record)
            if not torch.equal(payload["source_idx"], torch.arange(cursor, record["source_stop"])):
                raise ValueError("Cycle source rows changed")
            selected.append((cursor, record["source_stop"], payload))
            cursor = record["source_stop"]
        if cursor != start + rows:
            raise ValueError("Incomplete cycle sidecar role")
        return CycleDataset(dataset, selected)
    return replace(data, train=wrap(data.train, data.train_source_start, data.train_rows),
                   development=wrap(data.development, data.development_source_start, data.development_rows)
                   if data.development is not None else None)
