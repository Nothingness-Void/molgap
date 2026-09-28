"""Accepted-graph-only, CPU-built 2/3-hop path sidecar for MetaGIN2D."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file


FORMAT = "molgap-pcqm-fixed100k-metagin-hop-sidecar-v1"
ACCEPTANCE_FORMAT = "molgap-pcqm-fixed100k-metagin-hop-acceptance-v1"


def derive_hops(edge_index, node_count: int):
    """Count directed simple paths of length two and three, without labels.

    The underlying graph must be an undirected molecular bond graph represented
    by paired directed edges. Path multiplicities follow the paper's 2D
    angle/torsion-topology notion; no geometric angle or coordinate is used.
    """
    import torch

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Real-bond edge_index must have shape [2, E]")
    if node_count < 1:
        raise ValueError("A molecule must contain at least one atom")
    edges = [(int(u), int(v)) for u, v in edge_index.t().tolist()]
    edge_set = set(edges)
    if (
        len(edge_set) != len(edges)
        or any(u == v or u < 0 or v < 0 or u >= node_count or v >= node_count for u, v in edges)
        or any((v, u) not in edge_set for u, v in edges)
    ):
        raise ValueError("Expected unique, bidirectional, loop-free real bonds")
    neighbors = [set() for _ in range(node_count)]
    for u, v in edges:
        neighbors[u].add(v)

    second: Counter[tuple[int, int]] = Counter()
    third: Counter[tuple[int, int]] = Counter()
    for source in range(node_count):
        for middle in neighbors[source]:
            for target in neighbors[middle]:
                if target != source:
                    second[source, target] += 1
                for destination in neighbors[target]:
                    if len({source, middle, target, destination}) == 4:
                        third[source, destination] += 1

    def tensors(paths):
        pairs = sorted(paths)
        indices = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if not pairs:
            indices = torch.empty((2, 0), dtype=torch.long)
        counts = torch.tensor([paths[pair] for pair in pairs], dtype=torch.long)
        return indices, counts

    hop2_index, hop2_count = tensors(second)
    hop3_index, hop3_count = tensors(third)
    return {
        "hop2_edge_index": hop2_index,
        "hop2_count": hop2_count,
        "hop3_edge_index": hop3_index,
        "hop3_count": hop3_count,
    }


def _aggregate(shards: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in shards:
        digest.update(
            f"{item['role']}\t{item['parent_file']}\t{item['file']}\t{item['sha256']}\n".encode()
        )
    return digest.hexdigest()


def build_sidecar(output: Path, *, source_commit: str) -> dict:
    """Build independently hashed chunks from the immutable fixed graph cache."""
    import torch

    from .pcqm_k1_variants_runner import (
        DEVELOPMENT_ROWS, FIXED_GEOMETRY_SHA256, FIXED_MANIFEST_SHA256,
        TRAIN_ROWS, _PackedGraphDatasetFactory, find_fixed_cache,
    )

    if len(source_commit) != 40:
        raise ValueError("A full executable-source commit is required")
    fixed_root, fixed = find_fixed_cache()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise RuntimeError("Existing final sidecar cannot be overwritten")
    shards = []
    counts = {"train": 0, "development": 0}
    next_source_idx = {"train": 0, "development": TRAIN_ROWS}
    for item in fixed["geometry_shards"]:
        role = item["role"]
        if role not in counts or sha256_file(fixed_root / item["file"]) != item["sha256"]:
            raise RuntimeError("Fixed parent shard identity changed")
        dataset = _PackedGraphDatasetFactory.load(fixed_root / item["file"])
        rows = []
        for graph in dataset:
            source_idx = int(graph.source_idx.view(-1)[0])
            if source_idx != next_source_idx[role]:
                raise RuntimeError("Fixed graph row order changed")
            topology = derive_hops(graph.edge_index, int(graph.num_nodes))
            rows.append({"source_idx": source_idx, "node_count": int(graph.num_nodes), **topology})
            next_source_idx[role] += 1
            counts[role] += 1
        if len(rows) != item["rows"]:
            raise RuntimeError("Fixed parent shard row count changed")
        name = Path(item["file"]).stem + "-metagin-hops.pt"
        path = output / name
        atomic_torch_save(path, rows)
        shards.append({
            "role": role, "parent_file": item["file"],
            "parent_sha256": item["sha256"], "file": name,
            "sha256": sha256_file(path), "rows": len(rows),
            "source_idx_min": rows[0]["source_idx"],
            "source_idx_max": rows[-1]["source_idx"],
        })
        atomic_json(output / "progress.json", {
            "format": FORMAT, "complete": False, "source_commit": source_commit,
            "completed_shards": shards, "role_counts": counts,
            "official_validation_role_read": False, "test_dev_role_read": False,
            "test_challenge_role_read": False,
        })
        print(f"metagin sidecar {role} {counts[role]}", flush=True)
    if counts != {"train": TRAIN_ROWS, "development": DEVELOPMENT_ROWS}:
        raise RuntimeError(f"Incomplete role coverage: {counts}")
    manifest = {
        "format": FORMAT, "complete": True, "source_commit": source_commit,
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "fixed_geometry_sha256": FIXED_GEOMETRY_SHA256,
        "role_counts": counts, "shards": shards, "aggregate_sha256": _aggregate(shards),
        "derived_from_real_bond_topology_only": True,
        "gap_labels_read": False, "model_inference_executed": False, "gpu_used": False,
        "official_validation_role_read": False, "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def accept_sidecar(root: Path, *, fixed_root: Path, expected_source_commit: str) -> dict:
    """CPU-only full recomputation against the accepted graph payload."""
    import torch

    from .pcqm_k1_variants_runner import (
        DEVELOPMENT_ROWS, FIXED_GEOMETRY_SHA256, FIXED_MANIFEST_SHA256,
        TRAIN_ROWS, _PackedGraphDatasetFactory, find_fixed_cache,
    )

    import os
    previous = os.environ.get("MOLGAP_FIXED_CACHE_ROOT")
    try:
        os.environ["MOLGAP_FIXED_CACHE_ROOT"] = str(fixed_root)
        accepted_root, fixed = find_fixed_cache()
    finally:
        if previous is None:
            os.environ.pop("MOLGAP_FIXED_CACHE_ROOT", None)
        else:
            os.environ["MOLGAP_FIXED_CACHE_ROOT"] = previous
    if accepted_root.resolve() != fixed_root.resolve():
        raise RuntimeError("Fixed graph root changed")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("format") != FORMAT or manifest.get("complete") is not True
        or manifest.get("source_commit") != expected_source_commit
        or manifest.get("fixed_manifest_sha256") != FIXED_MANIFEST_SHA256
        or manifest.get("fixed_geometry_sha256") != FIXED_GEOMETRY_SHA256
        or manifest.get("role_counts") != {"train": TRAIN_ROWS, "development": DEVELOPMENT_ROWS}
        or manifest.get("derived_from_real_bond_topology_only") is not True
        or any(manifest.get(key) is not False for key in (
            "gap_labels_read", "model_inference_executed", "gpu_used",
            "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read",
        ))
        or len(manifest.get("shards", [])) != len(fixed["geometry_shards"])
    ):
        raise RuntimeError("MetaGIN sidecar identity or role seal invalid")
    covered = {"train": 0, "development": TRAIN_ROWS}
    for parent, item in zip(fixed["geometry_shards"], manifest["shards"], strict=True):
        path = (root / item["file"]).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or item["parent_file"] != parent["file"]
            or item["parent_sha256"] != parent["sha256"]
            or item["role"] != parent["role"]
            or item["rows"] != parent["rows"]
            or item["source_idx_min"] != covered[parent["role"]]
            or sha256_file(path) != item["sha256"]
            or sha256_file(fixed_root / parent["file"]) != parent["sha256"]
        ):
            raise RuntimeError("MetaGIN sidecar shard identity invalid")
        rows = torch.load(path, map_location="cpu", weights_only=False)
        graphs = _PackedGraphDatasetFactory.load(fixed_root / parent["file"])
        if len(rows) != len(graphs) or len(rows) != item["rows"]:
            raise RuntimeError("MetaGIN sidecar shard row count invalid")
        for row, graph in zip(rows, graphs, strict=True):
            source_idx = int(graph.source_idx.view(-1)[0])
            expected = derive_hops(graph.edge_index, int(graph.num_nodes))
            if (
                row["source_idx"] != covered[parent["role"]]
                or row["node_count"] != int(graph.num_nodes)
                or source_idx != row["source_idx"]
                or any(not torch.equal(row[key], value) for key, value in expected.items())
            ):
                raise RuntimeError(f"MetaGIN hop derivation changed at {source_idx}")
            covered[parent["role"]] += 1
        if item["source_idx_max"] != covered[parent["role"]] - 1:
            raise RuntimeError("MetaGIN sidecar shard endpoint invalid")
    if covered != {"train": TRAIN_ROWS, "development": TRAIN_ROWS + DEVELOPMENT_ROWS}:
        raise RuntimeError("MetaGIN sidecar coverage invalid")
    if _aggregate(manifest["shards"]) != manifest["aggregate_sha256"]:
        raise RuntimeError("MetaGIN sidecar aggregate invalid")
    acceptance = {
        "format": ACCEPTANCE_FORMAT, "accepted": True,
        "source_commit": expected_source_commit,
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "aggregate_sha256": manifest["aggregate_sha256"],
        "rows_recomputed": TRAIN_ROWS + DEVELOPMENT_ROWS,
        "model_inference_executed": False, "gap_labels_read": False,
        "official_validation_role_read": False, "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(root / "acceptance.json", acceptance)
    return acceptance


def attach_accepted_sidecar(roles, root: Path, *, expected_source_commit: str):
    """Add accepted row-aligned hops without altering the immutable base cache."""
    import torch

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    acceptance = json.loads((root / "acceptance.json").read_text(encoding="utf-8"))
    if (
        manifest.get("format") != FORMAT or manifest.get("complete") is not True
        or manifest.get("source_commit") != expected_source_commit
        or manifest.get("fixed_manifest_sha256") != "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
        or manifest.get("fixed_geometry_sha256") != "bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5"
        or manifest.get("role_counts") != {"train": 100_000, "development": 50_000}
        or acceptance.get("format") != ACCEPTANCE_FORMAT
        or acceptance.get("accepted") is not True
        or acceptance.get("source_commit") != expected_source_commit
        or acceptance.get("fixed_manifest_sha256") != manifest["fixed_manifest_sha256"]
        or acceptance.get("rows_recomputed") != 150_000
        or acceptance.get("manifest_sha256") != sha256_file(root / "manifest.json")
        or acceptance.get("aggregate_sha256") != manifest.get("aggregate_sha256")
        or any(acceptance.get(key) is not False for key in (
            "gap_labels_read", "model_inference_executed",
            "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read",
        ))
    ):
        raise RuntimeError("MetaGIN sidecar lacks independent CPU acceptance")
    all_rows = {"train": [], "development": []}
    for item in manifest["shards"]:
        role = item["role"]
        path = (root / item["file"]).resolve()
        if (
            role not in all_rows or not path.is_relative_to(root.resolve())
            or sha256_file(path) != item["sha256"]
            or item["source_idx_min"] != len(all_rows[role]) + (100_000 if role == "development" else 0)
        ):
            raise RuntimeError("MetaGIN sidecar shard binding invalid")
        rows = torch.load(path, map_location="cpu", weights_only=False)
        if len(rows) != item["rows"]:
            raise RuntimeError("MetaGIN sidecar shard length invalid")
        all_rows[role].extend(rows)
    if _aggregate(manifest["shards"]) != manifest["aggregate_sha256"]:
        raise RuntimeError("MetaGIN sidecar aggregate changed")

    class AttachedRole(torch.utils.data.Dataset):
        def __init__(self, base, rows, offset):
            self.base, self.rows, self.offset = base, rows, offset
            if len(base) != len(rows):
                raise RuntimeError("MetaGIN sidecar role size changed")

        def __len__(self):
            return len(self.base)

        def __getitem__(self, index):
            graph = self.base[index]
            row = self.rows[index]
            if (
                int(graph.source_idx.view(-1)[0]) != index + self.offset
                or row["source_idx"] != index + self.offset
                or row["node_count"] != int(graph.num_nodes)
            ):
                raise RuntimeError("MetaGIN sidecar graph row alignment changed")
            for key in ("hop2_edge_index", "hop2_count", "hop3_edge_index", "hop3_count"):
                setattr(graph, key, row[key])
            return graph

    return {
        role: AttachedRole(roles[role], all_rows[role], offset)
        for role, offset in (("train", 0), ("development", 100_000))
    }, manifest, acceptance
