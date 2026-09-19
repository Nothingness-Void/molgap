"""Deterministic functional-group sidecar for the fixed PCQM V4 screen.

The accepted graph dataset remains immutable.  This module derives one binary
atom-by-group incidence matrix from each graph's embedded sanitized SMILES and
stores the matrices in row-aligned, independently hashed shards.  It never
reads Gap labels or protected PCQM roles.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file


FORMAT = "molgap-pcqm-fixed100k-functional-group-sidecar-v1"
ACCEPTANCE_FORMAT = "molgap-pcqm-fixed100k-functional-group-sidecar-acceptance-v1"
FUNCTIONAL_GROUP_RULES = {
    "aromatic": "atom.is_aromatic",
    "carbonyl": "C=O",
    "amide": "N-C(=O)",
    "ester": "C(=O)-O-C",
    "carboxyl": "C(=O)-O(H)",
    "nitrile": "C#N",
    "amine": "N and not N-C(=O)",
    "alcohol": "O(H)-C",
    "ether": "C-O-C",
    "alkene": "C=C",
    "alkyne": "C#C",
    "ring_heteroatom": "ring atom in {N,O,S}",
}
GROUP_NAMES = tuple(FUNCTIONAL_GROUP_RULES)


def functional_group_contract_sha256() -> str:
    payload = json.dumps(
        FUNCTIONAL_GROUP_RULES, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def functional_group_labels_from_graph(graph):
    """Derive chemistry roles only from the accepted OGB x/edge tensors."""
    import torch

    x = graph.x.long()
    edge_index = graph.edge_index.long()
    edge_attr = graph.edge_attr.long()
    labels = torch.zeros((int(graph.num_nodes), len(GROUP_NAMES)), dtype=torch.uint8)
    atomic_number = x[:, 0] + 1
    total_h = x[:, 4]
    aromatic = x[:, 7] == 1
    in_ring = x[:, 8] == 1
    labels[:, 0] = aromatic.to(torch.uint8)
    labels[:, 11] = (
        in_ring
        & ((atomic_number == 7) | (atomic_number == 8) | (atomic_number == 16))
    ).to(torch.uint8)

    neighbors: list[dict[int, int]] = [dict() for _ in range(int(graph.num_nodes))]
    for edge_id in range(int(edge_index.shape[1])):
        source = int(edge_index[0, edge_id])
        target = int(edge_index[1, edge_id])
        neighbors[source][target] = int(edge_attr[edge_id, 0])

    def mark(column: int, atoms) -> None:
        labels[list(dict.fromkeys(int(atom) for atom in atoms)), column] = 1

    carbonyl_oxygens: dict[int, list[int]] = {}
    for carbon in range(int(graph.num_nodes)):
        if int(atomic_number[carbon]) != 6:
            continue
        oxygens = [
            neighbor
            for neighbor, bond_type in neighbors[carbon].items()
            if bond_type == 1 and int(atomic_number[neighbor]) == 8
        ]
        if oxygens:
            carbonyl_oxygens[carbon] = oxygens
            mark(1, [carbon, *oxygens])

    amide_nitrogens: set[int] = set()
    for carbon, oxygens in carbonyl_oxygens.items():
        single_hetero = [
            neighbor
            for neighbor, bond_type in neighbors[carbon].items()
            if bond_type == 0 and neighbor not in oxygens
        ]
        for neighbor in single_hetero:
            element = int(atomic_number[neighbor])
            if element == 7:
                amide_nitrogens.add(neighbor)
                mark(2, [neighbor, carbon, *oxygens])
            if element != 8:
                continue
            carbon_neighbors = [
                atom
                for atom, bond_type in neighbors[neighbor].items()
                if atom != carbon and bond_type == 0 and int(atomic_number[atom]) == 6
            ]
            if carbon_neighbors:
                mark(3, [carbon, *oxygens, neighbor, *carbon_neighbors])
            if int(total_h[neighbor]) > 0:
                mark(4, [carbon, *oxygens, neighbor])

    for source in range(int(graph.num_nodes)):
        for target, bond_type in neighbors[source].items():
            if source >= target:
                continue
            elements = {int(atomic_number[source]), int(atomic_number[target])}
            if bond_type == 2 and elements == {6, 7}:
                mark(5, [source, target])
            if bond_type == 1 and elements == {6}:
                mark(9, [source, target])
            if bond_type == 2 and elements == {6}:
                mark(10, [source, target])

    for atom in range(int(graph.num_nodes)):
        element = int(atomic_number[atom])
        if element == 7 and atom not in amide_nitrogens:
            mark(6, [atom])
        if element != 8:
            continue
        carbon_neighbors = [
            neighbor
            for neighbor, bond_type in neighbors[atom].items()
            if bond_type == 0 and int(atomic_number[neighbor]) == 6
        ]
        if int(total_h[atom]) > 0 and carbon_neighbors:
            mark(7, [atom, *carbon_neighbors])
        if len(carbon_neighbors) >= 2:
            mark(8, [atom, *carbon_neighbors])
    return labels


def _aggregate_sha256(shards: list[dict]) -> str:
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(
            (
                f"{shard['role']}\t{shard['parent_file']}\t{shard['file']}\t"
                f"{shard['sha256']}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def _packed_dataset(path: Path):
    from .pcqm_k1_variants_runner import _PackedGraphDatasetFactory

    return _PackedGraphDatasetFactory.load(path)


def build_functional_group_sidecar(
    fixed_root: Path,
    output_root: Path,
    *,
    source_commit: str,
) -> dict:
    """Build an atomic sidecar from only the accepted fixed-cache graph rows."""
    import torch

    from .pcqm_k1_variants_runner import (
        DEVELOPMENT_ROWS,
        FIXED_GEOMETRY_SHA256,
        FIXED_MANIFEST_SHA256,
        TRAIN_ROWS,
        find_fixed_cache,
    )

    if len(source_commit) != 40:
        raise ValueError("A full source commit is required")
    os.environ["MOLGAP_FIXED_CACHE_ROOT"] = str(fixed_root)
    accepted_root, fixed_manifest = find_fixed_cache()
    if accepted_root.resolve() != fixed_root.resolve():
        raise RuntimeError("Fixed-cache root resolution changed")

    output_root.mkdir(parents=True, exist_ok=True)
    final_path = output_root / "manifest.json"
    if final_path.is_file():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if existing.get("complete") is True:
            return existing
        raise RuntimeError("Incomplete final sidecar manifest exists")

    shards: list[dict] = []
    role_counts = {"train": 0, "development": 0}
    positive_counts = torch.zeros(len(GROUP_NAMES), dtype=torch.long)
    covered_atoms = 0
    total_atoms = 0
    for item in fixed_manifest["geometry_shards"]:
        parent_path = fixed_root / item["file"]
        if sha256_file(parent_path) != item["sha256"]:
            raise RuntimeError(f"Fixed shard changed: {item['file']}")
        dataset = _packed_dataset(parent_path)
        labels: list[torch.Tensor] = []
        source_indices: list[int] = []
        for graph in dataset:
            source_idx = int(graph.source_idx.view(-1)[0])
            incidence = functional_group_labels_from_graph(graph)
            if incidence.shape != (int(graph.num_nodes), len(GROUP_NAMES)):
                raise RuntimeError(f"Atom/group alignment changed at {source_idx}")
            labels.append(incidence)
            source_indices.append(source_idx)
            positive_counts += incidence.sum(dim=0).long()
            covered_atoms += int(incidence.any(dim=1).sum())
            total_atoms += int(incidence.shape[0])

        expected = list(range(int(item["source_idx_min"]), int(item["source_idx_max"]) + 1))
        if source_indices != expected:
            raise RuntimeError(f"Source order changed in {item['file']}")
        output_name = Path(item["file"]).stem + "-functional-groups.pt"
        output_path = output_root / output_name
        atomic_torch_save(
            output_path,
            {
                "source_idx": torch.tensor(source_indices, dtype=torch.long),
                "functional_group_y": labels,
            },
        )
        record = {
            "role": item["role"],
            "file": output_name,
            "parent_file": item["file"],
            "parent_sha256": item["sha256"],
            "source_idx_min": source_indices[0],
            "source_idx_max": source_indices[-1],
            "rows": len(labels),
            "sha256": sha256_file(output_path),
        }
        shards.append(record)
        role_counts[item["role"]] += len(labels)
        atomic_json(
            output_root / "progress.json",
            {
                "format": FORMAT,
                "complete": False,
                "source_commit": source_commit,
                "completed_shards": shards,
                "role_counts": role_counts,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
                "test_challenge_role_read": False,
                "model_inference_executed": False,
            },
        )

    if role_counts != {"train": TRAIN_ROWS, "development": DEVELOPMENT_ROWS}:
        raise RuntimeError(f"Sidecar role counts are incomplete: {role_counts}")
    manifest = {
        "format": FORMAT,
        "complete": True,
        "source_commit": source_commit,
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "fixed_geometry_aggregate_sha256": FIXED_GEOMETRY_SHA256,
        "functional_group_names": list(GROUP_NAMES),
        "functional_group_rules": FUNCTIONAL_GROUP_RULES,
        "functional_group_contract_sha256": functional_group_contract_sha256(),
        "functional_group_positive_counts": positive_counts.tolist(),
        "covered_atoms": covered_atoms,
        "total_atoms": total_atoms,
        "covered_atom_fraction": covered_atoms / total_atoms,
        "train_graphs": role_counts["train"],
        "development_graphs": role_counts["development"],
        "shards": shards,
        "aggregate_sha256": _aggregate_sha256(shards),
        "gpu_used": False,
        "gap_labels_read": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(final_path, manifest)
    return manifest


def accept_functional_group_sidecar(
    root: Path,
    *,
    expected_source_commit: str | None = None,
) -> dict:
    """Validate sidecar identity and all row-aligned shard contents."""
    import torch

    from .pcqm_k1_variants_runner import (
        DEVELOPMENT_ROWS,
        FIXED_GEOMETRY_SHA256,
        FIXED_MANIFEST_SHA256,
        TRAIN_ROWS,
    )

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checks = {
        "format": manifest.get("format") == FORMAT,
        "complete": manifest.get("complete") is True,
        "source_commit": expected_source_commit is None
        or manifest.get("source_commit") == expected_source_commit,
        "fixed_manifest": manifest.get("fixed_manifest_sha256")
        == FIXED_MANIFEST_SHA256,
        "fixed_geometry": manifest.get("fixed_geometry_aggregate_sha256")
        == FIXED_GEOMETRY_SHA256,
        "smarts_contract": manifest.get("functional_group_contract_sha256")
        == functional_group_contract_sha256(),
        "group_names": manifest.get("functional_group_names") == list(GROUP_NAMES),
        "train_graphs": manifest.get("train_graphs") == TRAIN_ROWS,
        "development_graphs": manifest.get("development_graphs")
        == DEVELOPMENT_ROWS,
        "gpu_unused": manifest.get("gpu_used") is False,
        "labels_sealed": manifest.get("gap_labels_read") is False,
        "no_model_inference": manifest.get("model_inference_executed") is False,
        "protected_roles_sealed": all(
            manifest.get(name) is False
            for name in (
                "official_validation_role_read",
                "test_dev_role_read",
                "test_challenge_role_read",
            )
        ),
    }
    observed = {"train": 0, "development": 0}
    shard_checks: list[bool] = []
    for item in manifest.get("shards", []):
        path = root / item["file"]
        valid = path.is_file() and sha256_file(path) == item.get("sha256")
        payload = torch.load(path, map_location="cpu", weights_only=False) if valid else {}
        indices = payload.get("source_idx")
        labels = payload.get("functional_group_y")
        valid = valid and isinstance(indices, torch.Tensor) and isinstance(labels, list)
        if valid:
            expected = torch.arange(
                int(item["source_idx_min"]),
                int(item["source_idx_max"]) + 1,
                dtype=torch.long,
            )
            valid = (
                torch.equal(indices, expected)
                and len(labels) == int(item["rows"])
                and all(
                    isinstance(value, torch.Tensor)
                    and value.ndim == 2
                    and value.shape[1] == len(GROUP_NAMES)
                    and value.dtype == torch.uint8
                    and bool(((value == 0) | (value == 1)).all())
                    for value in labels
                )
            )
        shard_checks.append(bool(valid))
        if valid:
            observed[item["role"]] += len(labels)
    checks["all_shards"] = bool(shard_checks) and all(shard_checks)
    checks["role_counts"] = observed == {
        "train": TRAIN_ROWS,
        "development": DEVELOPMENT_ROWS,
    }
    checks["aggregate"] = _aggregate_sha256(manifest.get("shards", [])) == manifest.get(
        "aggregate_sha256"
    )
    acceptance = {
        "format": ACCEPTANCE_FORMAT,
        "accepted": all(checks.values()),
        "checks": checks,
        "source_commit": manifest.get("source_commit"),
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "functional_group_contract_sha256": manifest.get(
            "functional_group_contract_sha256"
        ),
        "train_graphs": observed["train"],
        "development_graphs": observed["development"],
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(root / "acceptance.json", acceptance)
    if not acceptance["accepted"]:
        raise RuntimeError(f"Functional-group sidecar acceptance failed: {checks}")
    return acceptance


class _FunctionalGroupDataset:
    def __init__(self, base, source_idx, labels):
        self.base = base
        self.source_idx = source_idx
        self.labels = labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        graph = self.base[index].clone()
        observed = int(graph.source_idx.view(-1)[0])
        expected = int(self.source_idx[index])
        if observed != expected:
            raise RuntimeError("Functional-group sidecar row alignment changed")
        graph.functional_group_y = self.labels[index].float()
        return graph


def find_functional_group_sidecar() -> tuple[Path, dict]:
    candidates: list[tuple[Path, dict]] = []
    explicit = os.environ.get("MOLGAP_FUNCTIONAL_GROUP_SIDECAR_ROOT")
    roots = [Path(explicit)] if explicit else [Path("/kaggle/input")]
    for search_root in roots:
        paths = (
            [search_root / "manifest.json"]
            if (search_root / "manifest.json").is_file()
            else search_root.rglob("manifest.json")
        )
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("format") == FORMAT and payload.get("complete") is True:
                candidates.append((path.parent, payload))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one functional-group sidecar, found {candidates}")
    return candidates[0]


def attach_functional_group_roles(roles, fixed_manifest: dict):
    """Return row-aligned datasets whose graphs carry binary group incidence."""
    import torch
    from torch.utils.data import ConcatDataset

    root, manifest = find_functional_group_sidecar()
    if manifest.get("fixed_geometry_aggregate_sha256") != fixed_manifest.get(
        "geometry_aggregate_sha256"
    ):
        raise RuntimeError("Functional-group sidecar parent identity changed")
    by_parent = {item["parent_file"]: item for item in manifest["shards"]}
    graph_items = fixed_manifest["geometry_shards"]
    if set(by_parent) != {item["file"] for item in graph_items}:
        raise RuntimeError("Functional-group sidecar shard set changed")
    wrapped = {"train": [], "development": []}
    base_parts = {
        role: iter(roles[role].datasets) for role in ("train", "development")
    }
    for item in graph_items:
        sidecar = by_parent[item["file"]]
        path = root / sidecar["file"]
        if sha256_file(path) != sidecar["sha256"]:
            raise RuntimeError(f"Functional-group sidecar shard changed: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        base = next(base_parts[item["role"]])
        if len(base) != len(payload["functional_group_y"]):
            raise RuntimeError("Functional-group sidecar length changed")
        wrapped[item["role"]].append(
            _FunctionalGroupDataset(
                base,
                payload["source_idx"],
                payload["functional_group_y"],
            )
        )
    return {role: ConcatDataset(parts) for role, parts in wrapped.items()}, manifest
