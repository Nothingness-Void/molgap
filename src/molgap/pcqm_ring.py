"""Deterministic smallest-ring hierarchy for the PCQM Gap screen.

The representation is derived only from the selected molecule's canonical
SMILES.  Ring ids are local to a graph and are made batch-safe by
``RingHierarchyData``.  No target or prediction enters cache construction.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from .pcqm_wedge import WedgeData


RING_FEATURE_CHANNELS = 12
RING_EDGE_FEATURE_CHANNELS = 4


class RingHierarchyData(WedgeData):
    """Geometry/wedge graph with independently indexed ring tensors."""

    @property
    def num_rings(self) -> int:
        features = getattr(self, "ring_features", None)
        return int(features.shape[0]) if features is not None else 0

    def __inc__(self, key, value, *args, **kwargs):
        if key == "atom_ring_index":
            return torch.tensor(
                [[int(self.num_nodes)], [self.num_rings]], dtype=torch.long
            )
        if key == "ring_edge_index":
            return self.num_rings
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {"atom_ring_index", "ring_edge_index"}:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def _canonical_smallest_rings(molecule) -> list[tuple[int, ...]]:
    """Return deterministic symmetrized smallest-ring atom tuples."""
    from rdkit import Chem

    rings = {
        tuple(sorted(int(atom) for atom in ring))
        for ring in Chem.GetSymmSSSR(molecule)
    }
    if any(len(ring) < 3 for ring in rings):
        raise ValueError("RDKit returned an invalid ring smaller than three atoms")
    return sorted(rings, key=lambda ring: (len(ring), ring))


def _validate_graph_alignment(graph, molecule) -> None:
    from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

    node_count = int(graph.num_nodes)
    if molecule.GetNumAtoms() != node_count:
        raise ValueError(
            "SMILES atom count does not match the accepted parent graph: "
            f"{molecule.GetNumAtoms()} != {node_count}"
        )
    edge_index = graph.edge_index.long()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("parent edge_index must have shape [2, E]")
    expected = {
        (int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))
        for bond in molecule.GetBonds()
    }
    expected |= {(target, source) for source, target in tuple(expected)}
    actual = {
        (int(source), int(target))
        for source, target in edge_index.t().tolist()
    }
    if actual != expected:
        raise ValueError("SMILES bonds do not match the accepted parent graph")
    expected_nodes = torch.tensor(
        [atom_to_feature_vector(atom) for atom in molecule.GetAtoms()],
        dtype=torch.long,
    )
    if not torch.equal(graph.x.detach().cpu().long(), expected_nodes):
        raise ValueError("SMILES atom features do not match the accepted parent graph")
    expected_edge_features = {}
    for bond in molecule.GetBonds():
        features = tuple(int(value) for value in bond_to_feature_vector(bond))
        source = int(bond.GetBeginAtomIdx())
        target = int(bond.GetEndAtomIdx())
        expected_edge_features[(source, target)] = features
        expected_edge_features[(target, source)] = features
    actual_edge_features = {
        (int(source), int(target)): tuple(int(value) for value in features)
        for (source, target), features in zip(
            edge_index.t().tolist(), graph.edge_attr.detach().cpu().tolist()
        )
    }
    if actual_edge_features != expected_edge_features:
        raise ValueError("SMILES bond features do not match the accepted parent graph")


def _ring_relations(molecule, rings: list[tuple[int, ...]]):
    ring_sets = [set(ring) for ring in rings]
    relations: list[tuple[int, int, tuple[float, float, float, float]]] = []
    degrees = [0] * len(rings)
    spiro = [False] * len(rings)
    fused = [False] * len(rings)
    for left in range(len(rings)):
        for right in range(left + 1, len(rings)):
            shared = ring_sets[left].intersection(ring_sets[right])
            edge_features = None
            if shared:
                is_spiro = len(shared) == 1
                is_fused = len(shared) >= 2
                edge_features = (
                    float(is_spiro),
                    float(is_fused),
                    0.0,
                    0.0,
                )
                spiro[left] |= is_spiro
                spiro[right] |= is_spiro
                fused[left] |= is_fused
                fused[right] |= is_fused
            else:
                connecting = []
                for source in rings[left]:
                    for target in rings[right]:
                        bond = molecule.GetBondBetweenAtoms(source, target)
                        if bond is not None:
                            connecting.append(bond)
                if connecting:
                    edge_features = (
                        0.0,
                        0.0,
                        1.0,
                        float(any(bond.GetIsConjugated() for bond in connecting)),
                    )
            if edge_features is not None:
                relations.append((left, right, edge_features))
                relations.append((right, left, edge_features))
                degrees[left] += 1
                degrees[right] += 1
    return relations, degrees, spiro, fused


def _ring_feature_rows(
    molecule,
    rings: list[tuple[int, ...]],
    degrees: list[int],
    spiro: list[bool],
    fused: list[bool],
) -> list[list[float]]:
    membership_count = [0] * molecule.GetNumAtoms()
    for ring in rings:
        for atom in ring:
            membership_count[atom] += 1

    rows: list[list[float]] = []
    for ring_id, ring in enumerate(rings):
        ring_atoms = [molecule.GetAtomWithIdx(atom) for atom in ring]
        ring_set = set(ring)
        internal_bonds = [
            bond
            for bond in molecule.GetBonds()
            if bond.GetBeginAtomIdx() in ring_set
            and bond.GetEndAtomIdx() in ring_set
        ]
        size = float(len(ring))
        element_counts = [
            sum(atom.GetAtomicNum() == atomic_number for atom in ring_atoms)
            for atomic_number in (6, 7, 8, 16)
        ]
        other_count = len(ring) - sum(element_counts)
        composition = [count / size for count in (*element_counts, other_count)]
        aromatic_fraction = sum(atom.GetIsAromatic() for atom in ring_atoms) / size
        conjugated_fraction = (
            sum(bond.GetIsConjugated() for bond in internal_bonds)
            / max(len(internal_bonds), 1)
        )
        shared_fraction = (
            sum(membership_count[atom] > 1 for atom in ring) / size
        )
        rows.append(
            [
                min(size, 12.0) / 12.0,
                *composition,
                aromatic_fraction,
                conjugated_fraction,
                shared_fraction,
                float(spiro[ring_id]),
                float(fused[ring_id]),
                min(float(degrees[ring_id]), 8.0) / 8.0,
            ]
        )
    return rows


def with_ring_hierarchy(graph, smiles: str) -> RingHierarchyData:
    """Copy one accepted geometry graph and attach its ring hierarchy."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("RDKit could not parse the selected canonical SMILES")
    _validate_graph_alignment(graph, molecule)
    rings = _canonical_smallest_rings(molecule)
    relations, degrees, spiro, fused = _ring_relations(molecule, rings)
    feature_rows = _ring_feature_rows(
        molecule, rings, degrees, spiro, fused
    )

    memberships = [
        (atom, ring_id)
        for ring_id, ring in enumerate(rings)
        for atom in ring
    ]
    if memberships:
        atom_ring_index = torch.tensor(memberships, dtype=torch.long).t()
    else:
        atom_ring_index = torch.empty((2, 0), dtype=torch.long)

    if relations:
        ring_edge_index = torch.tensor(
            [(source, target) for source, target, _ in relations],
            dtype=torch.long,
        ).t()
        ring_edge_attr = torch.tensor(
            [features for _, _, features in relations], dtype=torch.float32
        )
    else:
        ring_edge_index = torch.empty((2, 0), dtype=torch.long)
        ring_edge_attr = torch.empty(
            (0, RING_EDGE_FEATURE_CHANNELS), dtype=torch.float32
        )

    if feature_rows:
        ring_features = torch.tensor(feature_rows, dtype=torch.float32)
    else:
        ring_features = torch.empty(
            (0, RING_FEATURE_CHANNELS), dtype=torch.float32
        )
    if ring_features.shape[1] != RING_FEATURE_CHANNELS:
        raise RuntimeError("ring feature width changed")
    if not torch.isfinite(ring_features).all() or not torch.isfinite(
        ring_edge_attr
    ).all():
        raise RuntimeError("ring hierarchy contains non-finite values")

    result = RingHierarchyData(**graph.to_dict())
    result.ring_features = ring_features
    result.atom_ring_index = atom_ring_index
    result.ring_edge_index = ring_edge_index
    result.ring_edge_attr = ring_edge_attr
    return result


def hierarchy_counts(graphs: Iterable[RingHierarchyData]) -> dict[str, int]:
    """Summarize cache cardinalities without model execution."""
    totals = {"rings": 0, "memberships": 0, "directed_relations": 0}
    for graph in graphs:
        totals["rings"] += int(graph.ring_features.shape[0])
        totals["memberships"] += int(graph.atom_ring_index.shape[1])
        totals["directed_relations"] += int(graph.ring_edge_index.shape[1])
    return totals


__all__ = [
    "RING_EDGE_FEATURE_CHANNELS",
    "RING_FEATURE_CHANNELS",
    "RingHierarchyData",
    "hierarchy_counts",
    "with_ring_hierarchy",
]
