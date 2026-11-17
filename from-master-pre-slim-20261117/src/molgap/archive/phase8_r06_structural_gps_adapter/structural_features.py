"""Deterministic ring and conjugation features for optional 2D graph adaptors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...utils import require_rdkit


STRUCTURAL_ATOM_FEATURE_NAMES = (
    "atom_in_ring",
    "smallest_ring_size_norm",
    "ring_membership_count_norm",
    "fused_ring_membership_count_norm",
)
STRUCTURAL_EDGE_FEATURE_NAMES = (
    "bond_in_ring",
    "bond_is_conjugated",
)


@dataclass(frozen=True)
class StructuralTopology:
    """Topology features aligned to RDKit atom and directed-bond order."""

    atom_features: np.ndarray
    edge_features: np.ndarray


def structural_topology_from_mol(mol) -> StructuralTopology:
    """Return bounded topology features without molecule-global categorical IDs.

    Ring identifiers are intentionally excluded: their numeric values are local to
    an RDKit molecule and would be meaningless as a learned category across rows.
    """
    require_rdkit()
    if mol is None:
        raise ValueError("mol must be a valid RDKit molecule")

    atom_rings = mol.GetRingInfo().AtomRings()
    rings_for_atom: list[list[int]] = [[] for _ in range(mol.GetNumAtoms())]
    for ring in atom_rings:
        size = len(ring)
        for atom_idx in ring:
            rings_for_atom[atom_idx].append(size)

    atom_rows = []
    for ring_sizes in rings_for_atom:
        membership = len(ring_sizes)
        smallest = min(ring_sizes) if ring_sizes else 0
        fused_membership = membership if membership > 1 else 0
        atom_rows.append([
            float(membership > 0),
            min(float(smallest), 8.0) / 8.0,
            min(float(membership), 4.0) / 4.0,
            min(float(fused_membership), 4.0) / 4.0,
        ])

    edge_rows = []
    for bond in mol.GetBonds():
        row = [float(bond.IsInRing()), float(bond.GetIsConjugated())]
        edge_rows.extend([row, row])

    return StructuralTopology(
        atom_features=np.asarray(atom_rows, dtype=np.float32),
        edge_features=np.asarray(edge_rows, dtype=np.float32).reshape(-1, 2),
    )


def structural_summary_from_mol(mol) -> dict[str, float]:
    """Return molecule-level, auditable summaries used by the G0 audit only."""
    topology = structural_topology_from_mol(mol)
    atom = topology.atom_features
    edge = topology.edge_features
    ring_mask = atom[:, 0] > 0
    smallest_sizes = atom[ring_mask, 1] * 8.0
    memberships = atom[:, 2] * 4.0
    fused = atom[:, 3] > 0

    return {
        "has_ring": float(ring_mask.any()),
        "atom_in_ring_fraction": float(ring_mask.mean()),
        "smallest_ring_size": float(smallest_sizes.min()) if len(smallest_sizes) else 0.0,
        "max_ring_membership_count": float(memberships.max()) if len(memberships) else 0.0,
        "has_fused_ring_atom": float(fused.any()),
        "fused_ring_atom_fraction": float(fused.mean()),
        "ring_bond_fraction": float(edge[:, 0].mean()) if len(edge) else 0.0,
        "has_conjugated_bond": float((edge[:, 1] > 0).any()) if len(edge) else 0.0,
        "conjugated_bond_fraction": float(edge[:, 1].mean()) if len(edge) else 0.0,
    }
