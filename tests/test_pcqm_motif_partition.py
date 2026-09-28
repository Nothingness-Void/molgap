"""Synthetic topology-only guards for the proposed motif hierarchy."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from molgap.pcqm_motif_partition import derive_motif_partition


def graph(node_count: int, bonds, *, ring=(), hetero=()):
    x = torch.zeros((node_count, 9), dtype=torch.long)
    x[:, 0] = 5  # OGB carbon category.
    x[list(ring), 8] = 1
    x[list(hetero), 0] = 7  # OGB oxygen category.
    pairs = []
    attrs = []
    for u, v, bond_type in bonds:
        pairs.extend(((u, v), (v, u)))
        attrs.extend(((bond_type, 0, 0),) * 2)
    return SimpleNamespace(
        x=x,
        edge_index=torch.tensor(pairs, dtype=torch.long).t().reshape(2, -1),
        edge_attr=torch.tensor(attrs, dtype=torch.long).reshape(-1, 3),
        num_nodes=node_count,
    )


def test_ring_stays_intact_while_three_bridge_rules_partition():
    bonds = [(i, (i + 1) % 6, 3) for i in range(6)]
    bonds += [(0, 6, 0), (6, 7, 0), (6, 8, 1)]
    result = derive_motif_partition(graph(9, bonds, ring=range(6), hetero=(7,)))
    assigned = result["motif_index"].tolist()
    assert result["motif_count"] == 4
    assert len(set(assigned[:6])) == 1
    assert len(set(assigned[6:])) == 3
    assert result["motif_edge_index"].shape == (2, 6)
    assert sorted(set(result["motif_bridge_rule_mask"].tolist())) == [1, 2, 4]


def test_node_relabeling_preserves_partition_equivalence():
    bonds = [(0, 1, 0), (1, 2, 0), (2, 0, 0), (2, 3, 0), (3, 4, 1)]
    original = graph(5, bonds, ring=(0, 1, 2), hetero=(4,))
    permutation = (3, 0, 4, 1, 2)  # old index -> new index
    remapped = graph(
        5, [(permutation[u], permutation[v], kind) for u, v, kind in bonds],
        ring=tuple(permutation[i] for i in (0, 1, 2)),
        hetero=(permutation[4],),
    )
    before = derive_motif_partition(original)["motif_index"].tolist()
    after = derive_motif_partition(remapped)["motif_index"].tolist()
    assert all(
        (before[i] == before[j]) == (after[permutation[i]] == after[permutation[j]])
        for i in range(5) for j in range(5)
    )


def test_isolated_and_disconnected_atoms_are_supported():
    result = derive_motif_partition(graph(3, []))
    assert result["motif_count"] == 3
    assert result["motif_edge_index"].shape == (2, 0)


def test_mismatched_reverse_edge_fails_closed():
    input_graph = graph(2, [(0, 1, 0)])
    input_graph.edge_attr[1, 0] = 1
    try:
        derive_motif_partition(input_graph)
    except ValueError as exc:
        assert "bidirectional" in str(exc)
    else:
        raise AssertionError("Asymmetric real bonds must fail")
