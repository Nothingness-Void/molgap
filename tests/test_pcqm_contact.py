from __future__ import annotations

import torch
from torch_geometric.data import Data

from molgap.pcqm_contact import contact_statistics, with_non_covalent_contacts


def chain_graph(positions: list[list[float]], *, valid: bool = True) -> Data:
    node_count = len(positions)
    directed = []
    for node in range(node_count - 1):
        directed.extend(((node, node + 1), (node + 1, node)))
    return Data(
        x=torch.tensor([[5, 0]] * node_count, dtype=torch.long),
        edge_index=torch.tensor(directed, dtype=torch.long).t().contiguous(),
        pos=torch.tensor(positions, dtype=torch.float32),
        geometry_valid=torch.tensor([1.0 if valid else 0.0]),
        row_index=torch.tensor([0]),
    )


def test_contact_excludes_pairs_within_three_covalent_hops() -> None:
    graph = chain_graph([[float(index), 0.0, 0.0] for index in range(5)])
    result = with_non_covalent_contacts(graph)
    assert result.contact_edge_index.tolist() == [[0, 4], [4, 0]]
    assert result.contact_distance.flatten().tolist() == [4.0, 4.0]


def test_contact_cutoff_and_order_are_deterministic() -> None:
    graph = chain_graph([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [5.1, 0, 0], [4.5, 0, 0]])
    first = with_non_covalent_contacts(graph)
    second = with_non_covalent_contacts(graph)
    assert torch.equal(first.contact_edge_index, second.contact_edge_index)
    assert torch.equal(first.contact_distance, second.contact_distance)
    assert float(first.contact_distance.max()) <= 5.0


def test_invalid_geometry_retains_graph_with_empty_contacts() -> None:
    graph = chain_graph([[0, 0, 0], [1, 0, 0]], valid=False)
    result = with_non_covalent_contacts(graph)
    assert result.contact_edge_index.shape == (2, 0)
    assert result.contact_distance.shape == (0, 1)


def test_contact_statistics_do_not_require_targets() -> None:
    graph = with_non_covalent_contacts(
        chain_graph([[float(index), 0.0, 0.0] for index in range(5)])
    )
    stats = contact_statistics([graph])
    assert stats["graphs"] == 1
    assert stats["undirected_pairs"] == 1
    assert stats["directed_edges"] == 2
    assert stats["atom_type_pairs"] == {"5:5": 1}
