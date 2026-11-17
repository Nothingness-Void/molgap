from __future__ import annotations

import torch

from molgap.portable_radius import radius_graph as portable_radius_graph


def _edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(map(tuple, edge_index.t().tolist()))


def test_portable_radius_matches_torch_cluster_without_neighbor_truncation():
    from torch_cluster import radius_graph

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    batch = torch.tensor([0, 0, 0, 1, 1])
    expected = radius_graph(
        positions, r=1.5, batch=batch, max_num_neighbors=32
    )
    actual = portable_radius_graph(
        positions, r=1.5, batch=batch, max_num_neighbors=32
    )
    assert _edge_set(actual) == _edge_set(expected)


def test_portable_radius_limits_each_target_to_nearest_neighbors():
    positions = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    )
    edge_index = portable_radius_graph(
        positions, r=10.0, max_num_neighbors=2
    )
    _, counts = torch.unique(edge_index[1], return_counts=True)
    assert counts.max().item() == 2
    assert (1, 0) in _edge_set(edge_index)

