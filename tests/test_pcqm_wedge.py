from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from torch_geometric.data import Batch

from molgap.pcqm_wedge import WedgeData, directed_nonbacktracking_wedges


def test_directed_wedges_exclude_immediate_reversal():
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.long,
    )
    pairs = directed_nonbacktracking_wedges(edge_index)
    assert pairs.tolist() == [[0, 2], [3, 1]]


def test_wedge_edge_ids_increment_by_directed_edge_count_when_batched():
    first = WedgeData(
        x=torch.zeros((3, 1)),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 1)),
        wedge_edge_ids=torch.tensor([[0, 1]], dtype=torch.long),
    )
    second = WedgeData(
        x=torch.zeros((3, 1)),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 1)),
        wedge_edge_ids=torch.tensor([[0, 1]], dtype=torch.long),
    )
    batch = Batch.from_data_list([first, second])
    assert batch.wedge_edge_ids.tolist() == [[0, 1], [2, 3]]
