import torch
from torch_geometric.data import Data

from molgap.qm9_conformer import _paired_views


def _graph(source_idx):
    return Data(source_idx=torch.tensor([source_idx]))


def test_paired_views_preserve_primary_order_and_drop_failures():
    first = [_graph(5), _graph(2), _graph(9)]
    second = [_graph(9), _graph(5), _graph(7)]

    first_view, second_view = _paired_views(first, second)

    assert [int(graph.source_idx) for graph in first_view] == [5, 9]
    assert [int(graph.source_idx) for graph in second_view] == [5, 9]
