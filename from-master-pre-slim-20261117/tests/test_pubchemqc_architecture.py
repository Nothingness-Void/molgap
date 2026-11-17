import torch
from torch_geometric.data import Data

from molgap.pubchemqc_architecture import align_graph_views, average_views


def graph(index: int, value: float) -> Data:
    return Data(
        z=torch.tensor([6]),
        pos=torch.zeros(1, 3),
        y=torch.tensor([[value, value + 1, value + 2]]),
        source_idx=torch.tensor([index]),
    )


def test_align_graph_views_uses_common_source_indices_and_roles():
    primary = [graph(1, 1.0), graph(2, 2.0), graph(3, 3.0)]
    secondary = [graph(3, 3.0), graph(1, 1.0), graph(2, 2.0)]
    split = {1: "train", 2: "validation", 3: "test", 4: "validation"}

    roles = align_graph_views(primary, secondary, split)

    assert [int(item.source_idx) for item in roles["train"][0]] == [1]
    assert [int(item.source_idx) for item in roles["validation"][0]] == [2]
    assert [int(item.source_idx) for item in roles["test"][0]] == [3]


def test_average_views_preserves_alignment_and_averages_outputs():
    first = {
        "predictions": torch.tensor([[1.0, 2.0, 3.0]]),
        "embeddings": torch.tensor([[2.0, 4.0]]),
        "targets": torch.tensor([[1.5, 2.5, 3.5]]),
        "source_idx": torch.tensor([7]),
    }
    second = {
        "predictions": torch.tensor([[2.0, 3.0, 4.0]]),
        "embeddings": torch.tensor([[4.0, 6.0]]),
        "targets": first["targets"].clone(),
        "source_idx": first["source_idx"].clone(),
    }

    result = average_views(first, second)

    assert torch.equal(result["source_idx"], torch.tensor([7]))
    assert torch.equal(result["predictions"], torch.tensor([[1.5, 2.5, 3.5]]))
    assert torch.equal(result["embeddings"], torch.tensor([[3.0, 5.0]]))
