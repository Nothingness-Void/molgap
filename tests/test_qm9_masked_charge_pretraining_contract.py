from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from molgap.qm9_charge_adapter import BATCH_SIZE, make_charge_encoder
from molgap.qm9_masked_charge_pretraining import (
    CONTROL_EPOCHS,
    FINETUNE_EPOCHS,
    MASK_RATE,
    PRETRAIN_EPOCHS,
    MaskedReconstructionHeads,
    _node_mask,
    _paired_edge_mask,
    _pretrain_loss,
)


ROOT = Path(__file__).parents[1]


def _graphs(count=2):
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    return [
        Data(
            x=torch.zeros((4, 9), dtype=torch.long),
            edge_index=edge_index,
            edge_attr=torch.zeros((6, 3), dtype=torch.long),
            y=torch.tensor([0.1]),
            random_walk_pe=torch.zeros((4, 16)),
            gasteiger_features=torch.randn((4, 2)) * 0.1,
            row_id=torch.tensor([index]),
        )
        for index in range(count)
    ]


def test_equal_exposure_and_frozen_batch():
    assert BATCH_SIZE == 128
    assert PRETRAIN_EPOCHS == 20
    assert FINETUNE_EPOCHS == 40
    assert CONTROL_EPOCHS == 60
    assert MASK_RATE == 0.15


def test_undirected_bond_directions_share_one_mask():
    graph = _graphs(1)[0]
    mask = _paired_edge_mask(
        graph.edge_index, 0.5, torch.Generator().manual_seed(42)
    )
    assert mask.shape == (6,)
    assert torch.equal(mask[0::2], mask[1::2])
    assert mask.any()


def test_node_mask_is_nonempty():
    mask = _node_mask(4, 0.0, torch.Generator().manual_seed(42))
    assert mask.sum() == 1


def test_masked_reconstruction_forward_backward_is_finite():
    batch = next(iter(DataLoader(_graphs(), batch_size=2)))
    torch.manual_seed(42)
    model = make_charge_encoder([0.0, 0.0], [1.0, 1.0])
    heads = MaskedReconstructionHeads(model)
    loss, pieces = _pretrain_loss(
        model,
        heads,
        batch,
        torch.zeros((1, 2)),
        torch.ones((1, 2)),
        torch.Generator().manual_seed(71),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in pieces.values())
    assert any(parameter.grad is not None for parameter in model.parameters())
    heads.close()


def test_remote_runtime_is_complete_and_resumable():
    package = (
        ROOT
        / "platforms/scnet/qm9_masked_charge_pretraining/package_runtime.ps1"
    ).read_text()
    source = (
        ROOT / "src/molgap/qm9_masked_charge_pretraining.py"
    ).read_text()
    assert "src/molgap/qm9_charge_adapter.py" in package
    assert "src/molgap/qm9_masked_charge_pretraining.py" in package
    assert "last_checkpoint.pt" in source
    assert '"mask_rng"' in source
    assert "CONTROL_EPOCHS" in source
