import numpy as np
import torch
from torch_geometric.data import Data

from molgap.pcqm_expert import (
    EXPECTED_SELECTION_SHA256,
    MODEL_CONFIG,
    PCQMGINEExpert,
    PackedGraphDataset,
    _save_packed_graphs,
    array_sha256,
    expanded_train_indices,
    selected_train_indices,
)


def test_pcqm_250k_selection_contract():
    selected = selected_train_indices()

    assert selected.shape == (250_000,)
    assert np.all(selected[1:] > selected[:-1])
    assert array_sha256(selected) == EXPECTED_SELECTION_SHA256


def test_pcqm_model_matches_accepted_config():
    model = PCQMGINEExpert()

    assert model.hidden_channels == MODEL_CONFIG["hidden_channels"]
    assert len(model.convs) == MODEL_CONFIG["num_layers"]
    assert sum(parameter.numel() for parameter in model.parameters()) > 1_000_000
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_expanded_pcqm_sample_contains_accepted_250k():
    base = selected_train_indices()
    expanded = expanded_train_indices(300_000)

    assert expanded.shape == (300_000,)
    assert np.isin(base, expanded).all()


def test_packed_graph_shard_round_trip(tmp_path):
    graphs = [
        Data(
            x=torch.tensor([[1], [2]], dtype=torch.long),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.tensor([[0], [0]], dtype=torch.long),
            y=torch.tensor([0.5]),
            sample_idx=torch.tensor([17]),
            split_code=torch.tensor([0], dtype=torch.int8),
        )
    ]
    path = tmp_path / "train_shard_000.pt"

    _save_packed_graphs(path, graphs)
    dataset = PackedGraphDataset(path)

    assert len(dataset) == 1
    assert dataset[0].sample_idx.item() == 17
    assert dataset[0].y.item() == 0.5


def test_pcqm_expert_batch_norm_can_be_frozen_without_freezing_affine():
    model = PCQMGINEExpert()
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()

    batch_norms = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    ]
    assert batch_norms
    assert all(not module.training for module in batch_norms)
    assert all(module.weight.requires_grad for module in batch_norms)
