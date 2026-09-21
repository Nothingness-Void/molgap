"""Unit tests for GPTrans Dual-Stream Attentive Readout (DSAR)."""
from __future__ import annotations

import torch
import pytest

from molgap.gptrans import OGBGPTransTiny
from molgap.noisy_nodes import GPTransNoisyNodes
from molgap.gptrans_variants import apply_variant


def test_dual_stream_readout_shapes_and_backward():
    x = torch.zeros((12, 9), dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 0, 5, 6, 7, 4, 9, 8]],
        dtype=torch.long,
    )
    edge_attr = torch.zeros((10, 3), dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    # 1. Attentive readout
    model_att = OGBGPTransTiny(num_layers=2, readout_mode="dual_stream_attentive")
    out_att = model_att(x, edge_index, edge_attr, batch)
    assert out_att.shape == (3, 1)
    out_att.sum().backward()

    # 2. Mean readout
    model_mean = OGBGPTransTiny(num_layers=2, readout_mode="dual_stream_mean")
    out_mean = model_mean(x, edge_index, edge_attr, batch)
    assert out_mean.shape == (3, 1)
    out_mean.sum().backward()

    # 3. Legacy virtual readout
    model_virt = OGBGPTransTiny(num_layers=2, readout_mode="virtual")
    out_virt = model_virt(x, edge_index, edge_attr, batch)
    assert out_virt.shape == (3, 1)


def test_dual_stream_with_noisy_nodes_and_pair_norm():
    x = torch.zeros((8, 9), dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.zeros((4, 3), dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    model = GPTransNoisyNodes(num_layers=2, readout_mode="dual_stream_attentive")
    model = apply_variant(model, "pair_update_norm")

    model.train()
    pred, aux_loss = model(
        x, edge_index, edge_attr, batch, return_aux_loss=True
    )
    assert pred.shape == (2, 1)
    assert aux_loss.ndim == 0
    (pred.sum() + 0.1 * aux_loss).backward()

    model.eval()
    pred_eval = model(x, edge_index, edge_attr, batch)
    assert pred_eval.shape == (2, 1)
