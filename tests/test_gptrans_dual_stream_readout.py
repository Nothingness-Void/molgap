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


def test_dual_stream_optimizer_step_both_arms():
    from torch_geometric.data import Data, Batch
    from molgap.noisy_nodes import _optimizer_step_noisy_nodes
    from molgap.pcqm_gptrans_v4 import ExponentialMovingAverage

    g1 = Data(
        x=torch.zeros((4, 9), dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        edge_attr=torch.zeros((4, 3), dtype=torch.long),
        y=torch.tensor([4.5], dtype=torch.float),
    )
    g2 = Data(
        x=torch.zeros((4, 9), dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        edge_attr=torch.zeros((4, 3), dtype=torch.long),
        y=torch.tensor([5.2], dtype=torch.float),
    )
    batch = Batch.from_data_list([g1, g2])
    mean = torch.tensor(5.0)
    std = torch.tensor(1.0)

    # Arm A: GPTrans Baseline + Pair Norm + DSAR (no noisy nodes)
    model_a = OGBGPTransTiny(num_layers=2, readout_mode="dual_stream_attentive")
    model_a = apply_variant(model_a, "pair_update_norm")
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    ema_a = ExponentialMovingAverage(model_a)
    gap_loss_a, aux_a = _optimizer_step_noisy_nodes(
        model_a, opt_a, ema_a, batch, mean, std, check_finite=True
    )
    assert gap_loss_a > 0.0
    assert aux_a == 0.0

    # Arm B: GPTrans Noisy Nodes + Pair Norm + DSAR
    model_b = GPTransNoisyNodes(num_layers=2, readout_mode="dual_stream_attentive", noise_std=0.15, loss_weight=0.1)
    model_b = apply_variant(model_b, "pair_update_norm")
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    ema_b = ExponentialMovingAverage(model_b)
    gap_loss_b, aux_b = _optimizer_step_noisy_nodes(
        model_b, opt_b, ema_b, batch, mean, std, check_finite=True
    )
    assert gap_loss_b > 0.0
    assert aux_b > 0.0

