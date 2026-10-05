"""Focused CPU checks for the composed 500K family hooks."""
from __future__ import annotations

import gc
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from molgap import pcqm_composed_500k as composed
from molgap import pcqm_gptrans_v4
from molgap.gptrans_capacity import BondLocalBlock
from molgap.k1_screen_training import FORBIDDEN_MODEL_FIELDS
from molgap.v4_runtime import state_dict_sha256


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "platforms/_records/kaggle/staging/pcqm_k1_gptrans_package_transfer_500k/inputs"


def _synthetic_ogb_batch(rows: int = 32) -> Batch:
    graphs = []
    for index in range(rows):
        graphs.append(Data(
            x=torch.tensor([[6, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 3), dtype=torch.long),
            random_walk_pe=torch.zeros((1, 16), dtype=torch.float32),
            y=torch.tensor([index / 10], dtype=torch.float32),
        ))
    return Batch.from_data_list(graphs)


def test_factories_load_the_exact_staged_initial_states_on_cpu():
    pins = json.loads((INPUTS / "initial_pins.json").read_text(encoding="utf-8"))
    expected = {
        composed.K1: ("k1_initial.pt", composed.INITIAL_FILES[composed.K1], composed.INITIAL_TENSORS[composed.K1]),
        composed.GP: ("gptrans_initial.pt", composed.INITIAL_FILES[composed.GP], composed.INITIAL_TENSORS[composed.GP]),
    }

    for arm, (filename, file_sha, tensor_sha) in expected.items():
        pin = pins["arms"][arm]
        checkpoint = INPUTS / filename
        assert pin["file"] == filename
        assert pin["sha256"] == file_sha
        assert pin["state_sha256"] == tensor_sha
        assert checkpoint.is_file()
        assert checkpoint.resolve().is_relative_to(INPUTS.resolve())

        model = composed.make_model(arm, checkpoint)
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        assert sum(parameter.numel() for parameter in model.parameters()) == composed.PARAMETERS[arm]
        assert state_dict_sha256(model.state_dict()) == tensor_sha
        if arm == composed.GP:
            local_blocks = [block for block in model.blocks if isinstance(block, BondLocalBlock)]
            assert len(local_blocks) == 12
            assert all(torch.count_nonzero(block.output.weight) == 0 for block in local_blocks)
            assert all(torch.count_nonzero(block.output.bias) == 0 for block in local_blocks)
        del model
        gc.collect()


class _DropoutRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, batch):
        atom_signal = batch.x[:, :1].float() / 6.0
        return self.dropout(atom_signal).reshape(-1) * self.scale


def test_k1_uses_two_teacher_free_forwards_with_live_consistency_gradients(monkeypatch):
    batch = _synthetic_ogb_batch()
    model = _DropoutRegressor()
    calls = []

    def forward(model, batch):
        calls.append(len(calls))
        return model(batch).reshape(-1)

    monkeypatch.setattr(pcqm_gptrans_v4, "_forward", forward)
    torch.manual_seed(4201)
    loss, terms = composed._objective(composed.K1, model, batch, 0.0, 1.0)
    assert calls == [0, 1]
    assert terms["teacher_mse"] is None
    assert float(terms["disagreement"]) > 0
    consistency_gradient = torch.autograd.grad(terms["disagreement"], model.scale)[0]
    assert torch.isfinite(consistency_gradient)
    assert float(consistency_gradient.abs()) > 0
    assert torch.isfinite(loss)

    calls.clear()
    torch.manual_seed(4201)
    signal = composed.qualify_signal(composed.K1, model, batch, 0.0, 1.0)
    assert calls == [0, 1]
    assert signal["accepted"] is True
    assert signal["teacher_used"] is False
    assert signal["gradient_squared_norm"] > 0
    assert composed.make_ema(composed.K1, model) is None


def test_step_exposes_detached_tensor_components_for_epoch_aggregation(monkeypatch):
    batch = _synthetic_ogb_batch(rows=128)
    model = _DropoutRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    monkeypatch.setattr(pcqm_gptrans_v4, "_forward", lambda model, batch: model(batch).reshape(-1))

    torch.manual_seed(4202)
    supervised = composed.step(composed.K1, model, optimizer, batch, 0.0, 1.0)
    components = composed.step.last_components

    assert set(components) == {"supervised_l1", "disagreement", "combined_loss"}
    assert all(isinstance(value, torch.Tensor) and value.ndim == 0 for value in components.values())
    assert all(not value.requires_grad and value.grad_fn is None for value in components.values())
    assert torch.isfinite(supervised)
    assert float(components["disagreement"]) > 0
    assert torch.isfinite(sum(components.values(), torch.zeros(())))


def test_geometry_stripping_uses_the_existing_forbidden_field_contract():
    retained = {"x": torch.zeros((2, 9)), "edge_index": torch.tensor([[0], [1]]),
                "edge_attr": torch.zeros((1, 3)), "y": torch.tensor([0.25])}
    stores = []
    for _ in range(2):
        data = dict(retained)
        data.update({field: torch.ones(1) for field in FORBIDDEN_MODEL_FIELDS})
        dataset = SimpleNamespace(_data=data, slices={field: torch.tensor([0, 1]) for field in data})
        stores.append(SimpleNamespace(datasets=[dataset]))
    roles = {"train": stores[0], "validation": stores[1]}

    composed.strip_geometry(roles)

    assert set(FORBIDDEN_MODEL_FIELDS).isdisjoint(roles["train"].datasets[0]._data)
    assert set(FORBIDDEN_MODEL_FIELDS).isdisjoint(roles["validation"].datasets[0]._data)
    assert set(FORBIDDEN_MODEL_FIELDS).isdisjoint(roles["train"].datasets[0].slices)
    assert set(roles["train"].datasets[0]._data) == set(retained)
    assert set(roles["validation"].datasets[0]._data) == set(retained)


class _PassThroughCore(nn.Module):
    def forward(self, node, pair, padding):
        return node, pair


def test_bond_local_zero_output_learns_through_projection_and_ema_state_resumes():
    torch.manual_seed(42)
    block = BondLocalBlock(_PassThroughCore(), channels=4, pair_channels=2)
    node = torch.tensor([[[0.1, 0.7, 1.4, 2.0], [0.4, 1.1, 1.8, 2.5]]])
    pair = torch.tensor([[[[0.0, 0.2], [0.3, 0.5]], [[0.6, 0.8], [1.0, 1.2]]]])
    padding = torch.zeros((1, 2), dtype=torch.bool)
    edges = (torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    initial_node, _ = block(node, pair, padding, edges)
    assert torch.equal(initial_node, node)
    assert torch.count_nonzero(block.output.weight) == 0
    assert torch.count_nonzero(block.output.bias) == 0

    target = node.clone()
    target[0, 1] += 1.0
    optimizer = torch.optim.SGD(block.parameters(), lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = block(node, pair, padding, edges)
    (prediction - target).square().mean().backward()
    assert block.output.weight.grad is not None
    assert float(block.output.weight.grad.abs().sum()) > 0
    optimizer.step()
    assert torch.count_nonzero(block.output.weight) > 0

    live = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        live.weight.zero_()
    ema = composed.make_ema(composed.GP, live)
    assert ema.decay == pytest.approx(0.999)
    with torch.no_grad():
        live.weight.fill_(1.0)
    ema.update(live)
    assert float(ema.state_dict()["weight"].item()) == pytest.approx(0.001)

    payload = io.BytesIO()
    torch.save(ema.state_dict(), payload)
    payload.seek(0)
    restored_model = nn.Linear(1, 1, bias=False)
    restored = composed.make_ema(composed.GP, restored_model)
    restored.load_state_dict(torch.load(payload, map_location="cpu", weights_only=True))
    with torch.no_grad():
        restored_model.weight.fill_(2.0)
    restored.update(restored_model)
    assert float(restored.state_dict()["weight"].item()) == pytest.approx(0.002999)


def test_both_composed_schedules_match_the_frozen_60_epoch_budget():
    k1 = [composed.schedule(composed.K1, epoch) for epoch in range(60)]
    gp = [composed.schedule(composed.GP, epoch) for epoch in range(60)]
    assert k1[0] == pytest.approx(4e-4)
    assert k1[-1] == pytest.approx(1e-6)
    assert all(left >= right for left, right in zip(k1, k1[1:]))

    assert gp[0] == pytest.approx(2.5e-4)
    assert gp[3] == pytest.approx(1e-3)
    assert gp[4] == pytest.approx(1e-3)
    assert gp[-1] == pytest.approx(1e-6)
    assert all(left <= right for left, right in zip(gp[:4], gp[1:4]))
    assert all(left >= right for left, right in zip(gp[4:], gp[5:]))

    for arm in (composed.K1, composed.GP):
        contract = composed.scientific_contract(arm)
        assert contract["epochs"] == 60
        assert contract["steps_per_epoch"] == 3_906
        assert contract["optimizer_steps"] == 234_360
        assert contract["sample_exposure"] == 29_998_080
        assert contract["physical_batch_per_device"] == 128
