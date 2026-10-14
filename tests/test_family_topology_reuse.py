"""Exact CPU topology reuse; synthetic inputs only, no native speed claim."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch, Data

from molgap import gptrans_bond_local as local
from molgap.gptrans import OGBGPTransTiny
from molgap.gptrans_execution_backend import (
    bind_model_layout, prepare_layout, prepared_forward, prepared_shortest_path,
)
from molgap.k1_execution_layout import layout_from_cpu_batch


@pytest.fixture(autouse=True)
def restore_cpu_runtime():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def synthetic_batch(empty=False):
    graphs = []
    for nodes in (1, 2, 4):
        edge = torch.stack((torch.arange(nodes - 1), torch.arange(1, nodes)))
        edge = torch.cat((edge, edge.flip(0)), 1)
        if empty:
            edge = torch.empty(2, 0, dtype=torch.long)
        graphs.append(Data(x=torch.zeros(nodes, 9, dtype=torch.long), edge_index=edge,
                           edge_attr=torch.zeros(edge.shape[1], 3, dtype=torch.long)))
    return Batch.from_data_list(graphs)


def test_k1_padding_mask_is_built_once_per_private_layout():
    batch = synthetic_batch()
    layout = layout_from_cpu_batch(batch, "cpu")
    mask = layout.key_padding_mask
    assert torch.equal(mask, ~layout.valid.unsqueeze(1))
    assert mask.shape == (3, 1, 4)
    assert layout.key_padding_mask is mask
    assert layout_from_cpu_batch(batch, "cpu").key_padding_mask is not mask


class RebuiltMaskLayout:
    """Test-only oracle for the previous per-mixer mask expression."""
    def __init__(self, layout):
        self.layout = layout

    def __getattr__(self, name):
        return getattr(self.layout, name)

    @property
    def key_padding_mask(self):
        return ~self.layout.valid.unsqueeze(1)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("training", [False, True])
def test_k1_mask_reuse_keeps_outputs_gradients_optimizer_and_rng(training):
    from molgap.qm9_neural_atom import _NeuralAtomMixerFactory
    torch.manual_seed(123)
    mixer = _NeuralAtomMixerFactory.make(hidden_channels=8, latent_channels=8,
        num_heads=2, max_slots=4, active_slots=1, dropout=.2)
    torch.nn.init.normal_(mixer.return_projection.weight, std=.1)
    initial = deepcopy(mixer.state_dict())
    batch = synthetic_batch()
    layout = layout_from_cpu_batch(batch, "cpu")
    hidden = torch.randn(7, 8)
    rng = torch.get_rng_state().clone()
    results = []
    for use_cache in (False, True):
        mixer.load_state_dict(initial)
        mixer.train(training)
        mixer._execution_dense_layout = layout if use_cache else RebuiltMaskLayout(layout)
        optimizer = torch.optim.AdamW(mixer.parameters(), lr=.001)
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        value = hidden.clone().requires_grad_()
        # Mean2 reuses the layout for two stochastic forwards, not their activations.
        output = (mixer(value, batch.batch), mixer(value, batch.batch))
        loss = (output[0].square().mean() + output[1].square().mean()) / 2
        loss.backward()
        gradients = [parameter.grad.clone() for parameter in mixer.parameters()]
        optimizer.step()
        results.append(deepcopy((tuple(item.detach() for item in output), value.grad, gradients, mixer.state_dict(),
                                 optimizer.state_dict(), torch.get_rng_state())))
        del mixer._execution_dense_layout
    assert_tree_equal(*results)


class RepeatedTargetsGPTrans(local.BondLocalGPTrans):
    """Test-only oracle retaining the prior per-layer integer expression."""
    def forward(self, x, edge_index, edge_attr, batch, random_walk_pe=None):
        node, pair, padding = self._dense_inputs(x, edge_index, edge_attr, batch)
        graph, source, target = self._local_edges(edge_index, batch, len(x))
        edges = (graph, source + 1, target + 1)
        for block in self.blocks:
            node, pair = block(node, pair, padding, edges)
        return self.readout(torch.cat((node[:, 0], pair[:, :, 0, 0]), -1))


def small_local_model(cls):
    model = OGBGPTransTiny(node_channels=16, pair_channels=4, num_layers=3,
        num_heads=2, shortest_path_cap=6, dropout=.2, drop_path=.1)
    model.__class__ = cls
    model.blocks = torch.nn.ModuleList(local.BondLocalBlock(block, channels=16,
        pair_channels=4) for block in model.blocks)
    for block in model.blocks:
        torch.nn.init.normal_(block.output.weight, std=.1)
    return bind_model_layout(model)


@pytest.mark.parametrize("empty", [False, True])
def test_local_targets_once_keeps_two_updates_and_rng(empty, monkeypatch):
    torch.manual_seed(123)
    oracle = small_local_model(RepeatedTargetsGPTrans)
    candidate = small_local_model(local.BondLocalGPTrans)
    candidate.load_state_dict(oracle.state_dict())
    batch = prepare_layout(synthetic_batch(empty))
    rng = torch.get_rng_state().clone()
    original = local._flat_bond_targets
    counts = []
    def counted(*args):
        counts.append(None)
        return original(*args)
    monkeypatch.setattr(local, "_flat_bond_targets", counted)
    results, builds = [], []
    for model in (oracle, candidate):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=True)
        torch.set_rng_state(rng)
        counts.clear()
        steps = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = prepared_forward(model, batch)
            output.square().mean().backward()
            gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            steps.append(deepcopy((output.detach(), gradients, model.state_dict(), optimizer.state_dict(),
                                  torch.get_rng_state())))
        builds.append(len(counts))
        results.append(steps)
    assert builds == [6, 2]
    assert_tree_equal(*results)
    assert not hasattr(candidate, "_execution_layout")


@pytest.mark.parametrize("nodes", [1, 2, 4, 9])
@pytest.mark.parametrize("cap", [1, 3, 20])
def test_simple_path_bound_matches_original_capped_reachability(nodes, cap):
    torch.manual_seed(123)
    adjacency = torch.rand(5, nodes, nodes) < .3
    # Include directed cycles, self-loops, edgeless graphs and disconnected padding.
    adjacency[0].zero_()
    valid = torch.arange(nodes)[None, :] < torch.arange(1, 6).clamp_max(nodes)[:, None]
    pair_mask = valid[:, :, None] & valid[:, None, :]
    adjacency &= pair_mask
    model = SimpleNamespace(shortest_path_cap=cap)
    expected = OGBGPTransTiny._shortest_path(model, adjacency, pair_mask)
    calls = []
    original = torch.bmm
    from unittest.mock import patch
    def counted(*args):
        calls.append(None)
        return original(*args)
    with patch.object(torch, "bmm", counted):
        actual = prepared_shortest_path(model, adjacency, pair_mask)
    assert torch.equal(actual, expected)
    assert len(calls) == max(0, min(cap, nodes - 1) - 1)


def test_retained_local_model_parameter_identity_is_unchanged():
    from molgap.v4_runtime import state_dict_sha256
    model = local.make_bond_local_model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_871_201
    assert state_dict_sha256(model.state_dict()) == (
        "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b")
