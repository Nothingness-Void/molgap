"""Exact synthetic topology reuse; no dataset, retained training or speed claim."""
from copy import deepcopy
from dataclasses import replace
import random
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from molgap import gptrans_capacity as local
from molgap.gptrans import OGBGPTransTiny
from molgap.gptrans_execution_backend import (
    bind_model_layout, prepare_layout, prepared_forward, prepared_shortest_path,
)
from molgap.model_substrate import ModelSubstrate, SlotBinding, SlotName


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


@pytest.fixture(autouse=True)
def restore_cpu_runtime():
    rng, threads = rng_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        restore_rng(rng)


def assert_tree_equal(left, right):
    if torch.is_tensor(left):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def synthetic_batch(nodes=(1, 2, 4), *, empty=False):
    graphs = []
    for size in nodes:
        # Directed cycle/self-loop cases exercise ordering without symmetrizing.
        edge = torch.stack((torch.arange(size), torch.arange(size).roll(-1)))
        if empty:
            edge = torch.empty(2, 0, dtype=torch.long)
        graphs.append(Data(x=torch.zeros(size, 9, dtype=torch.long), edge_index=edge,
            edge_attr=torch.zeros(edge.shape[1], 3, dtype=torch.long)))
    return Batch.from_data_list(graphs)


class RepeatedTargetsGPTrans(local.BondLocalGPTrans):
    """Previous per-layer integer expression, retained only as a test oracle."""
    def forward(self, x, edge_index, edge_attr, batch, random_walk_pe=None):
        node, pair, padding = self._dense_inputs(x, edge_index, edge_attr, batch)
        graph, source, target = self._local_edges(edge_index, batch, len(x))
        edges = graph, source + 1, target + 1
        for block in self.blocks:
            node, pair = block(node, pair, padding, edges)
        return self.readout(torch.cat((node[:, 0], pair[:, :, 0, 0]), -1))


class ReadoutDelta(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.tensor(.125))

    def forward(self, payload):
        return replace(payload, embedding=payload.embedding + self.shift)


def small_local_model(cls, *, pair_channels=4):
    model = OGBGPTransTiny(node_channels=16, pair_channels=pair_channels, num_layers=3,
        num_heads=2, shortest_path_cap=6, dropout=.2, drop_path=.1)
    model.__class__ = cls
    model.blocks = torch.nn.ModuleList(local.BondLocalBlock(block, channels=16,
        pair_channels=pair_channels) for block in model.blocks)
    for block in model.blocks:
        torch.nn.init.normal_(block.output.weight, std=.1)
    bind_model_layout(model)
    substrate = ModelSubstrate(model, "gptrans_t")
    substrate.attach("readout_delta", ReadoutDelta(), SlotBinding("gptrans_t", SlotName.PRE_READOUT))
    substrate.seal()
    return model


def two_updates(model, batches, rng, *, training=True):
    model.train(training)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=True)
    restore_rng(rng)
    steps = []
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        # Two forwards retain independent dropout draws and autograd activations.
        output = prepared_forward(model, batch), prepared_forward(model, batch)
        (output[0].square().mean() + output[1].square().mean()).div(2).backward()
        gradients = {name: None if parameter.grad is None else parameter.grad.clone()
                     for name, parameter in model.named_parameters()}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        steps.append(deepcopy((tuple(value.detach() for value in output), gradients,
                               model.state_dict(), optimizer.state_dict(), rng_state())))
        assert not hasattr(model, "_execution_layout")
        assert not model._molgap_substrate._active
    return steps


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_targets_and_path_bound_keep_two_updates_gradients_optimizer_rng(empty, training, monkeypatch):
    torch.manual_seed(123)
    oracle = small_local_model(RepeatedTargetsGPTrans)
    # Preserve the full original reachability expansion in the oracle.
    oracle._shortest_path = MethodType(OGBGPTransTiny._shortest_path, oracle)
    candidate = small_local_model(local.BondLocalGPTrans)
    candidate.load_state_dict(oracle.state_dict())
    batches = [prepare_layout(synthetic_batch(empty=empty)),
               prepare_layout(synthetic_batch((3, 1, 2), empty=empty))]
    first_layout, second_layout = (batch.execution_layout for batch in batches)
    retained_layouts = [deepcopy(batch.execution_layout) for batch in batches]
    assert first_layout is not second_layout
    assert first_layout["max_nodes"] == 4 and second_layout["max_nodes"] == 3
    original = local._flat_bond_targets
    calls = []
    def counted(*args):
        calls.append(None)
        return original(*args)
    monkeypatch.setattr(local, "_flat_bond_targets", counted)
    rng = rng_state()
    expected = two_updates(oracle, batches, rng, training=training)
    assert len(calls) == 12
    calls.clear()
    actual = two_updates(candidate, batches, rng, training=training)
    assert len(calls) == 4
    assert_tree_equal(actual, expected)
    for batch, expected_layout in zip(batches, retained_layouts):
        assert_tree_equal(batch.execution_layout, expected_layout)


@pytest.mark.parametrize("kind", ["instance", "controlled", "wrapped", "bond-return"])
def test_legacy_derived_wrapped_and_instance_forward_abis_remain_exact(kind):
    from molgap.gptrans_local_control import ControlledBondLocalBlock
    from molgap.gptrans_local_relation import ConnectedPairLocalBlock, BondPairReturnBlock
    torch.manual_seed(123)
    oracle = small_local_model(RepeatedTargetsGPTrans, pair_channels=32)
    candidate = small_local_model(local.BondLocalGPTrans, pair_channels=32)
    for model in (oracle, candidate):
        block = model.blocks[1]
        if kind == "instance":
            def positional(self, node, pair, padding, edges):
                return local.BondLocalBlock.forward(self, node, pair, padding, edges)
            block.forward = MethodType(positional, block)
        elif kind == "controlled":
            block.__class__ = ControlledBondLocalBlock
        elif kind == "wrapped":
            model.blocks[1] = ConnectedPairLocalBlock(block)
        else:
            block.__class__ = BondPairReturnBlock
            block.pair_return = torch.nn.Linear(64, 32)
    candidate.load_state_dict(oracle.state_dict())
    batches = [prepare_layout(synthetic_batch()), prepare_layout(synthetic_batch((3, 1, 2)))]
    rng = rng_state()
    assert_tree_equal(two_updates(oracle, batches, rng), two_updates(candidate, batches, rng))


@pytest.mark.parametrize("nodes", [1, 2, 4, 9])
@pytest.mark.parametrize("cap", [1, 3, 20])
def test_simple_path_bound_keeps_directed_cycles_self_loops_padding_and_sentinel(nodes, cap, monkeypatch):
    generator = torch.Generator().manual_seed(123)
    adjacency = torch.rand(5, nodes, nodes, generator=generator) < .3
    adjacency[0].zero_()
    valid = torch.arange(nodes)[None, :] < torch.arange(1, 6).clamp_max(nodes)[:, None]
    pair_mask = valid[:, :, None] & valid[:, None, :]
    adjacency &= pair_mask
    model = SimpleNamespace(shortest_path_cap=cap)
    expected = OGBGPTransTiny._shortest_path(model, adjacency, pair_mask)
    calls = []
    original = torch.bmm
    def counted(*args):
        calls.append(None)
        return original(*args)
    monkeypatch.setattr(torch, "bmm", counted)
    rng = rng_state()
    actual = prepared_shortest_path(model, adjacency, pair_mask)
    assert torch.equal(actual, expected)
    assert len(calls) == max(0, min(cap, nodes - 1) - 1)
    assert bool((actual[~pair_mask] == cap + 1).all())
    assert_tree_equal(rng_state(), rng)


def test_prepared_layout_lifetime_clears_after_failure_and_for_new_batch():
    model = small_local_model(local.BondLocalGPTrans).eval()
    first, second = prepare_layout(synthetic_batch()), prepare_layout(synthetic_batch((2, 1)))
    def fail(*args):
        raise RuntimeError("synthetic block fault")
    handle = model.blocks[0].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="block fault"):
        prepared_forward(model, first)
    handle.remove()
    assert not hasattr(model, "_execution_layout") and not model._molgap_substrate._active
    assert prepared_forward(model, second).shape == (2,)
    assert not hasattr(model, "_execution_layout")
    model._execution_layout = first.execution_layout
    with pytest.raises(ValueError, match="Nested"):
        prepared_forward(model, second)
    assert model._execution_layout is first.execution_layout
    del model._execution_layout


def test_all_legacy_blocks_do_not_build_an_unused_target_cache(monkeypatch):
    class Legacy(torch.nn.Module):
        def forward(self, node, pair, padding, edges):
            return node, pair
    def forbidden(*args):
        raise AssertionError("Legacy-only forward constructed unused targets")
    monkeypatch.setattr(local, "_flat_bond_targets", forbidden)
    node, pair, padding = torch.ones(2, 3, 8), torch.ones(2, 2, 3, 3), torch.zeros(2, 1, 1, 3, dtype=torch.bool)
    empty = torch.empty(0, dtype=torch.long)
    actual = local._bond_local_blocks([Legacy(), Legacy()], node, pair, padding, (empty, empty, empty))
    assert actual[0] is node and actual[1] is pair


def test_positional_width_change_rebuilds_the_target_cache(monkeypatch):
    class Core(torch.nn.Module):
        def forward(self, node, pair, padding):
            return node, pair
    class Grow(torch.nn.Module):
        def forward(self, node, pair, padding, edges):
            return (torch.nn.functional.pad(node, (0, 0, 0, 1)),
                    torch.nn.functional.pad(pair, (0, 1, 0, 1)))
    torch.manual_seed(123)
    blocks = torch.nn.ModuleList([local.BondLocalBlock(Core(), channels=8, pair_channels=2),
                                 Grow(), local.BondLocalBlock(Core(), channels=8, pair_channels=2)])
    for block in (blocks[0], blocks[2]):
        torch.nn.init.normal_(block.output.weight, std=.1)
    node, pair = torch.randn(2, 3, 8), torch.randn(2, 2, 3, 3)
    padding = torch.zeros(2, 1, 1, 3, dtype=torch.bool)
    edges = torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([2, 1])
    expected_node, expected_pair = node, pair
    for block in blocks:
        expected_node, expected_pair = block(expected_node, expected_pair, padding, edges)
    original, widths = local._flat_bond_targets, []
    def counted(graph, target, width):
        widths.append(width)
        return original(graph, target, width)
    monkeypatch.setattr(local, "_flat_bond_targets", counted)
    actual = local._bond_local_blocks(blocks, node, pair, padding, edges)
    assert widths == [3, 4]
    assert_tree_equal(actual, (expected_node, expected_pair))


def test_local_bond_initial_tensor_identity_is_unchanged():
    from molgap.gptrans_author_variants import apply_author_variant
    from molgap.v4_runtime import state_dict_sha256
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        base = apply_author_variant(OGBGPTransTiny(), "degree_scale").state_dict()
    before = rng_state()
    model = local.construct("degree_bond_local_ema999", base)
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_871_201
    assert state_dict_sha256(model.state_dict()) == (
        "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b")
    assert_tree_equal(rng_state(), before)
