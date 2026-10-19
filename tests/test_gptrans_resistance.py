"""Synthetic CPU contracts only; no frozen artifacts, datasets, or training."""
from dataclasses import replace
from io import BytesIO
from unittest.mock import patch
import weakref

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from molgap.gptrans_bond_local import (
    BondLocalBlock, BondLocalGPTrans, construct_family_model, make_bond_local_model,
)
from molgap.gptrans_execution_backend import (
    bind_model_layout, dense_inputs, prepare_layout, prepared_forward,
)
from molgap.gptrans_resistance import (
    ResistanceBias, apply_addon, policy, resistance_features,
)
from molgap.model_substrate import (
    GPTransMessageInput, ModelSubstrate, SlotBinding, SlotName,
)


def topology(size, edges, real_nodes):
    adjacency = torch.zeros(1, size, size, dtype=torch.bool)
    for source, target in edges:
        adjacency[0, source, target] = True
    real = torch.zeros(1, size, dtype=torch.bool)
    real[0, real_nodes] = True
    return adjacency, real


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 2e-5), (torch.float64, 1e-10)])
def test_bridge_and_six_cycle_channel_values(dtype, tolerance):
    # A bridge attached to a six-cycle still has unit effective resistance.
    edges = [(i, i % 6 + 1) for i in range(1, 7)] + [(6, 7)]
    adjacency, real = topology(9, edges, list(range(1, 8)))
    features = resistance_features(adjacency, real, dtype=dtype)
    assert features.shape == (1, 4, 9, 9)
    assert features.dtype == dtype and not features.requires_grad
    assert torch.isfinite(features).all()
    assert features[0, 0, 6, 7].item() == pytest.approx(1, abs=tolerance)
    assert features[0, 0, 1, 2].item() == pytest.approx(5 / 6, abs=tolerance)
    assert features[0, 3, 1, 2].item() == pytest.approx(1 / 6, abs=tolerance)
    assert features[0, 3, 6, 7].item() == pytest.approx(0, abs=tolerance)
    assert features[0, 3, 1, 3] == 0
    assert torch.equal(features[:, 0].diagonal(dim1=-2, dim2=-1), torch.zeros(1, 9, dtype=dtype))


def test_two_atom_pseudoinverse_diagonal_channels():
    adjacency, real = topology(4, [(1, 2)], [1, 2])
    features = resistance_features(adjacency, real, dtype=torch.float64)
    torch.testing.assert_close(features[0, :, 1, 2], torch.tensor([1., .5, 0., 0.], dtype=torch.float64))
    torch.testing.assert_close(features[0, :, 1, 1], torch.tensor([0., .5, 0., 0.], dtype=torch.float64))


def test_disconnected_padding_virtual_duplicate_and_self_edges_are_masked():
    adjacency, real = topology(8, [(1, 2), (2, 1), (3, 4), (4, 4), (0, 1), (6, 1)], [1, 2, 3, 4, 5])
    features = resistance_features(adjacency, real)
    assert torch.isfinite(features).all()
    for invalid in (0, 6, 7):
        assert torch.count_nonzero(features[:, :, invalid, :]) == 0
        assert torch.count_nonzero(features[:, :, :, invalid]) == 0
    assert torch.count_nonzero(features[:, :, 1:3, 3:6]) == 0
    assert torch.count_nonzero(features[:, :, 3:6, 1:3]) == 0
    assert torch.count_nonzero(features[:, :, 5, :]) == 0
    assert features[0, 0, 1, 2].item() == pytest.approx(1, abs=1e-5)
    clean, _ = topology(8, [(1, 2), (3, 4)], [1, 2, 3, 4, 5])
    torch.testing.assert_close(features, resistance_features(clean, real))


def test_features_are_permutation_equivariant_and_padding_independent():
    adjacency, real = topology(8, [(1, 2), (2, 3), (3, 1), (4, 5)], [1, 2, 3, 4, 5])
    features = resistance_features(adjacency, real, dtype=torch.float64)
    order = torch.tensor([0, 4, 2, 5, 1, 3, 7, 6])
    permuted = resistance_features(adjacency[:, order][:, :, order], real[:, order], dtype=torch.float64)
    torch.testing.assert_close(permuted, features[:, :, order][:, :, :, order])
    smaller = resistance_features(adjacency[:, :6, :6], real[:, :6], dtype=torch.float64)
    torch.testing.assert_close(smaller, features[:, :, :6, :6])
    batched = resistance_features(torch.cat((adjacency, adjacency)), torch.cat((real, real)))
    torch.testing.assert_close(batched[0], batched[1])


def test_edgeless_features_are_finite_and_low_precision_pinv_is_rejected():
    adjacency, real = topology(5, [], [1, 2, 3])
    features = resistance_features(adjacency, real)
    assert torch.isfinite(features).all() and torch.count_nonzero(features) == 0
    with pytest.raises(ValueError, match="float32 or float64"):
        resistance_features(adjacency, real, dtype=torch.float16)


def test_features_disable_autocast_for_pinv_and_connectivity():
    adjacency, real = topology(5, [(1, 2), (2, 3)], [1, 2, 3])
    expected = resistance_features(adjacency, real)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with patch("torch.linalg.pinv", wraps=torch.linalg.pinv) as pinv:
            actual = resistance_features(adjacency, real)
            assert pinv.call_args.args[0].dtype == torch.float32
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def small_model():
    # Reduced synthetic shape uses the real wrapper and attention ABI.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = BondLocalGPTrans(node_channels=16, pair_channels=32, num_layers=2,
                                num_heads=4, dropout=0, drop_path=0)
        model.blocks = nn.ModuleList(BondLocalBlock(block, channels=16) for block in model.blocks)
    return model.eval()


def graph_batch():
    def graph(nodes, edges):
        directed = edges + [(target, source) for source, target in edges]
        edge_index = torch.tensor(directed, dtype=torch.long).reshape(-1, 2).T.contiguous()
        return Data(x=torch.zeros(nodes, 9, dtype=torch.long), edge_index=edge_index,
                    edge_attr=torch.zeros(len(directed), 3, dtype=torch.long))
    return prepare_layout(Batch.from_data_list([
        graph(3, [(0, 1), (1, 2)]),
        graph(6, [(i, (i + 1) % 6) for i in range(6)]),
        graph(5, [(0, 1), (2, 3)]),
    ]))


def test_policy_keeps_exact_bond_local_constructor_and_fixed_parameter_delta():
    assert policy("degree_bond_local_resistance").construct is construct_family_model
    with pytest.raises(ValueError, match="Unknown"):
        policy("reference")
    model = make_bond_local_model()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_871_201
    model.load_state_dict(initial, strict=True)
    bind_model_layout(model)
    rng = torch.get_rng_state().clone()
    assert apply_addon(model, {}) is model
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_871_329
    addon = model._molgap_addons["resistance"]
    assert sum(parameter.numel() for parameter in addon.parameters()) == 128
    assert addon.projection.bias is None and torch.count_nonzero(addon.projection.weight) == 0
    assert isinstance(addon._model_ref, weakref.ReferenceType) and addon._model_ref() is model
    assert all(module is not model for module in addon.modules())
    for name, value in initial.items():
        assert torch.equal(model.state_dict()[name], value)


def test_zero_initial_eval_identity_gradient_and_once_per_forward_pinv():
    model = small_model()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(initial, strict=True)
    bind_model_layout(model)
    batch = graph_batch()
    with torch.no_grad():
        expected = prepared_forward(model, batch)
    apply_addon(model)
    addon = model._molgap_addons["resistance"]
    assert model._dense_inputs.__func__ is dense_inputs
    assert model._molgap_substrate.bindings[0].binding == SlotBinding("gptrans_t", SlotName.PRE_MESSAGE, 1)
    assert len(model.blocks[0].core.attention._forward_pre_hooks) == 1
    assert not model.blocks[1].core.attention._forward_pre_hooks
    with patch("torch.linalg.pinv", wraps=torch.linalg.pinv) as pinv:
        actual = prepared_forward(model, batch)
        assert pinv.call_count == 1
        assert pinv.call_args.kwargs["hermitian"] is True
        assert pinv.call_args.args[0].dtype == torch.float32
        assert not pinv.call_args.args[0].requires_grad
        assert torch.equal(actual, expected)
        actual.square().sum().backward()
        assert addon.projection.weight.grad is not None
        assert torch.isfinite(addon.projection.weight.grad).all()
        assert addon.projection.weight.grad.abs().sum() > 0
        # A new batch must recompute rather than reuse previous graph features.
        prepared_forward(model, prepare_layout(Batch.from_data_list([
            Data(x=torch.zeros(1, 9, dtype=torch.long), edge_index=torch.empty(2, 0, dtype=torch.long),
                 edge_attr=torch.empty(0, 3, dtype=torch.long))])))
        assert pinv.call_count == 2
    assert not hasattr(model, "_execution_layout")


def test_nonzero_projection_keeps_padding_virtual_and_disconnected_pairs_zero():
    model = small_model()
    addon = ResistanceBias(model)
    with torch.no_grad():
        addon.projection.weight.fill_(1)
    batch = graph_batch()
    model._execution_layout = batch.execution_layout
    try:
        node, pair, mask = dense_inputs(model, batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        payload = GPTransMessageInput(node, torch.zeros_like(pair), mask)
        result = addon(payload)
        assert result.node is node and result.key_padding_mask is mask
        assert torch.count_nonzero(result.pair[:, :, 0, :]) == 0
        assert torch.count_nonzero(result.pair[:, :, :, 0]) == 0
        assert torch.count_nonzero(result.pair[0, :, 4:, :]) == 0
        assert torch.count_nonzero(result.pair[0, :, :, 4:]) == 0
        assert torch.count_nonzero(result.pair[2, :, 1:3, 3:6]) == 0
        assert torch.count_nonzero(result.pair[2, :, 3:6, 1:3]) == 0
        assert result.pair[0, :, 1, 2].abs().sum() > 0
    finally:
        del model._execution_layout


def test_reconstruction_state_roundtrip_and_rng_preserved():
    model = bind_model_layout(small_model())
    apply_addon(model)
    model._molgap_substrate.seal()
    with torch.no_grad():
        model._molgap_addons["resistance"].projection.weight.fill_(.02)
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    restored = bind_model_layout(small_model())
    rng = torch.get_rng_state().clone()
    apply_addon(restored)
    assert torch.equal(rng, torch.get_rng_state())
    buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    restored._molgap_substrate.seal()
    assert list(model.state_dict()) == list(restored.state_dict())
    assert list(model._molgap_substrate.named_addon_parameters())[0][0] == "_molgap_addons.resistance.projection.weight"
    with torch.no_grad():
        expected = prepared_forward(model, graph_batch())
        actual = prepared_forward(restored, graph_batch())
    assert torch.equal(expected, actual)
    assert torch.equal(rng, torch.get_rng_state())


def test_wrapper_substrate_targets_reviewed_core_without_changing_local_block():
    class ShiftPair(nn.Module):
        def forward(self, payload):
            return replace(payload, pair=payload.pair + 1)
    model = bind_model_layout(small_model())
    local = model.blocks[0]
    core = local.core
    owner = ModelSubstrate(model, "gptrans_t")
    binding = SlotBinding("gptrans_t", SlotName.PRE_MESSAGE, 1)
    assert owner._target(binding) is core.attention
    owner.attach("shift", ShiftPair(), binding)
    observed = []
    handle = core.attention.register_forward_pre_hook(lambda module, args: observed.append(args[1].detach().clone()))
    batch = graph_batch()
    try:
        prepared_forward(model, batch)
        shifted = observed[-1]
        owner.detach("shift")
        prepared_forward(model, batch)
        torch.testing.assert_close(shifted, observed[-1] + 1)
    finally:
        handle.remove()
    assert model.blocks[0] is local and local.core is core


def test_addon_rejects_wrong_model_unbound_backend_unknown_config_and_missing_layout():
    with pytest.raises(ValueError, match="BondLocalGPTrans"):
        apply_addon(nn.Linear(1, 1))
    model = small_model()
    with pytest.raises(ValueError, match="backend"):
        apply_addon(model)
    bind_model_layout(model)
    with pytest.raises(ValueError, match="configurable"):
        apply_addon(model, {"bias": True})
    apply_addon(model)
    with pytest.raises(ValueError, match="already attached"):
        apply_addon(model)
    with pytest.raises(ValueError, match="prepared execution layout"):
        model._molgap_addons["resistance"](None)


def test_substrate_does_not_unwrap_unreviewed_core_modules():
    model = small_model()
    wrapper = nn.Module()
    wrapper.core = model.blocks[0].core
    model.blocks[0] = wrapper
    owner = ModelSubstrate(model, "gptrans_t")
    with pytest.raises(ValueError, match="declared slot"):
        owner._target(SlotBinding("gptrans_t", SlotName.PRE_MESSAGE, 1))
