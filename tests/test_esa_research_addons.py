"""Synthetic ESA residual checks for Luna; no real data or scientific run."""
from copy import deepcopy
from dataclasses import asdict
from types import MethodType

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch

from molgap.esa_invariant import ESAInvariant, InvariantPMA
from molgap.esa_model import ESA
from molgap.esa_native import ESANative, ESANativeConfig, _attention_masks, packed_native_tokens
from molgap.esa_research_addons import apply_addon
from molgap.model_substrate import ModelSubstrate, SlotBinding, SlotName
from molgap.modern2d_inputs import validated_topology_views
from modern2d_fixtures import graph, model_config


CONFIG = {"variant": "identity_residual"}


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def native():
    return ESANative(ESANativeConfig(**model_config("esa_small")))


def mixed_graph(source=0):
    return graph(6, source, edges=torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 1]]))


def isolated_graph(nodes=1, source=1):
    return graph(nodes, source, edges=torch.empty((2, 0), dtype=torch.long))


def test_formal_base_state_parameters_modules_config_and_rng_are_unchanged():
    torch.manual_seed(42)
    model = ESANative().eval()
    config, values = model.config, asdict(model.config)
    state = deepcopy(model.state_dict())
    parameters, modules = dict(model.named_parameters()), dict(model.named_modules())
    count = sum(parameter.numel() for parameter in model.parameters())
    rng = torch.get_rng_state().clone()
    addon_config = deepcopy(CONFIG)
    assert apply_addon(model, addon_config) is model
    assert type(model) is ESANative and model.config is config
    assert asdict(model.config) == values and addon_config == CONFIG
    assert torch.equal(torch.get_rng_state(), rng)
    assert sum(parameter.numel() for parameter in model.parameters()) == count
    assert model.state_dict().keys() == state.keys()
    for name, value in state.items():
        assert torch.equal(model.state_dict()[name], value), name
    assert dict(model.named_parameters()).keys() == parameters.keys()
    for name, parameter in parameters.items():
        assert dict(model.named_parameters())[name] is parameter, name
    for name, module in modules.items():
        assert dict(model.named_modules())[name] is module, name
    assert isinstance(model.pool, InvariantPMA)
    assert "forward" not in model.__dict__ and "forward" not in model.pool.__dict__
    assert all(isinstance(block.forward, MethodType) for block in model.encoder)
    owner = model._molgap_substrate
    assert type(owner) is ModelSubstrate and owner.family == "esa_small"
    assert len(owner.bindings) == 1
    binding = owner.bindings[0]
    assert binding.binding == SlotBinding("esa_small", SlotName.PRE_READOUT)
    declaration = model._molgap_addons[binding.name]
    assert type(declaration) is nn.Identity and not declaration.training
    assert not tuple(declaration.parameters()) and not tuple(declaration.buffers())
    payload = object()
    assert declaration(payload) is payload


@pytest.mark.parametrize("kind", ["M", "S"])
def test_actual_encoder_block_matches_manual_equation_and_keeps_padding_mask(kind):
    torch.manual_seed(42)
    model = apply_addon(native(), CONFIG)
    block = next(block for block in model.encoder if block.kind == kind)
    endpoints = torch.tensor([[[0, 1], [1, 0], [-1, -1]],
                              [[0, -1], [1, -1], [2, -1]]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    masks = _attention_masks(endpoints, valid)
    allowed = masks[0 if kind == "M" else 1]
    x = torch.randn(2, 3, model.config.hidden) * 2 + 3
    normalized = block.norm(x)
    attention = block.attention(normalized, normalized, allowed)
    out = x + attention
    expected = out + block.mlp(block.norm_mlp(out))
    torch.testing.assert_close(block(x, allowed), expected, atol=0, rtol=0)
    legacy = normalized + attention
    legacy = legacy + block.mlp(block.norm_mlp(legacy))
    assert not torch.allclose(expected, legacy)
    perturbed = x.clone()
    perturbed[0, 2] = torch.arange(model.config.hidden) * 10.0
    torch.testing.assert_close(block(x, allowed)[0, :2], block(perturbed, allowed)[0, :2],
                               atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["M", "S"])
def test_zero_branches_leave_the_original_identity(kind, monkeypatch):
    model = apply_addon(native(), CONFIG)
    block = next(block for block in model.encoder if block.kind == kind)
    monkeypatch.setattr(block.attention, "forward", lambda q, kv, allowed=None: torch.zeros_like(q))
    monkeypatch.setattr(block.mlp, "forward", torch.zeros_like)
    x = (torch.randn(2, 4, model.config.hidden) + 3).requires_grad_()
    output = block(x)
    torch.testing.assert_close(output, x, atol=0, rtol=0)
    gradient, = torch.autograd.grad(output.sum(), (x,))
    torch.testing.assert_close(gradient, torch.ones_like(x), atol=0, rtol=0)


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("learning_rate", [3e-4, 1e-4])
def test_same_base_weights_support_finite_single_update_in_each_factorial_arm(residual, learning_rate):
    torch.manual_seed(42)
    model = native()
    state = deepcopy(model.state_dict())
    if residual:
        apply_addon(model, CONFIG)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in state.items())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    data = Batch.from_data_list([mixed_graph(), isolated_graph()])
    output = model(data)
    assert output.shape == (2,) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(model.state_dict()[name], value) for name, value in state.items())
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
        moments = optimizer.state[parameter]
        assert int(moments["step"]) == 1
        assert torch.isfinite(moments["exp_avg"]).all()
        assert torch.isfinite(moments["exp_avg_sq"]).all()


def test_padded_batch_isolated_atoms_and_graph_order_preserve_native_behavior(monkeypatch):
    torch.manual_seed(42)
    model = apply_addon(native().eval(), CONFIG)
    graphs = [mixed_graph(), isolated_graph(), isolated_graph(4, 2), graph(3, 3)]
    data = Batch.from_data_list(graphs)
    before = deepcopy(data.to_dict())
    with torch.no_grad():
        output = model(data)
        # The substrate owns a public __call__ frame even for the reference path.
        with monkeypatch.context() as patch:
            patch.setattr(model, "forward", model.forward_reference)
            reference = model(data)
        singles = torch.cat([model(Batch.from_data_list([sample])) for sample in graphs])
        reversed_output = model(Batch.from_data_list(list(reversed(graphs))))
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=4e-6, rtol=2e-5)
    torch.testing.assert_close(output, singles, atol=4e-6, rtol=2e-5)
    torch.testing.assert_close(output.flip(0), reversed_output, atol=4e-6, rtol=2e-5)
    assert all(torch.equal(data[key], value) for key, value in before.items())
    changed = isolated_graph()
    changed.x[0, 0] += 1
    assert not torch.equal(output[1:2], model(Batch.from_data_list([changed])))
    features, types, endpoints, valid = packed_native_tokens(validated_topology_views(data))
    perturbed = features.clone()
    perturbed[~valid] = 1000
    with monkeypatch.context() as patch, torch.no_grad():
        patch.setattr("molgap.esa_native.packed_native_tokens",
                      lambda views: (perturbed, types, endpoints, valid))
        altered = model(data)
    torch.testing.assert_close(output, altered, atol=0, rtol=0)


def test_atom_and_directed_bond_order_preserve_outputs_and_gradients():
    torch.manual_seed(42)
    model = apply_addon(native().eval(), CONFIG)
    original = mixed_graph()
    other = original.clone()
    atom_order = torch.tensor([5, 2, 0, 4, 1, 3])
    inverse = torch.argsort(atom_order)
    bond_order = torch.tensor([3, 1, 4, 0, 2])
    other.x = original.x[atom_order]
    other.edge_index = inverse[original.edge_index[:, bond_order]]
    other.edge_attr = original.edge_attr[bond_order]
    output = model(Batch.from_data_list([original]))
    before = torch.autograd.grad(output.sum(), tuple(model.parameters()))
    changed = model(Batch.from_data_list([other]))
    after = torch.autograd.grad(changed.sum(), tuple(model.parameters()))
    torch.testing.assert_close(output, changed, atol=4e-6, rtol=2e-5)
    for actual, expected in zip(before, after):
        torch.testing.assert_close(actual, expected, atol=7e-5, rtol=3e-4)


def test_strict_state_reload_and_sealed_declaration():
    torch.manual_seed(42)
    model = apply_addon(native(), CONFIG)
    model._molgap_substrate.seal()
    data = Batch.from_data_list([mixed_graph(), isolated_graph()])
    output = model(data)
    saved = deepcopy(model.state_dict())
    restored = native()
    restored.load_state_dict(saved, strict=True)
    apply_addon(restored, CONFIG)
    torch.testing.assert_close(output, restored(data), atol=0, rtol=0)
    restored.load_state_dict(saved, strict=True)
    torch.testing.assert_close(output, restored(data), atol=0, rtol=0)


@pytest.mark.parametrize("config", [None, [], {}, {"variant": "unknown"},
                                   {"variant": None}, {"variant": True},
                                   {"variant": "identity_residual", "extra": 1}])
def test_bad_configuration_is_rejected_without_mutating_model(config):
    model = native()
    with pytest.raises(ValueError, match="variant"):
        apply_addon(model, config)
    assert not hasattr(model, "_molgap_substrate")
    assert all("forward" not in block.__dict__ for block in model.encoder)


@pytest.mark.parametrize("factory", [nn.Linear, ESA, ESAInvariant])
def test_other_model_types_are_rejected(factory):
    model = factory(2, 2) if factory is nn.Linear else factory()
    with pytest.raises(TypeError, match="ESANative"):
        apply_addon(model, CONFIG)
    assert not hasattr(model, "_molgap_substrate")


def test_reapplication_and_sealed_owner_fail_before_encoder_changes():
    model = apply_addon(native(), CONFIG)
    methods = [block.forward for block in model.encoder]
    with pytest.raises(ValueError, match="already applied"):
        apply_addon(model, CONFIG)
    assert methods == [block.forward for block in model.encoder]
    assert len(model._molgap_substrate.bindings) == 1
    sealed = native()
    ModelSubstrate(sealed, "esa_small").seal()
    with pytest.raises(RuntimeError, match="sealed"):
        apply_addon(sealed, CONFIG)
    assert all("forward" not in block.__dict__ for block in sealed.encoder)


def test_exact_native_type_and_unmodified_encoder_are_required():
    class OtherNative(ESANative):
        pass

    subclass = OtherNative(ESANativeConfig(**model_config("esa_small")))
    with pytest.raises(TypeError, match="exact ESANative"):
        apply_addon(subclass, CONFIG)
    model = native()
    model.encoder[0].forward = lambda x, allowed=None: x
    with pytest.raises(ValueError, match="unmodified native"):
        apply_addon(model, CONFIG)
    assert not hasattr(model, "_molgap_substrate")


def test_existing_compatible_owner_is_reused_and_wrong_family_is_rejected():
    model = native()
    owner = ModelSubstrate(model, "esa_small")
    apply_addon(model, CONFIG)
    assert model._molgap_substrate is owner and len(owner.bindings) == 1
    other = native()
    ModelSubstrate(other, "ppgt_small")
    with pytest.raises(ValueError, match="owner differs"):
        apply_addon(other, CONFIG)
    assert all("forward" not in block.__dict__ for block in other.encoder)
