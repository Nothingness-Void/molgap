"""Synthetic surgery contracts only; no data roles or training-loop copies."""
import io

import pytest
import torch
from torch import nn

from molgap.k1_post_layernorm import apply_addon
from molgap.model_substrate import ModelSubstrate, ReadoutInput, SlotName
from molgap.qm9_neural_atom import make_encoder


@pytest.fixture
def model():
    return make_encoder("neural_atom_k1")


@pytest.fixture
def inputs():
    return {
        "x": torch.tensor([[0] * 9, [1] + [0] * 8, [2] + [0] * 8,
                           [3] + [0] * 8, [4] + [0] * 8]),
        "edge_index": torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]]),
        "edge_attr": torch.zeros(6, 3, dtype=torch.long),
        "batch": torch.tensor([0, 0, 0, 1, 1]),
        "random_walk_pe": torch.arange(80, dtype=torch.float32).reshape(5, 16) / 80,
    }


def _norm_prefixes():
    return tuple(f"local_blocks.{i}.{name}." for i in range(9)
                 for name in ("norm1", "norm3"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_surgery_preserves_other_state_parameters_and_rng(model, dtype):
    model.to(dtype=dtype).eval()
    with torch.no_grad():
        for i, block in enumerate(model.local_blocks):
            for name in ("norm1", "norm3"):
                norm = getattr(block, name).module
                norm.weight.fill_(0.75 + i / 20)
                norm.bias.fill_(-0.25 + i / 30)
                norm.running_mean.fill_(2)
                norm.running_var.fill_(3)
                norm.num_batches_tracked.fill_(7)
        model.local_blocks[0].norm1.module.weight.requires_grad_(False)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    parameters = dict(model.named_parameters())
    modules = dict(model.named_modules())
    count = sum(p.numel() for p in model.parameters())
    assert sum(isinstance(m, nn.BatchNorm1d) for m in model.modules()) == 18
    rng = torch.get_rng_state().clone()
    assert apply_addon(model, {}) is model
    assert torch.equal(rng, torch.get_rng_state())
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    assert sum(p.numel() for p in model.parameters()) == count
    after = model.state_dict()
    prefixes = _norm_prefixes()
    assert {k for k in after if not k.startswith(prefixes)} == {
        k for k in before if not k.startswith(prefixes)}
    for key in before:
        if not key.startswith(prefixes):
            assert torch.equal(before[key], after[key]), key
    for key, parameter in model.named_parameters():
        if not key.startswith(prefixes):
            assert parameter is parameters[key], key
    for key, module in modules.items():
        if key and not key.startswith(tuple(p[:-1] for p in prefixes)):
            assert dict(model.named_modules())[key] is module, key
    for prefix in prefixes:
        norm = model.get_submodule(prefix[:-1])
        assert type(norm) is nn.LayerNorm
        assert norm.normalized_shape == (192,)
        assert norm.eps == 1e-5 and not norm.training
        assert norm.weight.dtype == dtype and norm.weight.device.type == "cpu"
        for field in ("weight", "bias"):
            assert torch.equal(getattr(norm, field), before[prefix + "module." + field])
            assert getattr(norm, field).requires_grad == parameters[
                prefix + "module." + field].requires_grad
    assert not any(k.endswith(("running_mean", "running_var", "num_batches_tracked"))
                   for k in after)
    names = [id(p) for _, p in model.named_parameters(remove_duplicate=False)]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("training", [True, False])
def test_finite_forward_and_backward(model, inputs, training):
    apply_addon(model, {})
    model.train(training)
    output = model(**inputs)
    assert output.shape == (2, 1) and torch.isfinite(output).all()
    output.square().mean().backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    for block in model.local_blocks:
        for norm in (block.norm1, block.norm3):
            assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()
            assert norm.bias.grad is not None and torch.isfinite(norm.bias.grad).all()


@pytest.mark.parametrize("training", [True, False])
def test_graph_isolation_with_dropout_disabled(model, inputs, training):
    apply_addon(model, {})
    # This isolates normalization/batching, not dropout-mask or RNG equivalence.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0
    for block in model.local_blocks:
        block.dropout = 0.0
    model.train(training)
    alone = {
        "x": inputs["x"][:3], "edge_index": inputs["edge_index"][:, :4],
        "edge_attr": inputs["edge_attr"][:4], "batch": inputs["batch"][:3],
        "random_walk_pe": inputs["random_walk_pe"][:3],
    }
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs)[:1], model(**alone), rtol=1e-5, atol=1e-6)


def test_strict_state_roundtrip(model, inputs):
    base = {key: value.clone() for key, value in model.state_dict().items()}
    apply_addon(model, {})
    with torch.no_grad():
        model.local_blocks[0].norm1.weight.fill_(0.43)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = make_encoder("neural_atom_k1")
    restored.load_state_dict(base, strict=True)
    apply_addon(restored, {})
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs), restored(**inputs), rtol=0, atol=0)
    assert restored._molgap_substrate.bindings == model._molgap_substrate.bindings


def test_readout_identity_binding_and_unique_parameter_ownership(model, inputs):
    head = model.head
    apply_addon(model, {})
    owner = model._molgap_substrate
    assert type(owner) is ModelSubstrate and owner.model is model
    assert len(owner.bindings) == 1
    entry = owner.bindings[0]
    assert entry.name == "k1_post_layernorm"
    assert entry.binding.family == "neural_atom_k1"
    assert entry.binding.slot is SlotName.PRE_READOUT and entry.binding.layer is None
    assert model.head is head
    marker = model._molgap_addons[entry.name]
    assert list(marker.parameters()) == [] and list(marker.buffers()) == []
    assert list(owner.named_addon_parameters()) == []
    payload = ReadoutInput(torch.ones(2, 192))
    assert marker(payload) is payload
    calls = []
    handle = marker.register_forward_hook(lambda module, args, output: calls.append((args[0], output)))
    owner.seal()
    try:
        model.eval()
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    assert len(calls) == 1 and calls[0][0] is calls[0][1]


@pytest.mark.parametrize("config", [None, [], {"eps": 1e-5}, {"width": 192}])
def test_reject_nonempty_or_untyped_config_without_mutation(model, config):
    before = dict(model.named_modules())
    with pytest.raises(ValueError, match="config"):
        apply_addon(model, config)
    assert dict(model.named_modules()) == before
    assert not hasattr(model, "_molgap_substrate")


@pytest.mark.parametrize("case", ["depth", "k4", "norm", "untracked", "affine",
                                 "extra_bn", "shared", "width", "placement", "epsilon",
                                 "conv", "mlp", "readout"])
def test_reject_unexpected_topology_atomically(model, case):
    if case == "depth":
        del model.local_blocks[-1]
    elif case == "k4":
        model.neural_atom_mixers["3"].active_slots = 4
    elif case == "norm":
        model.local_blocks[-1].norm3 = nn.LayerNorm(192)
    elif case == "untracked":
        model.local_blocks[-1].norm3.module.track_running_stats = False
    elif case == "affine":
        model.local_blocks[-1].norm3.module.affine = False
    elif case == "extra_bn":
        model.extra_bn = nn.BatchNorm1d(192)
    elif case == "shared":
        model.local_blocks[-1].norm3 = model.local_blocks[0].norm1
    elif case == "width":
        model.local_blocks[-1].norm3.module = nn.BatchNorm1d(64)
    elif case == "placement":
        model.local_blocks[-1].norm_with_batch = True
    elif case == "conv":
        model.local_blocks[-1].conv = nn.Identity()
    elif case == "mlp":
        model.local_blocks[-1].mlp = nn.Sequential(nn.Linear(192, 192))
    elif case == "readout":
        model.head = nn.Identity()
    else:
        model.local_blocks[-1].norm3.module.eps = 1e-3
    before = dict(model.named_modules())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        apply_addon(model, {})
    assert dict(model.named_modules()) == before
    assert torch.equal(rng, torch.get_rng_state())
    assert not hasattr(model, "_molgap_substrate")


def test_reapplication_rejected(model):
    apply_addon(model, {})
    before = dict(model.named_modules())
    with pytest.raises(ValueError):
        apply_addon(model, {})
    assert dict(model.named_modules()) == before


@pytest.mark.parametrize("sealed", [True, False])
def test_existing_empty_substrate(model, sealed):
    owner = ModelSubstrate(model, "neural_atom_k1")
    if sealed:
        owner.seal()
        with pytest.raises(ValueError, match="substrate"):
            apply_addon(model, {})
        assert isinstance(model.local_blocks[0].norm1.module, nn.BatchNorm1d)
    else:
        apply_addon(model, {})
        assert model._molgap_substrate is owner and len(owner.bindings) == 1
