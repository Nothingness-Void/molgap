"""Tiny slot ABI proofs; no retained models, real graphs or remote resources."""
from dataclasses import replace
from collections import OrderedDict
from io import BytesIO

import pytest
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from molgap.model_substrate import (
    GPTransAggregationOutput, GPTransMessageInput, K1AggregationOutput,
    K1MessageInput, ModelSubstrate, ReadoutInput, SlotBinding, SlotName,
    bind_substrate,
)


class TinyConv(nn.Module):
    """PyG-shaped additive seam without importing the real GNN package."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2))
        self._aggregate_hooks = OrderedDict()

    def register_aggregate_forward_hook(self, hook):
        handle = RemovableHandle(self._aggregate_hooks)
        self._aggregate_hooks[handle.id] = hook
        return handle

    def forward(self, x, edge_index, edge_attr=None):
        messages = x[edge_index[0]] @ self.weight
        if edge_attr is not None:
            messages = messages + edge_attr
        output = torch.zeros_like(x).index_add(0, edge_index[1], messages)
        for hook in self._aggregate_hooks.values():
            result = hook(self, ({"index": edge_index[1]},), output)
            if result is not None:
                output = result
        return output + x


class TinyK1(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.conv = TinyConv()
        self.local_blocks = nn.ModuleList([block])
        self.head = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1., 2.]]))

    def forward(self, x, edge_index, edge_attr, batch, random_walk_pe=None):
        x = self.local_blocks[0].conv(x, edge_index, edge_attr=edge_attr)
        pooled = x.new_zeros((2, 2)).index_add(0, batch, x)
        return self.head(pooled)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2))

    def forward(self, node, pair, key_padding_mask):
        return node @ self.weight, pair * 2


class TinyGPTrans(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.attention = TinyAttention()
        self.blocks = nn.ModuleList([block])
        self.readout = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.readout.weight.fill_(1.)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        node, pair, mask = x
        node, pair = self.blocks[0].attention(node, pair, mask)
        return self.readout(torch.cat((node[:, 0], pair[:, :, 0, 0]), -1))


class Shift(nn.Module):
    def __init__(self, field, shift):
        super().__init__()
        self.field = field
        self.shift = nn.Parameter(torch.tensor(float(shift)))
        self.observed = None

    def forward(self, payload):
        self.observed = payload
        return replace(payload, **{self.field: getattr(payload, self.field) + self.shift})


def k1_input():
    return (torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]),
        torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]), torch.zeros(4, 2),
        torch.tensor([0, 0, 1, 1]))


def gptrans_input():
    return (torch.ones(2, 2, 2), torch.ones(2, 1, 2, 2),
        torch.zeros(2, 1, 1, 2, dtype=torch.bool))


@pytest.mark.parametrize("family,make,inputs", [
    ("neural_atom_k1", TinyK1, k1_input), ("gptrans_t", TinyGPTrans, gptrans_input),
])
def test_unattached_owner_and_detach_preserve_base_state_and_forward(family, make, inputs):
    model = make()
    arguments = inputs() if family == "neural_atom_k1" else (inputs(),)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    expected = model(*arguments)
    owner = ModelSubstrate(model, family)
    assert torch.equal(expected, model(*arguments))
    assert not list(owner.named_addon_parameters())
    owner.attach("shift", Shift("embedding", 1.), SlotBinding(family, SlotName.PRE_READOUT))
    assert not torch.equal(expected, model(*arguments))
    assert [name for name, _ in owner.named_addon_parameters()] == ["_molgap_addons.shift.shift"]
    assert owner.detach("shift").shift.item() == 1.
    assert torch.equal(expected, model(*arguments))
    assert model.state_dict().keys() == state.keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])
    assert not owner._root_handles


@pytest.mark.parametrize("slot,field,payload", [
    (SlotName.PRE_MESSAGE, "node", K1MessageInput),
    (SlotName.POST_AGGREGATION, "aggregated", K1AggregationOutput),
    (SlotName.PRE_READOUT, "embedding", ReadoutInput),
])
def test_k1_payload_location_shape_and_connected_addon(slot, field, payload):
    model = TinyK1()
    arguments = k1_input()
    owner = ModelSubstrate(model, "neural_atom_k1")
    addon = Shift(field, .5)
    binding = SlotBinding("neural_atom_k1", slot, None if slot is SlotName.PRE_READOUT else 1)
    owner.attach("shift", addon, binding)
    actual = model(*arguments)
    assert type(addon.observed) is payload
    if slot is not SlotName.PRE_READOUT:
        assert addon.observed.graph_index is arguments[3]
    actual.square().sum().backward()
    assert addon.shift.grad is not None and addon.shift.grad > 0
    assert model.local_blocks[0].conv.weight.grad.abs().sum() > 0
    assert owner._graph_index is None and not owner._active


@pytest.mark.parametrize("slot,field,payload", [
    (SlotName.PRE_MESSAGE, "node", GPTransMessageInput),
    (SlotName.POST_AGGREGATION, "pair_update", GPTransAggregationOutput),
    (SlotName.PRE_READOUT, "embedding", ReadoutInput),
])
def test_gptrans_payload_location_and_connected_addon(slot, field, payload):
    model = TinyGPTrans()
    owner = ModelSubstrate(model, "gptrans_t")
    addon = Shift(field, .5)
    owner.attach("shift", addon,
        SlotBinding("gptrans_t", slot, None if slot is SlotName.PRE_READOUT else 1))
    inputs = gptrans_input()
    output = model(inputs)
    assert type(addon.observed) is payload
    if slot is not SlotName.PRE_READOUT:
        assert addon.observed.key_padding_mask is inputs[2]
    output.square().sum().backward()
    assert addon.shift.grad is not None and addon.shift.grad > 0
    assert model.blocks[0].attention.weight.grad.abs().sum() > 0


def test_addon_order_is_explicit_and_state_roundtrip_reconstructs_bindings():
    class Scale(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(3.))
        def forward(self, payload):
            return replace(payload, embedding=payload.embedding * self.scale)
    binding = SlotBinding("gptrans_t", SlotName.PRE_READOUT)
    def make():
        return bind_substrate(TinyGPTrans(), "gptrans_t", [
            ("shift", Shift("embedding", 2.), binding), ("scale", Scale(), binding)])
    owner = make()
    assert [entry.name for entry in owner.bindings] == ["shift", "scale"]
    assert torch.equal(owner.model(gptrans_input()), torch.tensor([[30.], [30.]]))
    buffer = BytesIO()
    torch.save(owner.model.state_dict(), buffer)
    buffer.seek(0)
    restored = make()
    restored.model.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    for key, value in owner.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[key])
    assert torch.equal(owner.model(gptrans_input()), restored.model(gptrans_input()))


def test_reject_duplicates_incompatible_bindings_and_sealed_changes():
    model = TinyGPTrans()
    owner = ModelSubstrate(model, "gptrans_t")
    addon = Shift("embedding", 1.)
    binding = SlotBinding("gptrans_t", SlotName.PRE_READOUT)
    owner.attach("shift", addon, binding)
    with pytest.raises(ValueError, match="already attached"):
        owner.attach("shift", Shift("embedding", 1.), binding)
    with pytest.raises(ValueError, match="distinct"):
        owner.attach("other", addon, binding)
    with pytest.raises(ValueError, match="family"):
        owner.attach("wrong", Shift("embedding", 1.), SlotBinding("neural_atom_k1", SlotName.PRE_READOUT))
    with pytest.raises(ValueError, match="depth"):
        owner.attach("wrong", Shift("node", 1.), SlotBinding("gptrans_t", SlotName.PRE_MESSAGE, 2))
    with pytest.raises(ValueError, match="dtype"):
        owner.attach("wrong", Shift("embedding", 1.).double(), binding)
    owner.seal()
    assert owner.sealed
    with pytest.raises(RuntimeError, match="sealed"):
        owner.detach("shift")
    with pytest.raises(RuntimeError, match="sealed"):
        owner.attach("new", Shift("embedding", 1.), binding)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    assert any(parameter is addon.shift for group in optimizer.param_groups for parameter in group["params"])


@pytest.mark.parametrize("kind,error", [("shape", ValueError), ("payload", TypeError), ("mask", ValueError)])
def test_incompatible_payload_is_rejected_and_forward_context_is_cleared(kind, error):
    class Wrong(nn.Module):
        def forward(self, payload):
            if kind == "shape":
                return replace(payload, node=payload.node[..., :1])
            if kind == "payload":
                return payload.node
            return replace(payload, key_padding_mask=payload.key_padding_mask.clone())
    owner = bind_substrate(TinyGPTrans(), "gptrans_t", [("wrong", Wrong(),
        SlotBinding("gptrans_t", SlotName.PRE_MESSAGE, 1))])
    with pytest.raises(error):
        owner.model(gptrans_input())
    assert not owner._active and owner._graph_index is None
    owner.detach("wrong")
    assert torch.isfinite(owner.model(gptrans_input())).all()


def test_binding_factory_rolls_back_partial_attachment_and_reserves_owner():
    model = TinyGPTrans()
    binding = SlotBinding("gptrans_t", SlotName.PRE_READOUT)
    with pytest.raises(ValueError, match="already attached"):
        bind_substrate(model, "gptrans_t", [("same", Shift("embedding", 1.), binding),
            ("same", Shift("embedding", 2.), binding)])
    assert not hasattr(model, "_molgap_addons") and not hasattr(model, "_molgap_substrate")
    owner = ModelSubstrate(model, "gptrans_t")
    with pytest.raises(ValueError, match="already has"):
        ModelSubstrate(model, "gptrans_t")
    assert owner.bindings == ()


def test_failed_root_hook_installation_rolls_back_every_hook(monkeypatch):
    model = TinyGPTrans()
    owner = ModelSubstrate(model, "gptrans_t")
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic registration fault")
    monkeypatch.setattr(model, "register_forward_hook", fail)
    with pytest.raises(RuntimeError, match="registration fault"):
        owner.attach("shift", Shift("embedding", 1.), SlotBinding("gptrans_t", SlotName.PRE_READOUT))
    assert not model._forward_pre_hooks and not model.readout._forward_pre_hooks
    assert not hasattr(model, "_molgap_addons") and owner.bindings == ()


def test_reentrant_forward_rejected_without_losing_outer_context():
    model = TinyGPTrans()
    owner = ModelSubstrate(model, "gptrans_t")
    class Reentrant(nn.Module):
        def forward(self, payload):
            with pytest.raises(RuntimeError, match="reentrant"):
                model(gptrans_input())
            assert owner._active
            return payload
    owner.attach("reentrant", Reentrant(), SlotBinding("gptrans_t", SlotName.PRE_READOUT))
    assert torch.isfinite(model(gptrans_input())).all()
    assert not owner._active and not owner._calls


def test_reserved_module_name_rejects_before_namespace_mutation():
    owner = ModelSubstrate(TinyGPTrans(), "gptrans_t")
    with pytest.raises(ValueError, match="namespace"):
        owner.attach("items", Shift("embedding", 1.), SlotBinding("gptrans_t", SlotName.PRE_READOUT))
    assert not hasattr(owner.model, "_molgap_addons") and owner.bindings == ()


@pytest.mark.parametrize("binding,error", [
    (("unknown", SlotName.PRE_READOUT, None), ValueError),
    (("gptrans_t", "pre_readout", None), TypeError),
    (("gptrans_t", SlotName.PRE_READOUT, 1), ValueError),
    (("gptrans_t", SlotName.PRE_MESSAGE, True), ValueError),
])
def test_slot_descriptors_fail_closed(binding, error):
    with pytest.raises(error):
        SlotBinding(*binding)


def test_k1_residual_gain_preserves_accelerated_base_and_checkpoint():
    from copy import deepcopy
    from torch_geometric.data import Batch, Data
    from molgap.k1_execution_backend import K1ExecutionConfig, execution_context
    from molgap.k1_execution_layout import cpu_layout_context
    from molgap.k1_residual_gain import apply_addon
    from molgap.qm9_neural_atom import make_encoder

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            base = make_encoder("neural_atom_k1")
            candidate = deepcopy(base)
            old_parameters = dict(candidate.named_parameters())
            old_buffers = dict(candidate.named_buffers())
            blocks = tuple(candidate.local_blocks)
            owner = ModelSubstrate(candidate, "neural_atom_k1")
            assert apply_addon(candidate, {}) is candidate
            owner.seal()
            assert owner.mutations == tuple((f"local_residual_gain_{i}",
                f"local_blocks.{i}") for i in range(9))
            parameters = dict(candidate.named_parameters())
            for name, parameter in old_parameters.items():
                assert parameters[name] is parameter
            for name, buffer in old_buffers.items():
                assert dict(candidate.named_buffers())[name] is buffer
            gains = {name: value for name, value in parameters.items()
                     if name not in old_parameters}
            assert len(gains) == 18 and sum(p.numel() for p in gains.values()) == 3456
            for gain in gains.values():
                assert gain.shape == (192,) and torch.equal(gain, torch.ones_like(gain))
            for original, wrapped in zip(blocks, candidate.local_blocks):
                for name in ("conv", "norm1", "norm3", "mlp"):
                    assert getattr(wrapped, name) is getattr(original, name)
                assert wrapped.dropout == original.dropout
                assert wrapped.norm_with_batch is original.norm_with_batch
            batch = Batch.from_data_list([Data(
                x=torch.zeros(3, 9, dtype=torch.long),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                edge_attr=torch.zeros(4, 3, dtype=torch.long),
                random_walk_pe=torch.rand(3, 16)) for _ in range(2)])

            def forward(model):
                with cpu_layout_context(model, batch, "cpu"), execution_context(
                        model, batch, "cpu", K1ExecutionConfig()):
                    return model(batch.x, batch.edge_index, batch.edge_attr,
                                 batch.batch, batch.random_walk_pe)

            for training in (False, True):
                base.train(training)
                candidate.train(training)
                rng = torch.get_rng_state()
                with torch.no_grad():
                    expected = forward(base)
                    expected_rng = torch.get_rng_state()
                    torch.set_rng_state(rng)
                    actual = forward(candidate)
                assert torch.equal(actual, expected)
                assert torch.equal(torch.get_rng_state(), expected_rng)
                for name, value in base.state_dict().items():
                    assert torch.equal(candidate.state_dict()[name], value), name
            optimizer = torch.optim.AdamW(candidate.parameters(), lr=.001)
            forward(candidate).square().sum().backward()
            optimized = [p for group in optimizer.param_groups for p in group["params"]]
            assert len(optimized) == len({id(p) for p in optimized}) == len(parameters)
            for name, gain in gains.items():
                assert any(p is gain for p in optimized)
                assert gain.grad is not None and torch.isfinite(gain.grad).all(), name
                assert torch.count_nonzero(gain.grad), name
            optimizer.step()
            checkpoint = BytesIO()
            torch.save({"model": candidate.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
            checkpoint.seek(0)
            saved = torch.load(checkpoint, weights_only=True)
            restored = apply_addon(make_encoder("neural_atom_k1"), {})
            restored._molgap_substrate.seal()
            restored.load_state_dict(saved["model"], strict=True)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
            restored_optimizer.load_state_dict(saved["optimizer"])
            candidate.eval()
            restored.eval()
            with torch.no_grad():
                assert torch.equal(forward(candidate), forward(restored))
            for name, gain in restored.named_parameters():
                if name in gains:
                    assert torch.equal(restored_optimizer.state[gain]["exp_avg"],
                                       optimizer.state[parameters[name]]["exp_avg"])
            assert not hasattr(candidate, "_k1_execution_active")
            assert not hasattr(candidate, "_execution_graph_count")
    finally:
        torch.set_num_threads(threads)
