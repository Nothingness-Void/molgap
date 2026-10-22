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


def test_message_droppath_grouped_mask_identity_and_missing_layout():
    from molgap.k1_message_droppath import MessageDropPath
    model = TinyK1()
    graph_index = torch.arange(64).repeat_interleave(3)
    payload = K1AggregationOutput(torch.ones(192, 4, requires_grad=True), None, graph_index)
    with torch.random.fork_rng(devices=[]):
        addon = MessageDropPath(model, .5)
        rng = torch.get_rng_state()
        with pytest.raises(ValueError, match="accelerated graph count"):
            addon(payload)
        assert torch.equal(rng, torch.get_rng_state())
        addon.eval()
        assert addon(payload) is payload
        assert torch.equal(rng, torch.get_rng_state())
        assert MessageDropPath(model, 0)(payload) is payload
        assert torch.equal(rng, torch.get_rng_state())
        model._execution_graph_count = 64
        addon.train()
        torch.manual_seed(42)
        expected = torch.empty(64, 1).bernoulli_(.5).div_(.5)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(42)
        actual = addon(payload)
        assert torch.equal(actual.aggregated, expected[graph_index].expand(-1, 4))
        assert torch.equal(expected_rng, torch.get_rng_state())
        assert set(actual.aggregated.unique().tolist()) == {0., 2.}
        assert actual.graph_index is graph_index and actual.target_index is payload.target_index
        actual.aggregated.sum().backward()
        assert torch.equal(payload.aggregated.grad, expected[graph_index].expand(-1, 4))
        assert not list(addon.parameters()) and not list(addon.buffers())
        assert not list(addon.children())


@pytest.mark.parametrize("probability", [True, False, None, "0.05", -.01, 1, float("nan"), float("inf")])
def test_message_droppath_probability_contract_fails_closed(probability):
    from pathlib import Path
    from molgap.experiment_execution import build_addon_declaration
    from molgap.k1_message_droppath import MessageDropPath
    with pytest.raises(ValueError, match="0 <= p < 1"):
        build_addon_declaration(("neural_atom_k1", "3"), "k1_message_droppath",
            repo_root=Path(__file__).resolve().parents[1], config={"probability": probability})
    with pytest.raises(ValueError, match="0 <= p < 1"):
        MessageDropPath(TinyK1(), probability)


def test_message_droppath_real_k1_accelerated_optimizer_rng_roundtrip(tmp_path):
    from copy import deepcopy
    from pathlib import Path
    from torch_geometric.data import Data
    from molgap.experiment_execution import build_addon_declaration
    from molgap.fast_sandbox import _cpu_scope
    from molgap.family_training_data import DatasetSelection, TrainingData
    from molgap.family_training_kernel import FamilyKernel
    from molgap.family_training_loaders import FamilyLoaders
    from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
    from molgap.gptrans_execution_tuning import ExecutionTuning
    from molgap.k1_execution_backend import K1ExecutionConfig, execution_context
    from molgap.k1_execution_layout import cpu_layout_context
    from molgap.qm9_neural_atom import make_encoder
    from molgap.v4_runtime import state_dict_sha256

    def same(actual, expected):
        if torch.is_tensor(actual):
            assert torch.equal(actual, expected)
        elif isinstance(actual, dict):
            assert actual.keys() == expected.keys()
            for key in actual:
                same(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            assert len(actual) == len(expected)
            for left, right in zip(actual, expected):
                same(left, right)
        else:
            assert actual == expected

    kernels, pool = [], None
    try:
        with _cpu_scope(42, 1):
            base = make_encoder("neural_atom_k1")
            initial = deepcopy(base.state_dict())
            path = tmp_path / "synthetic-initial.pt"
            torch.save(initial, path)
            addon = build_addon_declaration(("neural_atom_k1", "3"), "k1_message_droppath",
                repo_root=Path(__file__).resolve().parents[1], config={"probability": .05})
            selection = DatasetSelection("synthetic.json", "2" * 64, False)
            recipe = FamilyRecipe("neural_atom_k1", "composed", 42, state_dict_sha256(initial),
                selection, 128, 0, 2, .001, .01, ScheduleSpec("cosine_step", 2),
                "final-live", "torch-pass-randperm", 1, target_mean_eV=0., target_std_eV=1.,
                k1_execution=K1ExecutionConfig(), addons=(addon,),
                torch_version=str(torch.__version__).split("+", 1)[0])
            assert FamilyRecipe.from_dict(recipe.to_dict()) == recipe
            graphs = [Data(x=torch.zeros(2, 9, dtype=torch.long),
                edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.zeros(2, 3, dtype=torch.long),
                random_walk_pe=torch.zeros(2, 16), y=torch.tensor([float(i % 3)]),
                source_idx=torch.tensor([i])) for i in range(128)]
            data = TrainingData(graphs, None, selection.manifest_sha256, 128, 0, 0, None, 0., 1.)
            pool = FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0))
            batch = next(iter(pool.train(0, 0)))
            kernel = FamilyKernel(recipe, path, device="cpu")
            kernels.append(kernel)
            same(kernel.model.state_dict(), initial)
            assert not list(kernel.model._molgap_substrate.named_addon_parameters())
            assert [entry.binding.layer for entry in kernel.model._molgap_substrate.bindings] == list(range(1, 10))
            base.eval()
            kernel.model.eval()
            rng = torch.get_rng_state()
            with torch.no_grad(), cpu_layout_context(base, batch, "cpu"), execution_context(
                    base, batch, "cpu", recipe.k1_execution):
                expected = base(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.random_walk_pe).view(-1)
            with torch.no_grad():
                assert torch.equal(kernel.forward(batch), expected)
            assert torch.equal(rng, torch.get_rng_state())
            calls = []
            def observed(module, args, output):
                assert kernel.model._execution_graph_count == 128
                assert kernel.model._k1_execution_active
                calls.append(module)
            for module in kernel.model._molgap_addons.values():
                module.register_forward_hook(observed)
            kernel.model.train()
            loss, _, rows = kernel.step(batch, check_finite=True)
            assert torch.isfinite(loss) and rows == 128
            assert calls == list(kernel.model._molgap_addons.values())
            assert all(any(p.grad is not None and torch.count_nonzero(p.grad) for p in block.conv.parameters())
                       for block in kernel.model.local_blocks)
            assert any(not torch.equal(initial[key], value) for key, value in kernel.model.state_dict().items())
            assert not hasattr(kernel.model, "_execution_graph_count")
            checkpoint = BytesIO()
            torch.save({"model": kernel.model.state_dict(), "optimizer": kernel.optimizer.state_dict(),
                "rng": torch.get_rng_state()}, checkpoint)
            kernel.step(batch, check_finite=True)
            expected_model = deepcopy(kernel.model.state_dict())
            expected_optimizer = deepcopy(kernel.optimizer.state_dict())
            expected_rng = torch.get_rng_state()
            restored = FamilyKernel(recipe, path, device="cpu")
            kernels.append(restored)
            checkpoint.seek(0)
            saved = torch.load(checkpoint, weights_only=True)
            restored.model.load_state_dict(saved["model"], strict=True)
            restored.optimizer.load_state_dict(saved["optimizer"])
            restored.model.train()
            torch.set_rng_state(saved["rng"])
            restored.step(batch, check_finite=True)
            same(restored.model.state_dict(), expected_model)
            same(restored.optimizer.state_dict(), expected_optimizer)
            assert torch.equal(torch.get_rng_state(), expected_rng)
            assert not hasattr(restored.model, "_execution_graph_count")
    finally:
        for kernel in kernels:
            kernel.close()
        if pool is not None:
            pool.close()
