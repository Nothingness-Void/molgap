"""Desktop a6b7f2cc K1 mask reuse, with tiny synthetic CPU tensors only."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from molgap import k1_execution_layout as owner
from molgap.model_substrate import SlotBinding, SlotName, bind_substrate
from molgap.qm9_neural_atom import _NeuralAtomMixerFactory


@pytest.fixture(autouse=True)
def restore_cpu_runtime():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def synthetic_batch(nodes=(1, 2, 4)):
    return Batch.from_data_list([Data(x=torch.zeros(count, 9, dtype=torch.long),
        edge_index=torch.empty(2, 0, dtype=torch.long), edge_attr=torch.empty(0, 3, dtype=torch.long))
        for count in nodes])


class RebuiltMaskLayout:
    """Previous mask expression, retaining identical CPU-derived packing."""
    def __init__(self, layout):
        self.layout, self.mask_builds = layout, 0

    def __getattr__(self, name):
        return getattr(self.layout, name)

    @property
    def key_padding_mask(self):
        self.mask_builds += 1
        return ~self.layout.valid.unsqueeze(1)


class TinyMixers(torch.nn.Module):
    """Actual K1 mixer code with a small multi-layer host and normal readout ABI."""
    def __init__(self):
        super().__init__()
        self.pooling = "mean"
        self.neural_atom_mixers = torch.nn.ModuleDict({str(i): _NeuralAtomMixerFactory.make(
            hidden_channels=8, latent_channels=8, num_heads=2, max_slots=4,
            active_slots=1, dropout=.2) for i in range(3)})
        for mixer in self.neural_atom_mixers.values():
            torch.nn.init.normal_(mixer.return_projection.weight, std=.1)
        self.head = torch.nn.Linear(8, 1)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None, random_walk_pe=None):
        for mixer in self.neural_atom_mixers.values():
            x = mixer(x, batch)
        count = getattr(self, "_execution_graph_count", None)
        return self.head(global_mean_pool(x, batch, size=count)).reshape(-1)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def assert_clean(model):
    assert not hasattr(model, "_execution_graph_count")
    assert all(not hasattr(mixer, "_execution_dense_layout")
               for mixer in model.neural_atom_mixers.values())


def test_padding_mask_is_private_once_per_layout_and_not_a_state_tensor():
    batch = synthetic_batch()
    layout = owner.layout_from_cpu_batch(batch, "cpu")
    mask = layout.key_padding_mask
    assert torch.equal(mask, ~layout.valid.unsqueeze(1))
    assert mask.shape == (3, 1, 4) and mask.dtype is torch.bool
    assert mask is layout.key_padding_mask
    assert owner.layout_from_cpu_batch(batch, "cpu").key_padding_mask is not mask
    model = TinyMixers()
    keys = set(model.state_dict())
    rng = torch.get_rng_state().clone()
    with owner.cpu_layout_context(model, batch, "cpu"):
        shared = [mixer._execution_dense_layout for mixer in model.neural_atom_mixers.values()]
        assert all(item is shared[0] and item.key_padding_mask is shared[0].key_padding_mask for item in shared)
        assert set(model.state_dict()) == keys
    assert_clean(model)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("training", [False, True])
def test_mask_reuse_preserves_two_forward_outputs_gradients_optimizer_and_rng(training, monkeypatch):
    torch.manual_seed(123)
    model = TinyMixers()
    initial = deepcopy(model.state_dict())
    batch = synthetic_batch()
    input_state = deepcopy(batch.to_dict())
    hidden = torch.randn(7, 8)
    rng = torch.get_rng_state().clone()
    native_layout = owner.layout_from_cpu_batch
    outputs, layouts = [], []
    for cached in (False, True):
        model.load_state_dict(initial, strict=True)
        model.train(training)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        torch.set_rng_state(rng)
        def prepare(cpu_batch, device):
            layout = native_layout(cpu_batch, device)
            if not cached:
                layout = RebuiltMaskLayout(layout)
            layouts.append(layout)
            return layout
        with monkeypatch.context() as scoped:
            scoped.setattr(owner, "layout_from_cpu_batch", prepare)
            updates = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                value = hidden.clone().requires_grad_()
                with owner.cpu_layout_context(model, batch, "cpu"):
                    first = model(value, batch=batch.batch)
                    second = model(value, batch=batch.batch)
                    ((first.square().mean() + second.square().mean()) / 2).backward()
                assert_clean(model)
                gradients = {name: parameter.grad.clone() if parameter.grad is not None else None
                             for name, parameter in model.named_parameters()}
                optimizer.step()
                updates.append(deepcopy((first.detach(), second.detach(), value.grad,
                    gradients, model.state_dict(), optimizer.state_dict(), torch.get_rng_state())))
            outputs.append(updates)
    assert layouts[0].mask_builds == layouts[1].mask_builds == 6
    equal(*outputs)
    equal(batch.to_dict(), input_state)


@pytest.mark.parametrize("fault", ["body", "nested"])
def test_layout_cleanup_after_exception_or_nested_rejection_and_no_cross_batch_cache(fault):
    model = TinyMixers()
    batch = synthetic_batch()
    with pytest.raises((RuntimeError, ValueError)):
        with owner.cpu_layout_context(model, batch, "cpu"):
            first = next(iter(model.neural_atom_mixers.values()))._execution_dense_layout
            if fault == "body":
                raise RuntimeError("fixture")
            with owner.cpu_layout_context(model, batch, "cpu"):
                raise AssertionError("Nested scope must be rejected")
    assert_clean(model)
    other = synthetic_batch((3, 1))
    with owner.cpu_layout_context(model, other, "cpu"):
        fresh = next(iter(model.neural_atom_mixers.values()))._execution_dense_layout
        assert fresh is not first and fresh.key_padding_mask is not first.key_padding_mask
        assert fresh.key_padding_mask.shape == (2, 1, 3)
    assert_clean(model)


def test_default_uncached_mixer_fallback_remains_identical_to_cached_layout():
    torch.manual_seed(42)
    model, batch = TinyMixers().eval(), synthetic_batch()
    hidden = torch.randn(7, 8)
    with torch.no_grad():
        previous = model(hidden, batch=batch.batch)
        with owner.cpu_layout_context(model, batch, "cpu"):
            actual = model(hidden, batch=batch.batch)
    assert torch.equal(previous, actual)
    assert_clean(model)


def test_standard_readout_addon_remains_connected_and_layout_is_not_serialized():
    class Shift(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shift = torch.nn.Parameter(torch.tensor(.5))
        def forward(self, payload):
            return replace(payload, embedding=payload.embedding + self.shift)
    model, batch, addon = TinyMixers(), synthetic_batch(), Shift()
    substrate = bind_substrate(model, "neural_atom_k1", [
        ("shift", addon, SlotBinding("neural_atom_k1", SlotName.PRE_READOUT))])
    keys = set(model.state_dict())
    hidden = torch.randn(7, 8, requires_grad=True)
    with owner.cpu_layout_context(model, batch, "cpu"):
        model(hidden, batch=batch.batch).sum().backward()
    assert addon.shift.grad is not None and addon.shift.grad.abs() > 0
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    assert set(model.state_dict()) == keys
    assert not any("layout" in key or "padding_mask" in key for key in keys)
    assert not substrate._active and substrate._graph_index is None
    assert_clean(model)
