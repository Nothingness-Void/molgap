import copy

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

import molgap.k1_execution_projection as projection
from molgap.gps import _PersistentEdgeUpdate
from molgap.k1_execution_layout import DenseLayout, cpu_layout_context
from molgap.k1_execution_projection import (
    project_before_gather_context,
    project_before_pack_context,
)
from molgap.qm9_neural_atom import _NeuralAtomMixerFactory


@pytest.fixture(autouse=True)
def cheap_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    yield
    torch.set_num_threads(threads)


def batch_fixture():
    return Batch.from_data_list([
        Data(x=torch.zeros(n, 9), edge_index=torch.tensor(
            [[0, 0, n - 1, 1], [1, n - 1, 0, 0]], dtype=torch.long))
        for n in (2, 5, 3)
    ])


def wrapper():
    model = nn.Module()
    model.pooling = "mean"
    model.edge_updates = nn.ModuleList([_PersistentEdgeUpdate(12, 8, 0.2)])
    model.neural_atom_mixers = nn.ModuleDict({
        str(layer): _NeuralAtomMixerFactory.make(12, 8, 2, 4, 1, 0.2)
        for layer in (3, 6, 9)
    })
    for mixer in model.neural_atom_mixers.values():
        nn.init.normal_(mixer.return_projection.weight, std=0.1)
    return model.train()


def assert_close(a, b):
    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)


def assert_gradients(a, b):
    for (name, p), (other_name, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other_name
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert_close(p.grad, q.grad)


def assert_clean(model):
    assert not hasattr(model, "_execution_graph_count")
    for module in model.modules():
        for flag in ("_execution_project_before_gather", "_execution_project_before_pack",
                     "_execution_dense_layout"):
            assert not hasattr(module, flag)
        assert not module._forward_pre_hooks


@pytest.mark.parametrize("empty", [False, True])
def test_edge_output_inputs_weights_rng_and_default(empty):
    model = wrapper()
    candidate = copy.deepcopy(model)
    batch = batch_fixture()
    if empty:
        batch.edge_index = torch.empty((2, 0), dtype=torch.long)
    h = torch.randn(10, 12, requires_grad=True)
    state = torch.randn(batch.edge_index.shape[1], 8, requires_grad=True)
    other_h = h.detach().clone().requires_grad_(True)
    other_state = state.detach().clone().requires_grad_(True)
    edge = model.edge_updates[0]
    saved = copy.deepcopy(candidate.state_dict())
    rng = torch.get_rng_state()
    a = edge(h, batch.edge_index, state)
    a.square().sum().backward()
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    with project_before_gather_context(candidate, batch, "cpu"):
        b = candidate.edge_updates[0](other_h, batch.edge_index, other_state)
        b.square().sum().backward()
    assert_close(a, b)
    assert_close(h.grad, other_h.grad)
    assert_close(state.grad, other_state.grad)
    assert_gradients(model, candidate)
    assert torch.equal(after, torch.get_rng_state())
    assert_clean(candidate)
    assert saved.keys() == candidate.state_dict().keys()
    for name, value in saved.items():
        assert torch.equal(value, candidate.state_dict()[name])
    torch.set_rng_state(rng)
    assert torch.equal(a, candidate.edge_updates[0](h, batch.edge_index, state))


def test_three_mixers_assignments_gradients_rng_and_default():
    model = wrapper()
    candidate = copy.deepcopy(model)
    batch = batch_fixture()
    hidden = torch.randn(10, 12, requires_grad=True)
    other = hidden.detach().clone().requires_grad_(True)
    saved = copy.deepcopy(candidate.state_dict())

    def run(m, h):
        diagnostics = []
        for mixer in m.neural_atom_mixers.values():
            update, diagnostic = mixer.compute_update(h, batch.batch)
            diagnostics.append(diagnostic)
            h = h + update
        return h, diagnostics

    rng = torch.get_rng_state()
    a, da = run(model, hidden)
    a.square().mean().backward()
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    shapes = []
    handles = [m.node_key.register_forward_pre_hook(
        lambda module, args: shapes.append(tuple(args[0].shape)))
        for m in candidate.neural_atom_mixers.values()]
    try:
        with project_before_pack_context(candidate, batch, "cpu"):
            b, db = run(candidate, other)
            b.square().mean().backward()
            assert saved.keys() == candidate.state_dict().keys()
    finally:
        for handle in handles:
            handle.remove()
    assert shapes == [(10, 12)] * 3
    assert_close(a, b)
    assert_close(hidden.grad, other.grad)
    assert_gradients(model, candidate)
    assert torch.equal(after, torch.get_rng_state())
    for left, right in zip(da, db):
        assert torch.equal(left["valid"], right["valid"])
        assert_close(left["assignment"], right["assignment"])
        assert torch.count_nonzero(right["assignment"].masked_select(
            ~right["valid"].unsqueeze(1))) == 0
    assert_clean(candidate)
    for name, value in saved.items():
        assert torch.equal(value, candidate.state_dict()[name])
    torch.set_rng_state(rng)
    default, _ = run(candidate, hidden)
    assert torch.equal(a, default)


@pytest.mark.parametrize("context", [project_before_gather_context, project_before_pack_context])
def test_exception_nested_stale_and_layout_scope(context):
    model = wrapper()
    batch = batch_fixture()
    with pytest.raises(RuntimeError, match="body"):
        with context(model, batch, "cpu"):
            with pytest.raises(ValueError, match="Nested or stale"):
                with context(model, batch, "cpu"):
                    pass
            other = (project_before_pack_context if context is project_before_gather_context
                     else project_before_gather_context)
            with pytest.raises(ValueError, match="Nested or stale"):
                with other(model, batch, "cpu"):
                    pass
            raise RuntimeError("body")
    assert_clean(model)
    with cpu_layout_context(model, batch, "cpu"):
        with pytest.raises(ValueError, match="Nested or stale"):
            with context(model, batch, "cpu"):
                pass
    assert_clean(model)
    with pytest.raises(ValueError, match="Stale"):
        with context(model, batch, "cpu"):
            batch.batch[0] = 1
            if context is project_before_gather_context:
                model.edge_updates[0](torch.randn(10, 12), batch.edge_index, torch.randn(12, 8))
            else:
                model.neural_atom_mixers["3"].compute_update(torch.randn(10, 12), batch.batch)
    assert_clean(model)


def test_unsupported_classes_shapes_and_partial_flags_rejected():
    batch = batch_fixture()
    model = wrapper()
    model.edge_updates[0]._execution_project_before_gather = False
    with pytest.raises(ValueError, match="Nested or stale"):
        with project_before_gather_context(model, batch, "cpu"):
            pass
    assert model.edge_updates[0]._execution_project_before_gather is False
    del model.edge_updates[0]._execution_project_before_gather
    model.edge_updates[0] = nn.Identity()
    with pytest.raises(ValueError, match="Unsupported"):
        with project_before_gather_context(model, batch, "cpu"):
            pass
    assert_clean(model)


def test_masked_finite_padding_cannot_change_assignment_or_update(monkeypatch):
    model = wrapper().eval()
    batch = batch_fixture()
    hidden = torch.randn(10, 12)
    mixer = model.neural_atom_mixers["3"]
    with project_before_pack_context(model, batch, "cpu"):
        expected, diagnostic = mixer.compute_update(hidden, batch.batch)
    original = DenseLayout.pack

    def contaminated(layout, projected, membership):
        dense, valid = original(layout, projected, membership)
        dense = dense.clone()
        dense[~valid] = 10000.0
        return dense, valid

    monkeypatch.setattr(DenseLayout, "pack", contaminated)
    with project_before_pack_context(model, batch, "cpu"):
        actual, changed = mixer.compute_update(hidden, batch.batch)
    assert_close(expected, actual)
    assert torch.equal(diagnostic["valid"], changed["valid"])
    assert torch.equal(diagnostic["assignment"], changed["assignment"])
    assert_clean(model)


def test_partial_hook_setup_failure_restores_previous_modules(monkeypatch):
    model = wrapper()
    model.edge_updates.append(_PersistentEdgeUpdate(12, 8, 0.2))

    def fail(*args, **kwargs):
        raise RuntimeError("hook registration")

    monkeypatch.setattr(model.edge_updates[1], "register_forward_pre_hook", fail)
    with pytest.raises(RuntimeError, match="hook registration"):
        with project_before_gather_context(model, batch_fixture(), "cpu"):
            pass
    assert_clean(model)


def test_unsupported_slots_restore_scope():
    batch = batch_fixture()
    model = wrapper()
    model.neural_atom_mixers["6"].active_slots = 2
    with pytest.raises(ValueError, match="Unsupported"):
        with project_before_pack_context(model, batch, "cpu"):
            pass
    assert_clean(model)


def test_gather_layout_validation_is_cpu_only_and_keywords_work(monkeypatch):
    model = wrapper().eval()
    batch = batch_fixture()
    devices = []
    original = projection.layout_from_cpu_batch

    def recorded(batch, device):
        devices.append(device)
        return original(batch, device)

    def no_cuda():
        raise AssertionError("CPU scope must not discover a CUDA device")

    monkeypatch.setattr(projection, "layout_from_cpu_batch", recorded)
    monkeypatch.setattr(torch.cuda, "current_device", no_cuda)
    h, state = torch.randn(10, 12), torch.randn(12, 8)
    expected = model.edge_updates[0](h, batch.edge_index, state)
    with project_before_gather_context(model, batch, torch.device("cpu")):
        actual = model.edge_updates[0](h=h, edge_index=batch.edge_index, edge_state=state)
    assert devices == ["cpu"]
    assert_close(expected, actual)
    assert_clean(model)


def test_runtime_shapes_and_invalid_endpoints_restore_scope():
    batch = batch_fixture()
    model = wrapper()
    with pytest.raises(ValueError, match="shape"):
        with project_before_gather_context(model, batch, "cpu"):
            model.edge_updates[0](torch.randn(9, 12), batch.edge_index, torch.randn(12, 8))
    assert_clean(model)
    with pytest.raises(ValueError, match="differ"):
        with project_before_pack_context(model, batch, "cpu"):
            model.neural_atom_mixers["3"].compute_update(torch.randn(10, 12), batch.batch.clone())
    assert_clean(model)
    batch.edge_index[0, 0] = 10
    with pytest.raises(ValueError, match="endpoints"):
        with project_before_gather_context(model, batch, "cpu"):
            pass
    assert_clean(model)
