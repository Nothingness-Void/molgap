"""Synthetic CPU-only execution regressions, not native speed/MAE evidence."""
from copy import deepcopy
from io import BytesIO

import pytest
import torch
from torch_geometric.data import Batch

from modern2d_fixtures import graph
from molgap.modern2d_execution import PreparedTopology, prepare_topology, prepared_forward
from molgap.modern2d_inputs import validated_topology_views
from molgap.ppgt_author import PPGTAuthor
from molgap.ppgt_author_features import batched_positional_inputs, positional_inputs
from test_ppgt_author_fidelity import small_config, fixed_weights


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample():
    asymmetric = graph(4, 11, edges=torch.tensor([[0, 0, 2], [1, 1, 3]]))
    isolate = graph(4, 12, edges=torch.empty((2, 0), dtype=torch.long))
    parallel = graph(4, 13)
    parallel.edge_index = torch.cat((parallel.edge_index, parallel.edge_index[:, [0, 3]]), 1)
    parallel.edge_attr = torch.zeros(parallel.edge_index.shape[1], 3, dtype=torch.long)
    return Batch.from_data_list([asymmetric, isolate, parallel, graph(3, 14)])


@pytest.mark.parametrize("sym_norm", [True, False])
@pytest.mark.parametrize("steps", [1, 3, 8])
def test_batched_walks_preserve_parallel_asymmetric_isolate_channels(sym_norm, steps):
    views = validated_topology_views(sample(), packed=True)[:3]
    got = batched_positional_inputs(views, steps, sym_norm=sym_norm)
    expected = [positional_inputs(g, steps, sym_norm=sym_norm) for g in views]
    for value, elements in zip(got, zip(*expected)):
        torch.testing.assert_close(value, torch.stack(elements), atol=3e-7, rtol=2e-6)
        assert value.dtype == torch.float32 and not value.requires_grad
    assert not got[0][1, :, :, 1:].any()


def test_cpu_plan_retains_directed_order_owns_snapshot_and_excludes_labels():
    batch = sample()
    indices = torch.arange(batch.edge_index.shape[1]).flip(0)
    batch.edge_index, batch.edge_attr = batch.edge_index[:, indices], batch.edge_attr[indices]
    before = deepcopy(batch.to_dict())
    views = validated_topology_views(batch, packed=True)
    plan = prepare_topology(batch, device="cpu")
    for expected, actual in zip(views, plan.device_views):
        assert actual.source_idx == expected.source_idx
        assert not hasattr(actual, "y")
        for key in ("x", "edge_index", "edge_attr"):
            assert torch.equal(getattr(actual, key), getattr(expected, key))
    for key, value in before.items():
        assert torch.equal(batch[key], value)
    saved = plan.cpu_views[0].x.clone()
    batch.x.zero_()
    assert torch.equal(plan.cpu_views[0].x, saved)


@pytest.mark.parametrize("change", ["role", "geometry", "cross_graph", "category", "ptr", "source", "cap"])
def test_preparation_is_fail_closed_before_any_device_transfer(change, monkeypatch):
    batch = sample()
    options = {}
    if change == "role": batch.role_name = ["test_dev"] * batch.num_graphs
    elif change == "geometry": batch.pos = torch.zeros(batch.num_nodes, 3)
    elif change == "cross_graph": batch.edge_index[1, 0] = len(batch.x)-1
    elif change == "category": batch.x[0, 0] = 999
    elif change == "ptr": batch.ptr[1] += 1
    elif change == "source": batch.source_idx[1] = batch.source_idx[0]
    else: options["max_nodes"] = 2
    # CUDA is never initialized by this negative-input check.
    with pytest.raises(ValueError):
        prepare_topology(batch, device="cuda", **options)


def test_prepared_layout_rejects_fake_issuer_caps_and_device():
    model = PPGTAuthor(small_config())
    plan = prepare_topology(sample(), device="cpu", max_nodes=model.config.max_nodes,
                            max_edges=model.config.max_edges)
    forged = PreparedTopology(plan.cpu_views, plan.device_views, plan.max_nodes, plan.max_edges, object())
    with pytest.raises(ValueError): model.forward_prepared(forged)
    with pytest.raises(ValueError): plan.validate_for(small_config(max_nodes=7), torch.device("cpu"))
    with pytest.raises(ValueError): plan.validate_for(model.config, torch.device("cuda"))


@pytest.mark.parametrize("change", ["atom", "edge", "bond", "cpu_view", "native_view", "source"])
def test_prepared_layout_rejects_mutated_or_replaced_topology(change):
    from dataclasses import replace
    model = PPGTAuthor(small_config())
    plan = prepare_topology(sample(), device="cpu", max_nodes=model.config.max_nodes,
                            max_edges=model.config.max_edges)
    graph = plan.cpu_views[0]
    if change == "atom": graph.x[0, 0] = 999
    elif change == "edge": graph.edge_index[0, 0] = 999
    elif change == "bond": graph.edge_attr[0, 0] = 999
    elif change == "cpu_view": plan = replace(plan, cpu_views=(replace(graph, x=graph.x.clone()), *plan.cpu_views[1:]))
    elif change == "native_view": plan = replace(plan, device_views=(replace(plan.device_views[0], x=graph.x.clone()), *plan.device_views[1:]))
    else: plan = replace(plan, cpu_views=(replace(graph, source_idx=-1), *plan.cpu_views[1:]))
    with pytest.raises(ValueError, match="mutated"):
        model.forward_prepared(plan)


def test_prepared_layout_is_version_tracked_under_outer_inference_mode():
    model = PPGTAuthor(small_config()).eval()
    batch = sample()
    with torch.inference_mode():
        value = prepared_forward(model, batch, device="cpu")
    assert value.shape == (batch.num_graphs,) and torch.isfinite(value).all()


def test_ppgt_candidate_matches_bucketed_outputs_all_gradients_and_adam_updates():
    torch.manual_seed(42)
    reference = PPGTAuthor(small_config())
    fixed_weights(reference)
    model = deepcopy(reference)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=3e-4, weight_decay=1e-5)
    batch = sample()
    initial_keys = tuple(model.state_dict())
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        ref_opt.zero_grad(set_to_none=True)
        rng = torch.get_rng_state().clone()
        got = prepared_forward(model, batch, device="cpu")
        assert torch.equal(torch.get_rng_state(), rng)
        expected = reference.forward_bucketed(batch)
        torch.testing.assert_close(got, expected, atol=3e-6, rtol=2e-5)
        weights = torch.tensor([.7, -.4, .2, .5])
        (got * weights).sum().backward()
        (expected * weights).sum().backward()
        for (name, value), (_, other) in zip(model.named_parameters(), reference.named_parameters()):
            assert value.grad is not None and other.grad is not None
            torch.testing.assert_close(value.grad, other.grad, atol=1e-5, rtol=8e-5, msg=name)
        opt.step()
        ref_opt.step()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], atol=3e-6, rtol=2e-5, msg=name)
    assert tuple(model.state_dict()) == initial_keys
    buffer = BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = deepcopy(model)
    restored.load_state_dict(saved["model"])
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    restored_opt.load_state_dict(saved["optimizer"])
    assert restored_opt.state_dict()["param_groups"] == opt.state_dict()["param_groups"]
    torch.testing.assert_close(prepared_forward(restored, batch, device="cpu"),
        prepared_forward(model, batch, device="cpu"), atol=0, rtol=0)


def test_ppgt_stochastic_preset_keeps_graph_order_rng_policy():
    model = PPGTAuthor(small_config(attention_dropout=.1, stochastic_depth=.2)).train()
    batch = sample()
    torch.manual_seed(24)
    expected = model.forward_reference(batch)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(24)
    got = prepared_forward(model, batch, device="cpu")
    assert torch.equal(got, expected) and torch.equal(rng, torch.get_rng_state())


def test_prepared_dispatch_preserves_addon_hooks_and_restores_context_on_failure(monkeypatch):
    from molgap.substrate_scale_addon import apply_addon
    from molgap.modern2d_execution import current_prepared
    model = PPGTAuthor(small_config())
    fixed_weights(model)
    apply_addon(model, dict(family="ppgt_small", slot="pre_readout", name="scale"))
    model._molgap_substrate.seal()
    with torch.no_grad():
        model._molgap_addons["scale"].gain.fill_(1.25)
    model.forward = model.forward_bucketed
    batch = sample()
    expected = model(batch)
    got = prepared_forward(model, batch, device="cpu")
    torch.testing.assert_close(got, expected, atol=3e-6, rtol=2e-5)
    got.sum().backward()
    assert model._molgap_addons["scale"].gain.grad is not None
    assert current_prepared(model, batch) is None
    assert not model._molgap_substrate._active
    with monkeypatch.context() as patch:
        def failed(*args):
            raise RuntimeError("synthetic forward failure")
        patch.setattr(model, "forward_prepared", failed)
        with pytest.raises(RuntimeError, match="synthetic forward failure"):
            prepared_forward(model, batch, device="cpu")
    assert current_prepared(model, batch) is None
    assert not model._molgap_substrate._active and not model._molgap_substrate._calls
