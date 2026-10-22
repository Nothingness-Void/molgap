"""Synthetic ESA execution-candidate checks; no real data/native qualification."""
from copy import deepcopy
from dataclasses import replace
from io import BytesIO

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch

from modern2d_fixtures import graph, model_config
from molgap import esa_native
from molgap.esa_native import (
    ESANative, ESANativeConfig, native_tokens, packed_native_tokens,
    prepare_native_tokens,
)
from molgap.esa_research_addons import apply_addon as residual_addon
from molgap.modern2d_execution import (
    PreparedTopology, prepare_topology, prepared_forward,
)
from molgap.modern2d_inputs import validated_topology_views
from molgap.substrate_scale_addon import apply_addon as scale_addon


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def config(**overrides):
    return ESANativeConfig(**{**model_config("esa_small"), **overrides})


def sample(kind="mixed"):
    empty = torch.empty((2, 0), dtype=torch.long)
    mixed = graph(6, 91, edges=torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 1]]))
    # Give parallel directed bonds distinct categories to detect reordering.
    mixed.edge_attr[:, 0] = torch.tensor([0, 1, 2, 3, 4])
    graphs = {
        "mixed": [mixed, graph(1, 7, edges=empty), graph(3, 13),
                  graph(4, 5, edges=empty)],
        "atoms": [graph(1, 7, edges=empty), graph(4, 5, edges=empty)],
        "bonds": [graph(3, 91), graph(2, 7)],
    }[kind]
    batch = Batch.from_data_list(graphs)
    order = torch.arange(batch.edge_index.shape[1]).flip(0)
    batch.edge_index = batch.edge_index[:, order]
    batch.edge_attr = batch.edge_attr[order]
    return batch


def plan(batch, model=None):
    cfg = model.config if model is not None else config()
    return prepare_topology(batch, device="cpu", max_nodes=cfg.max_nodes,
                            max_edges=cfg.max_edges)


def assert_bytes(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.device == expected.device
    assert torch.equal(actual.detach().contiguous().reshape(-1).view(torch.uint8),
                       expected.detach().contiguous().reshape(-1).view(torch.uint8))


def assert_tree(actual, expected):
    if torch.is_tensor(actual):
        assert_bytes(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for value, other in zip(actual, expected):
            assert_tree(value, other)
    else:
        assert actual == expected


@pytest.mark.parametrize("kind", ["mixed", "atoms", "bonds"])
def test_prepared_token_bytes_match_packed_and_graphwise_oracles(kind):
    batch = sample(kind)
    original = deepcopy(batch.to_dict())
    prepared = plan(batch)
    rng = torch.get_rng_state().clone()
    got = prepare_native_tokens(prepared.cpu_views, "cpu")
    assert_bytes(torch.get_rng_state(), rng)
    expected = packed_native_tokens(validated_topology_views(batch, packed=True))
    singles = [native_tokens(view) for view in prepared.cpu_views]
    oracle = (
        pad_sequence([value.features for value in singles], batch_first=True),
        pad_sequence([value.token_types for value in singles], batch_first=True),
        pad_sequence([value.endpoints.t() for value in singles], batch_first=True,
                     padding_value=-1),
        torch.arange(got[1].shape[1])[None] <
        torch.tensor([len(value.token_types) for value in singles])[:, None],
    )
    for actual, packed, padded in zip(got, expected, oracle):
        assert_bytes(actual, packed)
        assert_bytes(actual, padded)
        assert not actual.requires_grad and actual.grad_fn is None
    assert [view.source_idx for view in prepared.cpu_views] == [
        int(value) for value in batch.source_idx]
    assert_tree(batch.to_dict(), original)


def test_cpu_pack_finishes_before_exactly_four_padded_tensor_transfers(monkeypatch):
    prepared = plan(sample())
    original_pack = esa_native.packed_native_tokens
    original_to = torch.Tensor.to
    packed, transferred = [], []

    def pack(views):
        assert views is prepared.cpu_views
        assert all(t.device.type == "cpu" for view in views
                   for t in (view.x, view.edge_index, view.edge_attr))
        result = original_pack(views)
        packed.extend(result)
        return result

    def transfer(tensor, *args, **kwargs):
        assert len(packed) == 4, "CPU pack must finish before any transfer"
        transferred.append(tensor)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(esa_native, "packed_native_tokens", pack)
    monkeypatch.setattr(torch.Tensor, "to", transfer)
    got = prepare_native_tokens(prepared.cpu_views, "cpu")
    assert len(transferred) == len(got) == 4
    assert all(actual is expected for actual, expected in zip(transferred, packed))


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("addon", ["none", "residual", "scale", "both"])
def test_public_dispatch_exact_outputs_all_gradients_adam_state_and_rng_across_updates(
        training, addon):
    torch.manual_seed(42)
    reference = ESANative(config()).train(training)
    candidate = ESANative(config()).train(training)
    # Reconstruct hooks independently: deepcopy of Python hook closures would
    # retain the original substrate owner rather than proving addon isolation.
    for model in (reference, candidate):
        if addon in {"residual", "both"}:
            residual_addon(model, {"variant": "identity_residual"})
        if addon in {"scale", "both"}:
            scale_addon(model, dict(family="esa_small", slot="pre_readout", name="scale"))
            with torch.no_grad():
                model._molgap_addons["scale"].gain.fill_(1.25)
        if addon != "none":
            model._molgap_substrate.seal()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    batch = sample()
    prepared = plan(batch, candidate)
    invariant = tuple(t.clone() for t in prepare_native_tokens(prepared.cpu_views, "cpu"))
    names = tuple(candidate.state_dict())
    parameters = dict(candidate.named_parameters())
    modules = dict(candidate.named_modules())
    cfg = candidate.config
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-4, weight_decay=1e-5)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4, weight_decay=1e-5)
    weights = torch.tensor([0.7, -0.4, 0.2, 0.5])
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        rng = torch.get_rng_state().clone()
        expected = reference(batch)
        assert_bytes(torch.get_rng_state(), rng)
        actual = prepared_forward(candidate, batch, device="cpu")
        assert_bytes(torch.get_rng_state(), rng)
        assert_bytes(actual, expected)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        for name, value in candidate.named_parameters():
            other = dict(reference.named_parameters())[name]
            assert value.grad is not None and other.grad is not None, name
            assert torch.isfinite(value.grad).all(), name
            assert_bytes(value.grad, other.grad)
        optimizer.step()
        ref_optimizer.step()
        assert_tree(candidate.state_dict(), reference.state_dict())
        assert_tree(optimizer.state_dict(), ref_optimizer.state_dict())
        assert_bytes(torch.get_rng_state(), rng)
    assert candidate.config is cfg and tuple(candidate.state_dict()) == names
    assert all(dict(candidate.named_parameters())[name] is value for name, value in parameters.items())
    assert all(dict(candidate.named_modules())[name] is value for name, value in modules.items())
    assert all(module.training == training for module in candidate.modules())
    assert "forward" not in candidate.__dict__
    assert_tree(prepare_native_tokens(prepared.cpu_views, "cpu"), invariant)
    if addon != "none":
        assert not candidate._molgap_substrate._active
        assert not candidate._molgap_substrate._calls


def test_default_and_direct_prepared_resume_keep_strict_state_and_optimizer():
    torch.manual_seed(42)
    model = ESANative(config()).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch = sample()
    prepared = plan(batch, model)

    def step(module, opt, layout):
        opt.zero_grad(set_to_none=True)
        output = module.forward_prepared(layout)
        output.square().mean().backward()
        opt.step()
        return output.detach().clone()

    step(model, optimizer, prepared)
    storage = BytesIO()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state()), storage)
    expected = step(model, optimizer, prepared)
    endpoint = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    storage.seek(0)
    saved = torch.load(storage, map_location="cpu", weights_only=True)
    restored = ESANative(config()).train()
    restored.load_state_dict(saved["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    restored_optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    assert_bytes(step(restored, restored_optimizer, plan(batch, restored)), expected)
    assert_tree((restored.state_dict(), restored_optimizer.state_dict()), endpoint)
    assert_bytes(torch.get_rng_state(), saved["rng"])
    assert_bytes(restored.forward_prepared(plan(batch, restored)), restored(batch))


def test_prepared_dispatch_uses_cpu_views_and_ignores_labels_rwse(monkeypatch):
    model = ESANative(config()).eval()
    batch = sample()
    expected = model(batch)
    batch.y.fill_(float("nan"))
    batch.random_walk_pe.fill_(float("nan"))
    prepared = plan(batch, model)
    parameters = deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    original_prepare = esa_native.prepare_native_tokens
    calls = []

    def prepare(views, device):
        assert views is prepared.cpu_views
        assert device == model.head.weight.device
        calls.append(views)
        return original_prepare(views, device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Prepared execution must not revalidate/split device topology")

    monkeypatch.setattr(esa_native, "prepare_native_tokens", prepare)
    monkeypatch.setattr(esa_native, "validated_topology_views", forbidden)
    monkeypatch.setattr(esa_native, "native_tokens", forbidden)
    assert_bytes(model.forward_prepared(prepared), expected)
    assert len(calls) == 1
    assert_tree(model.state_dict(), parameters)
    assert_bytes(torch.get_rng_state(), rng)


@pytest.mark.parametrize("change", ["raw", "subclass", "issuer", "caps", "device", "capability"])
def test_invalid_preparation_fails_before_token_work(change, monkeypatch):
    model = ESANative(config(tensor_device="cuda" if change == "capability" else "cpu_or_cuda"))
    prepared = plan(sample(), model)
    if change == "raw":
        prepared = prepared.cpu_views
    elif change == "subclass":
        class OtherPreparation(PreparedTopology):
            pass
        prepared = OtherPreparation(prepared.cpu_views, prepared.device_views,
                                    prepared.max_nodes, prepared.max_edges, prepared._issuer)
    elif change == "issuer":
        prepared = replace(prepared, _issuer=object())
    elif change == "caps":
        prepared = replace(prepared, max_nodes=prepared.max_nodes - 1)
    elif change == "device":
        views = tuple(replace(view, x=view.x.to("meta")) for view in prepared.device_views)
        prepared = replace(prepared, device_views=views)

    def forbidden(*args, **kwargs):
        raise AssertionError("Admission must precede token preparation")

    monkeypatch.setattr(esa_native, "prepare_native_tokens", forbidden)
    with pytest.raises((TypeError, ValueError)):
        model.forward_prepared(prepared)


@pytest.mark.parametrize("change", ["empty", "raw", "noncpu", "target"])
def test_low_level_preparation_rejects_invalid_placement_before_pack(change, monkeypatch):
    views = plan(sample()).cpu_views
    device = "cpu"
    if change == "empty":
        views = ()
    elif change == "raw":
        views = sample()
    elif change == "noncpu":
        views = (replace(views[0], edge_attr=views[0].edge_attr.to("meta")),)
    else:
        device = "meta"

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid placement must fail before packing")

    monkeypatch.setattr(esa_native, "packed_native_tokens", forbidden)
    with pytest.raises((TypeError, ValueError)):
        prepare_native_tokens(views, device)


def test_public_prepared_dispatch_rejects_requested_device_mismatch_without_cuda(monkeypatch):
    model = ESANative(config())

    def forbidden(*args, **kwargs):
        raise AssertionError("Requested device mismatch must fail before token work")

    monkeypatch.setattr(esa_native, "prepare_native_tokens", forbidden)
    with pytest.raises(ValueError):
        prepared_forward(model, sample(), device="cuda:0")


@pytest.mark.parametrize("kind", ["atoms", "bonds"])
def test_single_token_kind_retains_zero_gradient_for_unused_type(kind):
    model = ESANative(config())
    batch = sample(kind)
    actual = model.forward_prepared(plan(batch, model))
    expected = model(batch)
    assert_bytes(actual, expected)
    actual.sum().backward()
    unused = 0 if kind == "atoms" else 1
    assert not model.token_type_embedding.weight.grad[unused].any()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
