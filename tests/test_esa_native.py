"""ESA/3 synthetic CPU checks; no dataset, native qualification or training run."""
from copy import deepcopy
from dataclasses import asdict
from io import BytesIO

import pytest
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from molgap.esa_features import edge_tokens
from molgap.esa_invariant import ESAInvariant, ESAInvariantConfig, InvariantPMA
from molgap.esa_model import ESAConfig
from molgap.esa_native import (
    ATOM_ONLY_TOKEN, BOND_TOKEN, CONFIG_TYPE, ESANative, ESANativeConfig,
    NativeTokens, _attention_masks, make_model, native_tokens,
)
from molgap.modern2d_inputs import categorical_one_hot, validated_topology_views
from molgap.ogb_features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS
from modern2d_fixtures import graph


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config(**values):
    return ESANativeConfig(**{
        "hidden": 16, "heads": 2, "layer_types": ("M", "S", "P"),
        "graph_dim": 16, "mlp_hidden": 32, "seeds": 3, **values,
    })


def no_edges(nodes=1, source=0):
    return graph(nodes, source, edges=torch.empty((2, 0), dtype=torch.long))


def mixed_graph(source=0):
    # Two disconnected bonded components plus two isolates, with directed
    # duplicates retained. The final isolates have distinct atom categories.
    result = graph(6, source, edges=torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 1]]))
    result.edge_attr[:, 0] = torch.tensor([0, 0, 1, 1, 0])
    return result


def views(graphs):
    return validated_topology_views(Batch.from_data_list(graphs))


def gradients(model, output):
    values = torch.autograd.grad(output, tuple(model.parameters()))
    assert all(torch.isfinite(value).all() for value in values)
    return dict(zip(dict(model.named_parameters()), values))


def explicit_attention(attention, query, keys, allowed):
    """Independent softmax math, rather than calling the candidate's SDPA."""
    width = attention.hidden // attention.heads

    def split(value):
        return value.reshape(1, -1, attention.heads, width).transpose(1, 2)

    q = split(attention.fc_q(query))
    k = split(attention.fc_k(keys))
    v = split(attention.fc_v(keys))
    logits = q @ k.transpose(-1, -2) / width ** 0.5
    logits = logits.masked_fill(~allowed[None, None], -torch.inf)
    result = (logits.softmax(-1) @ v).transpose(1, 2).reshape(1, -1, attention.hidden)
    return result + F.mish(attention.fc_o(result))


def graphwise_cpu_oracle(model, view):
    """No padding; explicit incidence sets and independent attention math."""
    tokens = native_tokens(view)
    x = (model.token_mlp(tokens.features) + model.token_type_embedding(tokens.token_types))[None]
    initial = x
    endpoints = [{int(node) for node in pair if node >= 0} for pair in tokens.endpoints.t()]
    local = torch.tensor([[bool(a & b) for b in endpoints] for a in endpoints])
    for block in model.encoder:
        normalized = block.norm(x)
        allowed = local if block.kind == "M" else torch.ones_like(local)
        x = normalized + explicit_attention(block.attention, normalized, normalized, allowed)
        x = x + block.mlp(block.norm_mlp(x))
    x = model.pool.norm(x + initial)
    seeds = model.pool.seeds
    pooled = explicit_attention(model.pool.attention, seeds, x,
                                torch.ones(seeds.shape[1], x.shape[1], dtype=torch.bool))
    pooled = pooled + model.pool.mlp(model.pool.norm_mlp(pooled))
    return model.head(F.mish(model.graph_projection(pooled.mean(1)))).reshape(())


@pytest.mark.parametrize("nodes", [1, 4])
def test_no_edge_atom_tokens_have_no_invented_bonds_and_finite_gradient(nodes):
    torch.manual_seed(42)
    data = Batch.from_data_list([no_edges(nodes)])
    before = deepcopy(data.to_dict())
    view = validated_topology_views(data)[0]
    tokens = native_tokens(view)
    atom_dim = sum(ATOM_FEATURE_DIMS)
    assert tokens.features.shape == (nodes, 2 * atom_dim + sum(BOND_FEATURE_DIMS))
    assert torch.equal(tokens.features[:, :atom_dim], categorical_one_hot(view.x, ATOM_FEATURE_DIMS))
    assert not tokens.features[:, atom_dim:].any()
    assert tokens.token_types.eq(ATOM_ONLY_TOKEN).all()
    assert torch.equal(tokens.endpoints[0], torch.arange(nodes))
    assert tokens.endpoints[1].eq(-1).all()
    model = ESANative(small_config())
    output = model(data)
    assert output.shape == (1,) and output.dtype == torch.float32 and torch.isfinite(output).all()
    grads = gradients(model, output.sum())
    assert not grads["token_type_embedding.weight"][BOND_TOKEN].any()
    assert all(torch.equal(data[key], value) for key, value in before.items())


def test_mixed_components_keep_all_real_directed_bonds_and_isolates():
    view = views([mixed_graph()])[0]
    tokens = native_tokens(view)
    edge_count = view.edge_index.shape[1]
    assert len(tokens.features) == edge_count + 2
    assert torch.equal(tokens.features[:edge_count], edge_tokens(view))
    assert torch.equal(tokens.endpoints[:, :edge_count], view.edge_index)
    assert tokens.token_types[:edge_count].eq(BOND_TOKEN).all()
    assert tokens.token_types[edge_count:].eq(ATOM_ONLY_TOKEN).all()
    assert torch.equal(tokens.endpoints[:, edge_count:], torch.tensor([[4, 5], [-1, -1]]))
    valid = torch.ones(1, len(tokens.features), dtype=torch.bool)
    masked, global_mask = _attention_masks(tokens.endpoints.t()[None], valid)
    local = masked[0, 0]
    assert local.diagonal().all()
    assert local[0, 1] and local[0, 4] and not local[0, 2]
    assert not local[-1, :-1].any() and not local[-2, -1]
    assert global_mask.all()
    torch.manual_seed(42)
    model = ESANative(small_config()).eval()
    original = model(Batch.from_data_list([mixed_graph()]))
    changed = mixed_graph()
    changed.x[-1, 0] += 1
    assert not torch.equal(original, model(Batch.from_data_list([changed])))


def test_padding_masks_have_safe_queries_and_invisible_padding_keys():
    endpoints = torch.tensor([[[0, 1], [1, 0], [-1, -1]], [[0, -1], [1, -1], [2, -1]]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    masked, full = _attention_masks(endpoints, valid)
    for mask in (masked, full):
        assert mask.shape == (2, 1, 3, 3) and mask.any(-1).all()
        assert not mask[0, 0, :2, 2].any()
        assert torch.equal(mask[0, 0, 2], torch.tensor([False, False, True]))
    assert torch.equal(masked[1, 0], torch.eye(3, dtype=torch.bool))
    assert full[1].all()


@pytest.mark.parametrize("kind", ["atoms", "bonds", "mixed"])
def test_each_parameter_has_finite_gradient_and_adam_moments_even_one_token_kind(kind):
    torch.manual_seed(42)
    model = ESANative(small_config())
    sample = {"atoms": no_edges(3), "bonds": graph(3), "mixed": mixed_graph()}[kind]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model(Batch.from_data_list([sample])).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if kind != "mixed":
        unused = BOND_TOKEN if kind == "atoms" else ATOM_ONLY_TOKEN
        assert not model.token_type_embedding.weight.grad[unused].any()
    optimizer.step()
    for parameter in model.parameters():
        state = optimizer.state[parameter]
        assert int(state["step"]) == 1
        assert torch.isfinite(state["exp_avg"]).all() and torch.isfinite(state["exp_avg_sq"]).all()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_padded_batch_matches_independent_cpu_oracle_outputs_and_all_component_gradients():
    torch.manual_seed(73)
    model = ESANative(small_config())
    graphs = [no_edges(1, 91), mixed_graph(13), no_edges(4, 7), graph(3, 100)]
    data = Batch.from_data_list(graphs)
    output = model(data)
    coefficients = torch.tensor([0.7, -0.4, 1.3, 0.2])
    actual = gradients(model, (output * coefficients).sum())
    oracle = torch.stack([graphwise_cpu_oracle(model, view) for view in validated_topology_views(data)])
    expected = gradients(model, (oracle * coefficients).sum())
    torch.testing.assert_close(output, oracle, atol=4e-6, rtol=2e-5)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=7e-5, rtol=3e-4, msg=name)
    singles = torch.cat([model(Batch.from_data_list([g])) for g in graphs])
    torch.testing.assert_close(output, singles, atol=4e-6, rtol=2e-5)
    changed = model(Batch.from_data_list(list(reversed(graphs))))
    torch.testing.assert_close(output.flip(0), changed, atol=4e-6, rtol=2e-5)
    altered = deepcopy(graphs)
    altered[1].x[:, 0] = 77
    perturbed = model(Batch.from_data_list(altered))
    torch.testing.assert_close(output[[0, 2, 3]], perturbed[[0, 2, 3]], atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [2, 3, 4])
@pytest.mark.parametrize("stressed", [False, True])
def test_token_permutation_around_seed_count_preserves_pma_outputs_and_all_gradients(tokens, stressed):
    torch.manual_seed(101)
    model = ESANative(small_config()).eval()
    if stressed:
        with torch.no_grad():
            model.pool.seeds.mul_(30)
            for parameter in model.pool.mlp.parameters():
                parameter.mul_(3)
    isolated = native_tokens(views([no_edges(tokens)])[0])
    order = torch.arange(tokens - 1, -1, -1)
    shuffled = NativeTokens(isolated.features[order], isolated.token_types[order], isolated.endpoints[:, order])
    snapshots = []
    hook = model.pool.register_forward_hook(lambda module, args, value: snapshots.append(value.detach().clone()))
    output = model._forward_tokens([isolated])
    before = gradients(model, output.sum())
    changed = model._forward_tokens([shuffled])
    after = gradients(model, changed.sum())
    hook.remove()
    assert isinstance(model.pool, InvariantPMA)
    torch.testing.assert_close(output, changed, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(snapshots[0], snapshots[1], atol=3e-5, rtol=3e-5)
    for name in before:
        torch.testing.assert_close(before[name], after[name], atol=5e-4, rtol=2e-4, msg=name)


def test_atom_relabelling_and_directed_token_permutation_preserve_outputs_and_gradients():
    torch.manual_seed(42)
    model = ESANative(small_config()).eval()
    original = mixed_graph()
    other = original.clone()
    atom_order = torch.tensor([5, 2, 0, 4, 1, 3])
    inverse = torch.argsort(atom_order)
    bond_order = torch.tensor([3, 1, 4, 0, 2])
    other.x = original.x[atom_order]
    other.edge_index = inverse[original.edge_index[:, bond_order]]
    other.edge_attr = original.edge_attr[bond_order]
    output = model(Batch.from_data_list([original]))
    before = gradients(model, output.sum())
    changed = model(Batch.from_data_list([other]))
    after = gradients(model, changed.sum())
    torch.testing.assert_close(output, changed, atol=4e-6, rtol=2e-5)
    for name in before:
        torch.testing.assert_close(before[name], after[name], atol=7e-5, rtol=3e-4, msg=name)


@pytest.mark.parametrize("token_count", [2, 3, 4])
def test_bond_covered_v2_parameter_mapping_zero_type_embeddings_preserves_math(token_count):
    torch.manual_seed(42)
    config = small_config()
    old_config = ESAInvariantConfig(**{name: getattr(config, name) for name in ESAConfig.__dataclass_fields__})
    old = ESAInvariant(old_config).eval()
    model = ESANative(config).eval()
    result = model.load_state_dict(old.state_dict(), strict=False)
    assert result.missing_keys == ["token_type_embedding.weight"] and not result.unexpected_keys
    with torch.no_grad():
        model.token_type_embedding.weight.zero_()
    directed = graph(token_count + 1, edges=torch.stack((torch.arange(token_count), torch.arange(1, token_count + 1))))
    directed.source_idx.fill_(19)
    data = Batch.from_data_list([directed, graph(4, 3)])
    before = old(data)
    old_grads = gradients(old, before.sum())
    after = model(data)
    native_grads = gradients(model, after.sum())
    torch.testing.assert_close(before, after, atol=4e-6, rtol=2e-5)
    for name in old_grads:
        torch.testing.assert_close(old_grads[name], native_grads[name], atol=7e-5, rtol=3e-4, msg=name)
    assert not native_grads["token_type_embedding.weight"][ATOM_ONLY_TOKEN].any()


def test_serialized_state_reload_preserves_output_mode_and_gradients_without_disk():
    torch.manual_seed(42)
    model = ESANative(small_config()).train()
    data = Batch.from_data_list([no_edges(1, 7), mixed_graph(13)])
    output = model(data)
    before = gradients(model, output.sum())
    assert model.training and all(module.training for module in model.modules())
    storage = BytesIO()
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, storage)
    storage.seek(0)
    saved = torch.load(storage, map_location="cpu", weights_only=True)
    restored = make_model({"model_config": saved["config"]}).train()
    restored.load_state_dict(saved["state_dict"], strict=True)
    changed = restored(data)
    after = gradients(restored, changed.sum())
    torch.testing.assert_close(output, changed, atol=0, rtol=0)
    for name in before:
        torch.testing.assert_close(before[name], after[name], atol=0, rtol=0)
    restored.eval()
    with torch.no_grad():
        evaluated = restored(data)
    assert not evaluated.requires_grad and not restored.training
    torch.testing.assert_close(output, evaluated, atol=0, rtol=0)
    old_config = ESAInvariantConfig(**{
        name: getattr(model.config, name) for name in ESAConfig.__dataclass_fields__
    })
    with pytest.raises(RuntimeError):
        restored.load_state_dict(ESAInvariant(old_config).state_dict(), strict=True)


def test_factory_config_is_explicit_strict_bounded_and_does_not_mutate_input():
    values = asdict(small_config())
    values["layer_types"] = list(values["layer_types"])
    before = deepcopy(values)
    model = make_model({"model_config": values})
    assert CONFIG_TYPE is ESANativeConfig and type(model.config) is CONFIG_TYPE
    assert values == before
    assert model.config.tensor_device == "cpu_or_cuda"
    with pytest.raises(TypeError):
        make_model({"model_config": {**values, "unknown_field": 1}})
    with pytest.raises(TypeError):
        ESANative(ESAInvariantConfig())


@pytest.mark.parametrize("values", [
    {"max_nodes": 0}, {"max_nodes": 1025}, {"max_nodes": True},
    {"max_edges": 0}, {"max_edges": 4097}, {"max_edges": 1.5},
    {"include_self": False}, {"tensor_device": "auto"},
    {"isolated_atom_policy": "reject-pending-authorized-census"},
    {"readout_version": "index-residual-v1"}, {"tokenization_version": "directed-only-v1"},
])
def test_unreviewed_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        small_config(**values)


@pytest.mark.parametrize("role", ["train", "development", "train_100k", "train_500k", "common_dev"])
def test_authorized_role_strings_are_tensor_capability_not_real_dataset_qualification(role):
    data = Batch.from_data_list([no_edges(2)])
    data.role_name = role
    model = ESANative(small_config(tensor_device="cpu"))
    assert torch.isfinite(model(data)).all()
    with pytest.raises(ValueError, match="tensor_device"):
        ESANative(small_config(tensor_device="cuda"))(data)


@pytest.mark.parametrize("kind", ["unknown", "geometry", "role", "category", "source", "cross_graph"])
def test_forward_reuses_strict_topology_validator(kind):
    data = Batch.from_data_list([graph(2, 7), no_edges(2, 13)])
    if kind == "unknown":
        data.teacher = torch.ones(2)
    elif kind == "geometry":
        data.pos = torch.zeros(data.num_nodes, 3)
    elif kind == "role":
        data.role_name = "independent_audit"
    elif kind == "category":
        data.x[0, 0] = 999
    elif kind == "source":
        data.source_idx[1] = data.source_idx[0]
    else:
        data.edge_index[1, 0] = 2
    with pytest.raises(ValueError):
        ESANative(small_config())(data)


def test_caps_fail_before_token_derivation_and_do_not_truncate_members(monkeypatch):
    import molgap.esa_native as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Token construction must follow successful bounded validation")

    monkeypatch.setattr(module, "native_tokens", forbidden)
    data = Batch.from_data_list([graph(4)])
    for limits in ({"max_nodes": 3}, {"max_edges": 5}):
        with pytest.raises(ValueError, match="safety cap"):
            ESANative(small_config(**limits))(data)
    assert data.x.shape[0] == 4 and data.edge_index.shape[1] == 6


def test_labels_and_rwse_are_ignored_and_forward_does_no_file_or_network_io(monkeypatch):
    import builtins
    import socket

    torch.manual_seed(42)
    model = ESANative(small_config()).eval()
    data = Batch.from_data_list([no_edges(2, 7), mixed_graph(13)])
    expected = model(data)
    data.y.fill_(float("nan"))
    data.random_walk_pe.fill_(float("nan"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected IO or device move")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(torch.Tensor, "cuda", forbidden)
    torch.testing.assert_close(expected, model(data), atol=0, rtol=0)
