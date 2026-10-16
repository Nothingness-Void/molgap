from copy import deepcopy

import pytest
import torch
from torch_geometric.data import Batch

from molgap.modern2d_inputs import topology_views
from molgap.modern2d_feature_binding import validate_feature_binding, BINDING_FIELDS
from molgap.ppgt_features import rrwp, positional_inputs, sinusoidal_pe, feature_cost
from molgap.esa_features import shared_endpoint_mask, edge_tokens
from modern2d_fixtures import graph, batch


@pytest.mark.parametrize("nodes", [1, 2, 3, 6])
def test_rrwp_exact_dense_powers_and_zero_degree(nodes):
    g = topology_views(Batch.from_data_list([graph(nodes)]))[0]
    actual, degree = rrwp(g, 4)
    a = torch.zeros(nodes, nodes)
    a[g.edge_index[0], g.edge_index[1]] = 1
    w = a / a.sum(1).clamp_min(1)[:, None]
    for k in range(4):
        assert torch.equal(actual[..., k], torch.linalg.matrix_power(w, k))
    assert torch.equal(actual[..., 0], torch.eye(nodes))
    assert torch.allclose(w.sum(1), (degree > 0).float())


def test_cycle_disconnected_duplicate_edges_and_relabel_covariance():
    e = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]])
    original = graph(6, edges=torch.cat((e, e.flip(0), e), 1))
    a = topology_views(Batch.from_data_list([original]))[0]
    p, d = rrwp(a, 6)
    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    inverse = torch.argsort(permutation)
    relabeled = original.clone()
    relabeled.x = relabeled.x[permutation]
    relabeled.edge_index = inverse[relabeled.edge_index]
    q, qd = rrwp(topology_views(Batch.from_data_list([relabeled]))[0], 6)
    assert torch.equal(q, p[permutation][:, permutation])
    assert torch.equal(qd, d[permutation])
    assert not p[:3, 3:].any()


def test_rrwp_conflicting_parallel_bond_rejected():
    g = graph(2)
    g.edge_index = torch.cat((g.edge_index, g.edge_index[:, :1]), 1)
    g.edge_attr = torch.cat((g.edge_attr, torch.tensor([[1, 0, 0]])), 0)
    with pytest.raises(ValueError, match="Conflicting"):
        rrwp(topology_views(Batch.from_data_list([g]))[0], 3)


def test_spe_and_size_channels_are_not_rwse():
    g = topology_views(batch())[0]
    p, extended, node = positional_inputs(g, 4)
    assert extended.shape == (3, 3, 7) and node.shape == (3, 2)
    result = sinusoidal_pe(extended, 2)
    assert result.shape == (3, 3, 35)
    assert torch.equal(result.reshape(3, 3, 7, 5)[..., 0], extended)
    assert feature_cost(3)["rrwp_bytes_fp32"] == 4 * 3 * 3 * 24


def test_esa_shared_endpoint_self_disconnected_and_directed_tokens():
    edges = torch.tensor([[0, 1, 2, 4], [1, 0, 3, 5]])
    mask = shared_endpoint_mask(edges)
    assert mask.diagonal().all() and mask[0, 1] and not mask[0, 2]
    assert not shared_endpoint_mask(edges, include_self=False).diagonal().any()
    g = topology_views(Batch.from_data_list([graph(6, edges=edges)]))[0]
    tokens = edge_tokens(g)
    assert tokens.shape == (4, 361) and not torch.equal(tokens[0], tokens[1])
    empty = topology_views(Batch.from_data_list([graph(1)]))[0]
    with pytest.raises(ValueError, match="empty-edge"):
        edge_tokens(empty)


@pytest.mark.parametrize("kind", ["geometry", "unknown", "category", "cross_graph", "source", "role"])
def test_topology_fail_closed(kind):
    data = batch()
    if kind == "geometry": data.pos = torch.zeros(data.num_nodes, 3)
    elif kind == "unknown": data.teacher = torch.ones(2)
    elif kind == "category": data.x[0, 0] = 999
    elif kind == "cross_graph": data.edge_index[1, 0] = 3
    elif kind == "source": data.source_idx[1] = data.source_idx[0]
    else: data.role_name = "independent_audit"
    with pytest.raises(ValueError): topology_views(data)


def test_binding_checks_all_identities_and_no_fake_cache_sha():
    binding = {k: "a" * 64 for k in BINDING_FIELDS}
    binding.update(role="synthetic", scope="CPU_SYNTHETIC_ONLY", derived_feature_sha=None,
                   feature_schema_version="ogb-atom9-bond3-rrwp-spe-v1")
    assert len(validate_feature_binding(binding, deepcopy(binding))) == 64
    for key in BINDING_FIELDS:
        changed = deepcopy(binding)
        changed[key] = "b" * 64
        with pytest.raises(ValueError): validate_feature_binding(changed, binding)
    for role in ("independent_audit", "official_validation", "test", "common_dev"):
        changed = {**binding, "role": role}
        with pytest.raises(ValueError, match="Real/protected"):
            validate_feature_binding(changed, changed)


def test_cpu_safety_caps_precede_dense_features():
    large = graph(257)
    large.x[:, 0] = 5
    with pytest.raises(ValueError, match="node safety cap"):
        topology_views(Batch.from_data_list([large]))
    g = graph(2)
    g.edge_index = g.edge_index.repeat(1, 513)
    g.edge_attr = g.edge_attr.repeat(513, 1)
    with pytest.raises(ValueError, match="directed-edge safety cap"):
        topology_views(Batch.from_data_list([g]))
