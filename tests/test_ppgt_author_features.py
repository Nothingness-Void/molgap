"""Small synthetic CPU author feature oracle; no corpus or model execution."""
from contextlib import nullcontext
import pytest
import torch
from torch_geometric.data import Data

from modern2d_fixtures import graph
from molgap.ppgt_author_features import feature_cost, positional_inputs, rrwp, sinusoidal_pe
from ppgt_author_reference_harness import reference_namespace


@pytest.fixture(scope="module")
def author():
    return reference_namespace()


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("kind", ["chain", "asymmetric", "parallel", "isolates"])
def test_rrwp_powers_and_sparse_orientation_match_raw_author(author, symmetric, kind):
    sample = graph(4, 9)
    if kind == "asymmetric":
        sample.edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    elif kind == "parallel":
        sample.edge_index = torch.cat((sample.edge_index, sample.edge_index[:, [0, 3]]), 1)
    elif kind == "isolates":
        sample.edge_index = torch.tensor([[0, 1], [1, 0]])
    sample.edge_attr = torch.zeros(sample.edge_index.shape[1], 3, dtype=torch.long)
    raw = Data(x=sample.x.clone(), edge_index=sample.edge_index.clone(), edge_attr=sample.edge_attr.clone())
    with pytest.warns(UserWarning) if kind == "asymmetric" else nullcontext():
        author["add_full_rrwp"](raw, walk_length=3, sym_norm=symmetric)
    got, degree = rrwp(sample, 3, sym_norm=symmetric)
    dense = torch.zeros_like(got)
    dense[raw.rrwp_index[1], raw.rrwp_index[0]] = raw.rrwp_attr
    assert got.shape == (4, 4, 4)
    torch.testing.assert_close(got, dense, atol=3e-7, rtol=2e-6)
    torch.testing.assert_close(degree, raw.deg.float())
    torch.testing.assert_close(got.diagonal(dim1=0, dim2=1).t(), raw.rrwp)
    assert torch.equal(got[..., 0], torch.eye(4))
    if kind == "isolates":
        assert torch.count_nonzero(got[2:, :, 1:]) == 0


@pytest.mark.parametrize("bases", [0, 1, 3])
def test_fourier_channel_order_matches_author_method(author, bases):
    encoder = author["MLPSinEdgeEncoder"](4, 8, pe_name="rrwp", sin_L=bases)
    values = torch.linspace(-0.8, 1.2, 20).view(5, 4)
    torch.testing.assert_close(sinusoidal_pe(values, bases), encoder._sin_pe(values), atol=0, rtol=0)


def test_positional_scope_log_order_and_label_rwse_independence():
    sample = graph(3, 4)
    before = positional_inputs(sample, 2)
    sample.y.fill_(float("nan"))
    sample.random_walk_pe.fill_(float("inf"))
    after = positional_inputs(sample, 2)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    walk, pair, log_order = before
    assert pair.shape[-1] == 4 and torch.equal(pair[..., :-1], walk)
    torch.testing.assert_close(pair[..., -1], torch.full((3, 3), 1 / 3))
    torch.testing.assert_close(log_order, torch.full((3, 1), torch.log1p(torch.tensor(3.0))))
    assert not any(value.requires_grad for value in before)


def test_caps_reject_without_truncation_and_cost_counts_identity():
    sample = graph(4, 0)
    with pytest.raises(ValueError, match="safety cap"):
        rrwp(sample, 2, max_nodes=3)
    with pytest.raises(ValueError, match="safety cap"):
        rrwp(sample, 2, max_edges=2)
    with pytest.raises(ValueError, match="ceiling"):
        rrwp(sample, 2, max_nodes=1025)
    cost = feature_cost(4, steps=3, bases=2)
    assert cost["rrwp_bytes_fp32"] == 4 * 4 * 4 * 4
    assert cost["spe_bytes_fp32"] == 4 * 4 * 4 * 5 * 5
