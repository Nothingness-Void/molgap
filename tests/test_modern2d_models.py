from copy import deepcopy

import pytest
import torch
from torch_geometric.data import Batch

from molgap.fast_sandbox import run_sandbox
from molgap.ppgt_model import AdaRMSNorm, sl2_attention, stable_rms
from modern2d_fixtures import graph, batch, factory


@pytest.mark.parametrize("scale", [0., 1e-30, 1., 1e20])
def test_adarms_stable_forward_backward_and_initial_rms(scale):
    x = (torch.randn(5, 8) * scale).requires_grad_()
    norm = AdaRMSNorm(8)
    y = norm(x)
    assert torch.isfinite(y).all()
    assert torch.allclose(y, x / stable_rms(x, 1e-8), atol=1e-6)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in norm.parameters())
    with torch.no_grad(): norm.alpha.fill_(1); norm.beta.zero_()
    assert torch.allclose(norm(x), x, atol=1e-6, rtol=2e-6)


@pytest.mark.parametrize("phi", [False, True])
def test_sl2_independent_distance_formula_forward_and_gradients(phi):
    torch.random.default_generator.manual_seed(42)
    q, k, v = [torch.randn(2, 3, 4, requires_grad=True) for _ in range(3)]
    bias = torch.randn(2, 3, 3, requires_grad=True)
    multiplier = torch.randn(2, 3, 3, requires_grad=True) if phi else None
    actual = sl2_attention(q, k, v, bias=bias, multiplier=multiplier)
    distance = ((q.unsqueeze(-2) - k.unsqueeze(-3)) ** 2).sum(-1)
    weights = (-distance / (2 * 4 ** .5) + bias).softmax(-1)
    if phi: weights = weights * multiplier
    reference = weights @ v
    assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)
    parameters = (q, k, v, bias) + ((multiplier,) if phi else ())
    a = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    b = torch.autograd.grad(reference.square().sum(), parameters)
    for left, right in zip(a, b): assert torch.allclose(left, right, atol=1e-5, rtol=1e-4)


def test_sl2_padding_empty_queries_and_nonedge_bias():
    q = k = v = torch.ones(1, 3, 4)
    mask = torch.eye(3, dtype=torch.bool); mask[-1] = False
    result = sl2_attention(q, k, v, allowed=mask)
    assert torch.isfinite(result).all() and not result[:, -1].any()
    assert torch.equal(result[:, :2], torch.ones(1, 2, 4))


@pytest.mark.parametrize("family", ["ppgt_small", "esa_small"])
def test_disposable_cpu_sandbox_exact_roundtrip_and_all_gradients(family):
    report = run_sandbox(lambda: factory(family))
    assert report["status"] == "PASSED", report
    model = factory(family); model(batch()).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("family", ["ppgt_small", "esa_small"])
def test_batch_isolation_permutations_labels_and_unused_rwse(family):
    torch.set_num_threads(1)
    model = factory(family).eval()
    data = batch(); original = deepcopy(data.to_dict())
    expected = model(data)
    singles = torch.cat([model(Batch.from_data_list([g])) for g in data.to_data_list()])
    assert torch.equal(expected, singles)
    reversed_graphs = Batch.from_data_list(list(reversed(data.to_data_list())))
    assert torch.equal(model(reversed_graphs), expected.flip(0))
    graphs = data.to_data_list()
    for g in graphs:
        order = torch.arange(len(g.x) - 1, -1, -1)
        inverse = torch.argsort(order)
        g.x = g.x[order]; g.edge_index = inverse[g.edge_index]
        permutation = torch.arange(g.edge_index.shape[1] - 1, -1, -1)
        g.edge_index = g.edge_index[:, permutation]; g.edge_attr = g.edge_attr[permutation]
    assert torch.allclose(model(Batch.from_data_list(graphs)), expected, atol=2e-6, rtol=1e-5)
    assert all(torch.equal(data[k], value) for k, value in original.items())
    data.y.fill_(float("nan")); data.random_walk_pe.fill_(float("nan"))
    assert torch.equal(model(data), expected)


def test_ppgt_single_atom_and_disconnected_graph():
    assert torch.isfinite(factory("ppgt_small")(Batch.from_data_list([graph(1)]))).all()
    g = graph(4, edges=torch.tensor([[0, 1], [1, 0]]))
    assert torch.isfinite(factory("ppgt_small")(Batch.from_data_list([g]))).all()


@pytest.mark.parametrize("family", ["ppgt_small", "esa_small"])
def test_model_has_no_file_network_or_accelerator_io(family, monkeypatch):
    model = factory(family)
    import builtins, socket
    def forbidden(*a, **k): raise AssertionError("Unexpected IO/CUDA")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(torch.Tensor, "cuda", forbidden)
    assert model(batch()).dtype == torch.float32
