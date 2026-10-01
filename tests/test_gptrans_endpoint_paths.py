"""Tiny tensor algorithms only; no dataset, trained model or platform access."""
import torch
from torch import nn
from molgap.gptrans_endpoint_paths import endpoint_path_contrast, parameter_groups
from molgap.pcqm_gptrans_v4 import _ema_decay, _scientific_fields


def toy(edges, values, n):
    d = torch.full((1, n, n), 21, dtype=torch.long)
    d[0].fill_diagonal_(0)
    idx = torch.tensor(edges, dtype=torch.long).T
    d[0, idx[0], idx[1]] = 1
    for k in range(n):
        d = torch.minimum(d, d[:, :, k:k+1] + d[:, k:k+1, :])
    b = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    return d, idx, b


def run(d, idx, b):
    return endpoint_path_contrast(d, torch.zeros(idx.shape[1], dtype=torch.long), idx[0], idx[1], b)


def test_contrast_distinguishes_equal_histograms():
    edges = [(0,1),(1,0),(1,2),(2,1),(2,3),(3,2)]
    d,i,b = toy(edges, [1,1,1,1,2,2], 4)
    d2,i2,b2 = toy(edges, [1,1,2,2,1,1], 4)
    assert run(d,i,b)[0,0,0,3] != run(d2,i2,b2)[0,0,0,3]


def test_ties_permutation_covariance_reverse_and_direct_mask():
    edges = [(0,1),(1,0),(0,2),(2,0),(1,3),(3,1),(2,3),(3,2)]
    d,i,b = toy(edges, [1,1,3,3,2,2,4,4], 4)
    expected = run(d,i,b)
    assert expected[0,0,0,3] == -.5
    assert torch.equal(expected, -expected.transpose(-1,-2))
    assert (expected[d[:,None,:] < 2] == 0).all()
    p = torch.tensor([2,0,3,1])
    inverse = torch.argsort(p)
    actual = run(d[:,p][:,:,p], inverse[i], b)
    assert torch.allclose(actual, expected[:,:,p][:,:,:,p])
    b.requires_grad_()
    run(d,i,b).square().sum().backward()
    assert torch.isfinite(b.grad).all() and b.grad.abs().sum() > 0


def test_groups_exempt_only_bias_and_1d():
    model = nn.Sequential(nn.Embedding(4,3), nn.Linear(3,3), nn.LayerNorm(3))
    groups, inventory = parameter_groups(model, .05)
    by_name = {r['name']: r for r in inventory}
    assert by_name['0.weight']['weight_decay'] == .05
    assert by_name['1.weight']['weight_decay'] == .05
    assert by_name['1.bias']['weight_decay'] == 0
    assert by_name['2.weight']['weight_decay'] == 0
    assert len({id(p) for g in groups for p in g['params']}) == len(list(model.parameters()))


def test_recipe_identity_is_one_declared_intervention():
    reference = _scientific_fields('degree_scale_ema999')
    grouped = _scientific_fields('degree_group_decay_ema999')
    assert {k for k in reference if reference[k] != grouped[k]} == {'optimizer_fingerprint'}
    assert _scientific_fields('degree_path_endpoints_ema999') == reference
    assert _ema_decay('degree_group_decay_ema999') == .999
