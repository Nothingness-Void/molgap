"""Synthetic CPU graphs only; no dataset, training or protected role consumption."""
import copy

import pytest
import torch
from torch import nn
from torch_geometric.nn.norm import BatchNorm

from molgap.k1_norm_mechanisms import (
    VARIANTS, NodeRMSNorm, GraphRMSNorm, NodeAdaNorm, apply_addon,
)
from molgap.qm9_neural_atom import make_encoder


@pytest.fixture
def inputs():
    return dict(x=torch.tensor([[0]*9, [1]+[0]*8, [2]+[0]*8, [3]+[0]*8, [4]+[0]*8]),
                edge_index=torch.tensor([[0,1,1,2,3,4], [1,0,2,1,4,3]]),
                edge_attr=torch.zeros(6,3,dtype=torch.long), batch=torch.tensor([0,0,0,1,1]),
                random_walk_pe=torch.arange(80,dtype=torch.float32).reshape(5,16)/80)


def _prefixes():
    return tuple(f"local_blocks.{i}.{name}." for i in range(9) for name in ("norm1", "norm3"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_shared_weights_affine_rng_and_exact_sites(variant):
    model = make_encoder("neural_atom_k1").double().eval()
    with torch.no_grad():
        for i, block in enumerate(model.local_blocks):
            block.norm1.module.weight.fill_(0.75+i/20)
            block.norm3.module.bias.fill_(-0.25+i/30)
    model.local_blocks[0].norm1.module.weight.requires_grad_(False)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    params = dict(model.named_parameters())
    convs = [b.conv for b in model.local_blocks]
    mlps = [b.mlp for b in model.local_blocks]
    edge, mixers, head = model.edge_updates, model.neural_atom_mixers, model.head
    rng = torch.get_rng_state().clone()
    apply_addon(model, {"variant": variant})
    assert torch.equal(rng, torch.get_rng_state())
    assert model.edge_updates is edge and model.neural_atom_mixers is mixers and model.head is head
    assert len(model._molgap_substrate.mutations) == 9
    assert not any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    assert sum(isinstance(m, NodeRMSNorm) for m in model.modules()) == 18
    assert not hasattr(model, "final_norm")
    for key, value in model.state_dict().items():
        if not key.startswith(_prefixes()):
            assert torch.equal(value, before[key])
    for key, param in model.named_parameters():
        if not key.startswith(_prefixes()):
            assert param is params[key]
    for i, block in enumerate(model.local_blocks):
        assert block.conv is convs[i] and block.mlp is mlps[i]
        for name in ("norm1", "norm3"):
            norm = getattr(block, name)
            assert not norm.training and norm.weight.dtype == torch.float64
            for field in ("weight", "bias"):
                old = f"local_blocks.{i}.{name}.module.{field}"
                assert torch.equal(getattr(norm, field), before[old])
                assert getattr(norm, field).requires_grad == params[old].requires_grad
    ids = [id(p) for _,p in model.named_parameters(remove_duplicate=False)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("variant", VARIANTS)
def test_real_graph_forward_finite_grad_and_strict_resume(variant, inputs):
    model = make_encoder("neural_atom_k1")
    base = copy.deepcopy(model.state_dict())
    apply_addon(model, {"variant": variant})
    model.train()
    output = model(**inputs)
    assert output.shape == (2,1) and torch.isfinite(output).all()
    output.square().mean().backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    for block in model.local_blocks:
        for norm in (block.norm1, block.norm3):
            for p in norm.parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
    resumed = make_encoder("neural_atom_k1")
    resumed.load_state_dict(base, strict=True)
    with pytest.raises(RuntimeError):
        resumed.load_state_dict(model.state_dict(), strict=True)
    apply_addon(resumed, {"variant": variant})
    resumed.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    resumed.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs), resumed(**inputs), rtol=0, atol=0)
    with pytest.raises(ValueError):
        apply_addon(model, {"variant": variant})


@pytest.mark.parametrize("variant", VARIANTS)
def test_exact_residual_formula(variant):
    model = make_encoder("neural_atom_k1").eval()
    apply_addon(model, {"variant": variant})
    block = model.local_blocks[0]
    x = torch.arange(5*192,dtype=torch.float32).reshape(5,192)/100
    edges = torch.tensor([[0,1,3,4],[1,0,4,3]])
    edge_attr = torch.ones(4,64)
    batch = torch.tensor([0,0,0,1,1])
    with torch.no_grad():
        if variant == "pre_node_rms":
            h = x + block.conv(block.norm1(x,batch), edges, edge_attr=edge_attr)
            expected = h + block.mlp(block.norm3(h,batch))
        else:
            h = block.norm1(x + block.conv(x,edges,edge_attr=edge_attr),batch)
            expected = block.norm3(h + block.mlp(h),batch)
        torch.testing.assert_close(block(x,edges,batch,edge_attr=edge_attr),expected,rtol=0,atol=0)


def test_graph_rms_channel_formula_graph_independence_and_no_centering():
    base = BatchNorm(3).module
    norm = GraphRMSNorm(base)
    x = torch.tensor([[1.,2.,3.], [3.,4.,5.], [100.,200.,300.]])
    batch = torch.tensor([2,2,7])
    actual = norm(x,batch)
    expected = x[:2] / (x[:2].square().mean(0)+1e-5).sqrt()
    torch.testing.assert_close(actual[:2],expected)
    torch.testing.assert_close(actual[:2],norm(x[:2]))
    assert (actual > 0).all()
    changed = x.clone()
    changed[-1] *= 1000
    torch.testing.assert_close(norm(changed,batch)[:2],actual[:2],rtol=0,atol=0)


def test_ada_original_epsilon_initialization_and_amplitude_restoration():
    base = BatchNorm(4).module.double()
    rms, ada = NodeRMSNorm(base), NodeAdaNorm(base)
    x = torch.tensor([[1.,2.,4.,8.],[-3.,1.,0.,7.]],dtype=torch.float64)
    assert torch.equal(ada.alpha,torch.zeros_like(ada.alpha))
    assert torch.equal(ada.beta,torch.ones_like(ada.beta))
    # The original equal-epsilon formula has this explicit 5ppm init factor.
    torch.testing.assert_close(ada(x),rms(x)*(1+ada.eps)**0.5,rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(ada(x),rms(x),rtol=ada.eps,atol=0)
    with torch.no_grad():
        ada.alpha.fill_(1)
        ada.beta.zero_()
    torch.testing.assert_close(ada(x),x,rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(ada(10*x),10*x,rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize("case", ["depth", "shared", "extra_bn", "epsilon", "conv", "placement"])
def test_topology_rejects_atomically(case):
    model = make_encoder("neural_atom_k1")
    if case == "depth":
        del model.local_blocks[-1]
    elif case == "shared":
        model.local_blocks[-1].norm3 = model.local_blocks[0].norm1
    elif case == "extra_bn":
        model.extra_bn = nn.BatchNorm1d(192)
    elif case == "epsilon":
        model.local_blocks[-1].norm3.module.eps = 1e-3
    elif case == "conv":
        model.local_blocks[-1].conv = nn.Identity()
    else:
        model.local_blocks[-1].norm_with_batch = True
    before = dict(model.named_modules())
    with pytest.raises(ValueError):
        apply_addon(model, {"variant":"post_node_rms"})
    assert dict(model.named_modules()) == before
    assert not hasattr(model,"_molgap_substrate")


@pytest.mark.parametrize("config", [None, [], {}, {"variant":"unknown"}, {"variant":1}])
def test_invalid_config(config):
    model = make_encoder("neural_atom_k1")
    with pytest.raises(ValueError,match="config"):
        apply_addon(model,config)
    assert not hasattr(model,"_molgap_substrate")


def test_catalog_registration():
    from molgap.experiment_catalog import ADDONS
    from molgap.common.spec import validate_addon_config
    for variant in VARIANTS:
        contract = ADDONS[("k1_"+variant,"1")]
        assert contract.family_versions == ("3",)
        assert contract.execution_modes == (("3","composed"),)
        assert contract.apply_hook == "molgap.k1_norm_mechanisms:apply_addon"
        validate_addon_config(contract,{"variant":variant},family_version="3")
        with pytest.raises(ValueError):
            validate_addon_config(contract,{"variant":"other"},family_version="3")
