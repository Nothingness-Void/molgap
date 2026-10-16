"""R3 structural tests under finite noninitial parameter states, never training."""
from copy import deepcopy
from dataclasses import asdict
import pytest
import torch
from torch_geometric.data import Batch

from molgap.esa_model import ESA, ESAConfig
from molgap.esa_invariant import ESAInvariant, ESAInvariantConfig
from modern2d_fixtures import graph


def stressed(model):
    with torch.no_grad():
        for layer in model.encoder:
            for p in (*layer.attention.parameters(), *layer.mlp.parameters()): p.zero_()
        model.pool.seeds.mul_(30)
        for p in model.pool.mlp.parameters(): p.mul_(3)
    return model.eval()


def permuted(g, seed):
    other=g.clone(); gen=torch.Generator().manual_seed(seed)
    atoms=torch.randperm(g.num_nodes,generator=gen); inverse=torch.argsort(atoms)
    edges=torch.randperm(g.edge_index.shape[1],generator=gen)
    other.x=other.x[atoms]
    other.edge_index=inverse[g.edge_index[:,edges]]
    other.edge_attr=g.edge_attr[edges]
    return other


@pytest.mark.parametrize("seed", [42, 73, 101])
def test_frozen_v1_reproduces_only_seed_count_branch_defect(seed,record_property):
    torch.set_num_threads(1);torch.manual_seed(seed)
    old=stressed(ESA())
    errors={}
    with torch.no_grad():
        for n in (16,17,18):
            g=graph(n)
            errors[2*(n-1)]=float((old(Batch.from_data_list([g]))-
                old(Batch.from_data_list([permuted(g,seed+5)]))).abs().max())
    record_property("v1_synthetic_normalized_output_errors",str(errors))
    assert errors[32] > 1e-3, errors
    assert errors[30] < 3e-5 and errors[34] < 3e-5, errors


@pytest.mark.parametrize("seed", [42,73])
@pytest.mark.parametrize("tokens", [31,32,33,30,34])
@pytest.mark.parametrize("state", ["initial","finite-stressed"])
def test_v2_output_pma_and_parameter_gradients_at_all_token_boundaries(seed,tokens,state,record_property):
    torch.set_num_threads(1);torch.manual_seed(seed)
    cfg=ESAInvariantConfig(hidden=32,heads=4,layer_types=("M","S","P"),graph_dim=32,mlp_hidden=64)
    model=ESAInvariant(cfg).eval()
    if state != "initial": stressed(model)
    n=tokens+1 if tokens in (31,33) else tokens//2+1
    g=graph(n)
    if tokens in (31,33):
        g.edge_index=g.edge_index[:,:tokens];g.edge_attr=g.edge_attr[:tokens]
    other=permuted(g,seed+11)
    snapshots=[]
    handle=model.pool.register_forward_hook(lambda m,a,o:snapshots.append(o.detach().clone()))
    output=model(Batch.from_data_list([g]));output.sum().backward()
    before={k:p.grad.clone() for k,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    changed=model(Batch.from_data_list([other]));changed.sum().backward()
    handle.remove()
    torch.testing.assert_close(output,changed,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(snapshots[0],snapshots[1],atol=3e-5,rtol=3e-5)
    maximum=0.
    for k,p in model.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all())
        maximum=max(maximum,float((before[k]-p.grad).abs().max()))
        torch.testing.assert_close(before[k],p.grad,atol=5e-4,rtol=2e-4)
    record_property("v2_normalized_output_max_abs",float((output-changed).abs().max()))
    record_property("v2_parameter_gradient_max_abs",maximum)


def test_v2_pma_input_gradients_are_equivariant_not_index_paired():
    torch.manual_seed(42)
    model=ESAInvariant(ESAInvariantConfig(hidden=16,heads=2,layer_types=("M","P"),graph_dim=16,mlp_hidden=32,seeds=5))
    for tokens in (4,5,6):
        x=torch.randn(1,tokens,16,requires_grad=True); order=torch.randperm(tokens)
        y=model.pool(x);grad=torch.autograd.grad(y.square().sum(),x)[0]
        shuffled=x.detach()[:,order].requires_grad_();z=model.pool(shuffled)
        changed=torch.autograd.grad(z.square().sum(),shuffled)[0]
        torch.testing.assert_close(y,z,atol=3e-6,rtol=2e-5)
        torch.testing.assert_close(grad[:,order],changed,atol=1e-5,rtol=5e-5)


def test_isolated_atom_coverage_is_explicit_without_modifying_frozen_reference():
    torch.manual_seed(42);old=ESA().eval();new=ESAInvariant().eval()
    a=graph(2);b=graph(3,edges=a.edge_index)
    with torch.no_grad(): assert torch.equal(old(Batch.from_data_list([a])),old(Batch.from_data_list([b])))
    with pytest.raises(ValueError,match="isolated atoms unsupported"):
        new(Batch.from_data_list([b]))
    assert new.config.isolated_atom_policy=="reject-pending-authorized-census"
