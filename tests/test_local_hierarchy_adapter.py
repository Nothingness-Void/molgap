import pytest
import torch
from types import SimpleNamespace
from molgap.local_hierarchy_adapter import HierarchyConfig, undirected_mask, sparse_gptrans_states


@pytest.mark.parametrize("values", [{"mask_rate":0}, {"mask_rate":float('nan')}, {"group_weight":-1}, {"head_seed":True}])
def test_invalid_config(values):
    with pytest.raises(ValueError): HierarchyConfig(**values)


def test_bond_reverse_and_rng_resume():
    edges=torch.tensor([[0,1,1,2],[1,0,2,1]])
    generator=torch.Generator().manual_seed(42)
    saved=generator.get_state()
    mask=undirected_mask(edges,generator,.15)
    assert mask[0]==mask[1] and mask[2]==mask[3] and mask.any()
    generator.set_state(saved)
    assert torch.equal(mask,undirected_mask(edges,generator,.15))
    assert undirected_mask(torch.empty(2,0,dtype=torch.long),generator,.15).numel()==0


def test_only_real_nodes_and_bonds_gathered():
    node=torch.arange(2*4*2).reshape(2,4,2)
    pair=torch.arange(2*3*4*4).reshape(2,3,4,4)
    batch=SimpleNamespace(x=torch.zeros(5,9),batch=torch.tensor([0,0,0,1,1]),
                          edge_index=torch.tensor([[0,1,3,4],[1,0,4,3]]))
    n,e=sparse_gptrans_states(node,pair,batch)
    assert torch.equal(n,torch.cat([node[0,1:4],node[1,1:3]]))
    assert torch.equal(e[0],pair[0,:,1,2])
    assert torch.equal(e[2],pair[1,:,1,2])
    assert e.shape==(4,3)


def test_cross_graph_bond_rejected():
    batch=SimpleNamespace(x=torch.zeros(2,9),batch=torch.tensor([0,1]),edge_index=torch.tensor([[0],[1]]))
    with pytest.raises(ValueError): sparse_gptrans_states(torch.zeros(2,2,2),torch.zeros(2,3,2,2),batch)
