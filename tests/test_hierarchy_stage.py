import torch
from torch import nn
from molgap.local_hierarchy_adapter import ExistingLoopHeads
from molgap.training_reproducibility import capture_rng_state
from molgap.v4_runtime import model_state_sha256
from molgap import hierarchy_stage


class HookFixture(nn.Module):
    """Synthetic feature producer, not a MolGap model execution."""
    def __init__(self):
        super().__init__()
        self.anchor=nn.Parameter(torch.zeros(1))
        self.local_blocks=nn.ModuleList([nn.Identity()])
        self.edge_updates=nn.ModuleList([nn.Identity()])
        self.neural_atom_mixers=nn.ModuleDict({'9':nn.Identity()})
    def forward(self,x,edge_index,edge_attr,batch,rwse=None):
        node=self.local_blocks[-1](x)
        self.edge_updates[-1](edge_attr)
        return self.neural_atom_mixers['9'](node+3)


def test_existing_loop_uses_last_exchange_and_removes_hooks():
    model=HookFixture()
    heads=ExistingLoopHeads(model,'k1')
    x=torch.randn(4,192);edge=torch.randn(2,64)
    model(x,None,edge,None)
    assert torch.equal(heads.node_state,x+3)
    assert torch.equal(heads.edge_state,edge)
    heads.close()
    assert not model._forward_hooks
    assert not model.local_blocks[-1]._forward_hooks
    assert not model.neural_atom_mixers['9']._forward_hooks


def test_stage_keeps_encoder_but_restores_gap_head_and_rng(tmp_path,monkeypatch):
    class Fixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Linear(3,3)
            self.head=nn.Linear(3,1)
    model=Fixture()
    original={k:v.clone() for k,v in model.head.state_dict().items()}
    rng=capture_rng_state()['torch'].clone()
    identity=model_state_sha256(model)
    monkeypatch.setattr(hierarchy_stage,'LabelledTrainingGraphs',lambda *a: [])
    monkeypatch.setattr(hierarchy_stage,'qualify_stage',lambda *a,**kw: None)
    def pretrain(m,*a,**kw):
        with torch.no_grad():
            m.encoder.weight.fill_(8)
            m.head.weight.fill_(7)
        torch.rand(17)
        return m,{'epochs_completed':10}
    monkeypatch.setattr(hierarchy_stage,'pretrain_local_hierarchy',pretrain)
    hierarchy_stage.apply_pretraining(model,[],family='k1',output=tmp_path,
        config={'label_root':str(tmp_path),'label_manifest_sha256':'a'*64,
                'epochs':10,'initialization_sha256':identity},source_commit='b'*40)
    assert torch.equal(model.encoder.weight,torch.full_like(model.encoder.weight,8))
    assert all(torch.equal(v,model.head.state_dict()[k]) for k,v in original.items())
    assert torch.equal(rng,torch.get_rng_state())
    assert (tmp_path/'stage_complete.json').is_file()
