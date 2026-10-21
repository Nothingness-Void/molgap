"""Synthetic read-only diagnostics; no production dataset or remote resources."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from molgap.k1_norm_observations import NormDiagnostics
from molgap.k1_norm_mechanisms import apply_addon
from molgap.qm9_neural_atom import make_encoder


def test_gradient_windows_resume_and_stop_at_first_1000():
    diagnostic = NormDiagnostics("post_node_rms")
    observe = diagnostic.observer(1.0)
    for _ in range(150):
        observe.after_clip(None,torch.tensor(2.0))
    checkpoint = diagnostic.state_dict()
    resumed = NormDiagnostics("post_node_rms")
    resumed.restore(checkpoint,optimizer_step=150)
    observe = resumed.observer(1.0)
    for _ in range(900):
        observe.after_clip(None,torch.tensor(2.0))
    assert resumed.completed_updates == 1050
    assert len(resumed.gradient_windows) == 10 and not resumed.pending
    assert [w["first_step"] for w in resumed.gradient_windows] == list(range(1,1001,100))
    assert all(w["updates"] == 100 and w["clip_fraction"] == 1.0
               and w["preclip_grad_norm_mean"] == 2.0 for w in resumed.gradient_windows)
    assert checkpoint["completed_updates"] == 150
    with pytest.raises(ValueError):
        resumed.restore(checkpoint,optimizer_step=149)


def test_missing_resume_does_not_restart_gradient_window():
    diagnostic = NormDiagnostics("pre_node_rms")
    diagnostic.restore(None,optimizer_step=1200)
    diagnostic.observer(1.0).after_clip(None,torch.tensor(2.0))
    assert diagnostic.completed_updates == 1201
    assert not diagnostic.gradient_windows and not diagnostic.pending
    assert diagnostic.history_status == "missing_before_resume"


def test_update_observer_preserves_existing_callbacks():
    from molgap.family_training_core import UpdateObserver
    calls = []
    previous = UpdateObserver(after_backward=lambda model:calls.append("backward"),
        after_clip=lambda model,norm:calls.append("clip"))
    diagnostic = NormDiagnostics("post_node_rms")
    combined = diagnostic.observer(1.0,previous)
    combined.after_backward(None)
    combined.after_clip(None,torch.tensor(0.5))
    assert calls == ["backward","clip"]
    assert diagnostic.completed_updates == 1


@pytest.mark.parametrize("variant", ["pre_node_rms", "post_node_ada"])
def test_train_prefix_probe_preserves_state_rng_flags_and_gradient_counter(variant):
    model = make_encoder("neural_atom_k1")
    apply_addon(model,{"variant":variant})
    model.train()
    model.head[2].eval()  # Preserve mixed flags, too.
    diagnostic = model._k1_norm_diagnostics
    dataset = [Data(x=torch.zeros(3,9,dtype=torch.long),
        edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
        edge_attr=torch.zeros(4,3,dtype=torch.long),
        random_walk_pe=torch.ones(3,16)/16,
        y=torch.tensor([1.0]),source_idx=torch.tensor([10+i])) for i in range(2)]
    recipe = SimpleNamespace(batch_size=2,seed=42,target_mean_eV=0.,target_std_eV=1.)
    def forward(batch,*,model):
        return model(x=batch.x,edge_index=batch.edge_index,edge_attr=batch.edge_attr,
                     batch=batch.batch,random_walk_pe=batch.random_walk_pe)
    kernel = SimpleNamespace(model=model,recipe=recipe,forward=forward)
    loaders = SimpleNamespace(data=SimpleNamespace(train=dataset,train_rows=2,train_source_start=10))
    before = copy.deepcopy(model.state_dict())
    flags = [m.training for m in model.modules()]
    rng = torch.get_rng_state().clone()
    hooks = [len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules()]
    diagnostic.probe(kernel,loaders,completed_pass=40)
    assert torch.equal(torch.get_rng_state(),rng)
    assert flags == [m.training for m in model.modules()]
    assert hooks == [len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules()]
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())
    assert diagnostic.completed_updates == 0 and not diagnostic.pending
    result = diagnostic.probes[0]
    assert result["rows"] == 2 and result["role"] == "train"
    assert result["weights"] == "live" and result["mode"] == "eval"
    assert len([key for key in result["layer_means"] if key.endswith("output_rms")]) == 9
    assert any("adaptive_over_input_rms" in key for key in result["layer_means"]) == (variant == "post_node_ada")
