"""Synthetic tensor/statics only; never construct/execute a molecular model."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch

from molgap.gptrans_source_dependence import BASE, panel_indices, source_statistics, matched_prediction, validate_release_contract

ROOT=Path(__file__).resolve().parents[1]


def test_source_support_and_conditional_mass():
    p=torch.tensor([[[[.5,.25,.25,0.], [.8,.1,.1,0.], [.8,.1,.1,0.], [.1,.2,.3,.4]]]])
    valid=torch.tensor([[True,True,True,False]])
    result=source_statistics(p,valid)
    assert result["real_query/virtual_source_mass"].item()==pytest.approx(.8)
    assert result["real_query/real_conditional_maximum"].item()==pytest.approx(.5)
    assert result["real_query/real_conditional_effective_fraction"].item()==pytest.approx(1)


def test_padding_and_zero_real_mass_not_false_collapse():
    p=torch.zeros(1,1,3,3)
    p[:,:,:,0]=1
    result=source_statistics(p,torch.tensor([[True,True,False]]))
    assert result["real_query/real_mass_zero"].item()==1
    assert result["real_query/real_conditional_effective_fraction"].item()==0
    assert result["real_query/effective_source_fraction"].item()==.5


def test_panels_input_only_stable_and_disjoint():
    for role in ("original_100k","unseen_500k"):
        panel=panel_indices(role)
        assert len(panel)==512 and len(set(panel.tolist()))==512
        assert (panel==panel_indices(role)).all() and (panel[:-1]<panel[1:]).all()


def test_reproduction_alignment_fail_closed():
    saved=dict(source_idx=torch.tensor([1,3,7]),target_eV=torch.ones(3),prediction_eV=torch.ones(3))
    panel={k:v[[0,2]] for k,v in saved.items()}
    assert matched_prediction(panel,saved)["accepted"]
    panel["source_idx"]=torch.tensor([1,9])
    with pytest.raises(ValueError):
        matched_prediction(panel,saved)


@pytest.mark.parametrize("key,value",[("optimizer_steps",1),("diagnostic_rows_per_role",1024),("observer_policy","perturb"),("protected_roles_read",True)])
def test_release_rejects_scope_change(key,value):
    contract=json.loads((ROOT/BASE/"contract.json").read_text())
    release={"files":{n:s["sha256"] for n,s in {**contract["model_assets"],**contract["reference_payloads"]}.items()}}
    inventory={n:{"sha256":s} for n,s in contract["source_identities"].items()}
    meta=json.loads((ROOT/BASE/"kernel-metadata.json").read_text())
    validate_release_contract(contract,release,inventory,meta)
    altered=deepcopy(contract)
    altered[key]=value
    with pytest.raises(ValueError):
        validate_release_contract(altered,release,inventory,meta)


def test_no_training_or_geometry_calls():
    tree=ast.parse((ROOT/"src/molgap/gptrans_source_dependence.py").read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {"step","backward","train"} for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Attribute) and n.attr in {"pos","edge_distance","wedge_angle_cos"} for n in ast.walk(tree))


def test_saved_analysis_does_not_fit_or_analyze_labels():
    tree=ast.parse((ROOT/"src/molgap/gptrans_source_dependence_analysis.py").read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {"step","backward","train","fit"} for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value in {"target_eV","prediction_eV"} for n in ast.walk(tree))
