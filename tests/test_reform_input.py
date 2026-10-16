"""Editable references freeze through old identities; no real authority or API."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from molgap.experiment_input import freeze_experiment_input,prepare_experiment_input,resolve_experiment_input
from molgap.experiment_spec import ExperimentSpec
from test_experiment_spec import payload


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")


@pytest.fixture
def editable(tmp_path,payload):
    import sys
    repo=tmp_path/"checkout";repo.mkdir()
    write(repo/"spec.json",payload)
    write(repo/"workflow.json",{"spec_identity":ExperimentSpec(payload).identity,
        "source_files":["src/molgap/old_model.py","recipes/frozen.json"]})
    write(repo/"authority.json",{"fixture_only":True})
    profile=tmp_path/"machine.json"
    write(profile,{"format":"molgap-machine-paths-v1","python":sys.executable,
        **{name:str(tmp_path/name) for name in ("data_root","artifact_root","scratch_root")}})
    value={"format":"molgap-experiment-input-v1","spec_ref":"spec.json","workflow_ref":"workflow.json",
        "machine_profile":str(profile),"authority_ref":"authority.json",
        "monitor_paths":["experiments/question/STATUS.md"],"attempt_id":"v1"}
    path=repo/"experiment.json";write(path,value)
    return SimpleNamespace(repo=repo,path=path,value=value,payload=payload)


def test_normal_json_freeze_is_canonical_idempotent_and_outside_worktree(editable):
    a=freeze_experiment_input(editable.path,editable.repo)
    b=freeze_experiment_input(editable.path,editable.repo)
    assert a==b and not a["submitted"] and a["scientific_acceptance"]=="NOT_EVALUATED"
    frozen=Path(a["run_directory"])/"inputs"
    assert not frozen.is_relative_to(editable.repo)
    assert (frozen/"experiment_spec.json").read_bytes()==ExperimentSpec(editable.payload).to_json().encode()
    plan=json.loads((frozen/"workflow_plan.json").read_bytes())
    assert "source_files" not in plan and plan["source_profile"]=="runtime-v1"
    assert plan["source_records"]==["authority.json","recipes/frozen.json"]
    assert not (editable.repo/"research_memory").exists()


@pytest.mark.parametrize("change",["unknown","duplicate","nonfinite","authority","workflow","monitor","ambiguous"])
def test_input_errors_fail_before_publication(editable,change):
    value=editable.value
    if change=="unknown":value["user_authorized"]=True
    elif change=="authority":value["authority_ref"]="missing.json"
    elif change=="monitor":value["monitor_paths"]=[{},"../outside"]
    elif change in {"workflow","ambiguous"}:
        plan=json.loads((editable.repo/"workflow.json").read_text())
        if change=="workflow":plan["spec_identity"]="0"*64
        else:plan["source_profile"]="runtime-v1"
        write(editable.repo/"workflow.json",plan)
    write(editable.path,value)
    if change=="duplicate":editable.path.write_text('{"format":1,"format":2}')
    if change=="nonfinite":editable.path.write_text('{"format":NaN}')
    with pytest.raises((ValueError,TypeError)):
        freeze_experiment_input(editable.path,editable.repo)
    assert not (editable.repo.parent/"artifact_root").exists()


def test_changed_reference_cannot_overwrite_frozen_run(editable):
    freeze_experiment_input(editable.path,editable.repo)
    editable.payload["prospective"]["hypothesis"]="A different hypothesis"
    write(editable.repo/"spec.json",editable.payload)
    plan=json.loads((editable.repo/"workflow.json").read_text())
    plan["spec_identity"]=ExperimentSpec(editable.payload).identity;write(editable.repo/"workflow.json",plan)
    with pytest.raises(ValueError,match="[Ii]mmutable|overwrite|[Cc]onflict"):
        freeze_experiment_input(editable.path,editable.repo)


def test_prepare_once_then_reuse_existing_owner_and_package(editable,monkeypatch):
    def original_owner(spec,root,plan,output,*,source_cache):
        assert root==editable.repo and source_cache.is_relative_to(editable.repo.parent/"artifact_root")
        write(output/"workflow_report.json",{"status":"PREPARED_FOR_PLATFORM","package_identity":"a"*64})
        return {"status":"PREPARED_FOR_PLATFORM"}
    owner=Mock(side_effect=original_owner)
    monkeypatch.setattr("molgap.experiment_workflow.prepare_workflow",owner)
    verify=Mock(return_value={"spec_identity":ExperimentSpec(editable.payload).identity,"package_identity":"a"*64})
    monkeypatch.setattr("molgap.experiment_package.verify_experiment_source_package",verify)
    a=prepare_experiment_input(editable.path,editable.repo)
    b=prepare_experiment_input(editable.path,editable.repo)
    assert a["prepared"]==b["prepared"] and b["reused_preparation"]
    owner.assert_called_once();verify.assert_called_once()
    assert not b["submitted"] and b["authority_ref"]=="authority.json"


@pytest.mark.parametrize("report",[None,{"status":"PREPARATION_FAILED_RECONCILE_BEFORE_RETRY"}])
def test_interrupted_preparation_is_retained_not_retried(editable,monkeypatch,report):
    frozen=freeze_experiment_input(editable.path,editable.repo)
    prepared=Path(frozen["run_directory"])/"prepared";prepared.mkdir()
    if report:write(prepared/"workflow_report.json",report)
    owner=Mock(side_effect=AssertionError("blind retry"))
    monkeypatch.setattr("molgap.experiment_workflow.prepare_workflow",owner)
    assert prepare_experiment_input(editable.path,editable.repo)["status"]=="RECONCILIATION_REQUIRED"
    assert prepared.is_dir();owner.assert_not_called()


@pytest.fixture
def platform_entry():
    path=Path(__file__).resolve().parents[1]/"platforms/kaggle/submit_workflow.py"
    spec=importlib.util.spec_from_file_location("reform_platform_entry",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_new_platform_entry_has_no_post_without_execute(platform_entry,monkeypatch,capsys):
    prep=Mock(return_value={"status":"PREPARED_FOR_PLATFORM","submitted":False})
    backend=Mock(side_effect=AssertionError("no POST or SDK on preparation"))
    monkeypatch.setattr("molgap.experiment_input.prepare_experiment_input",prep)
    monkeypatch.setattr(platform_entry,"submit_prepared_workflow",backend)
    assert platform_entry.main(["--experiment","experiment.json"])==0
    assert not json.loads(capsys.readouterr().out)["submitted"]
    backend.assert_not_called();prep.assert_called_once()


def test_execute_requires_credentials_before_any_preparation(platform_entry,monkeypatch):
    prep=Mock(side_effect=AssertionError("preparation without credentials"))
    monkeypatch.setattr("molgap.experiment_input.prepare_experiment_input",prep)
    assert platform_entry.main(["--experiment","experiment.json","--execute"])==2
    prep.assert_not_called()


def test_execute_delegates_exact_frozen_bindings_to_original_journal(platform_entry,monkeypatch):
    values={"status":"PREPARED_FOR_PLATFORM","prepared":"prepared","operations":"operations",
        "authority_ref":"authority.json","monitor_paths":["STATUS.md"]}
    monkeypatch.setattr("molgap.experiment_input.prepare_experiment_input",Mock(return_value=values))
    backend=Mock(return_value={"status":"SUBMISSION_BOUND"})
    monkeypatch.setattr(platform_entry,"submit_prepared_workflow",backend)
    assert platform_entry.main(["--experiment","experiment.json","--credentials","private.json","--execute"])==0
    backend.assert_called_once_with(Path("prepared"),Path("private.json"),Path("operations"),
        authority_ref="authority.json",monitor_paths=["STATUS.md"],execute=True)
