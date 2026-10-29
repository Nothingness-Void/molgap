"""Editable references freeze through old identities; no real authority or API."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from molgap.experiment_input import freeze_experiment_input,prepare_experiment_input,locate_run
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


def test_locate_run_reuses_frozen_ids_without_fabricating_receipt_or_predictions(editable):
    frozen=freeze_experiment_input(editable.path,editable.repo)
    run=Path(frozen["run_directory"])
    before={str(p):p.read_bytes() for p in run.rglob("*") if p.is_file()}
    report=locate_run(editable.value["machine_profile"],editable.repo,"synthetic-run","v1")
    assert report["status"]=="RUN_LOCATED_ONLY" and report["receipt"] is None
    assert report["package_identity"] is None and not report["network_accessed"]
    assert report["files"]["operations/launch_receipt.json"] is None
    assert report["files"]["inputs/experiment_spec.json"]["sha256"]
    assert before=={str(p):p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with pytest.raises(FileNotFoundError):
        locate_run(editable.value["machine_profile"],editable.repo,"missing-run","v1")
    with pytest.raises(ValueError):
        locate_run(editable.value["machine_profile"],editable.repo,"../outside","v1")


@pytest.mark.parametrize("fault",["input_identity","format","authority_ref","monitor_paths","source_commit"])
def test_locate_run_rejects_damaged_original_binding(editable,fault):
    frozen=freeze_experiment_input(editable.path,editable.repo)
    path=Path(frozen["run_directory"])/"inputs/input_binding.json"
    binding=json.loads(path.read_bytes())
    binding[fault] = {"input_identity":"not-a-digest","format":"unknown",
        "authority_ref":"../escape.json","monitor_paths":["../escape.md"],
        "source_commit":"not-a-commit"}[fault]
    write(path,binding)
    with pytest.raises(ValueError):
        locate_run(editable.value["machine_profile"],editable.repo,"synthetic-run","v1")


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


@pytest.fixture
def frozen_scientific(editable):
    """Legacy frozen-input envelope; scientific/package execution is mocked."""
    from molgap.common.paths import MachinePaths
    from molgap.common.wire import canonical_fingerprint
    from molgap.experiment_launch import canonical_json
    value={"format":"molgap-scientific-experiment-v1","experiment_id":editable.payload["experiment_id"],
        "logical_run_id":editable.payload["logical_run_id"],
        **{k:editable.value[k] for k in ("attempt_id","machine_profile","authority_ref","monitor_paths")}}
    write(editable.path,value)
    spec=ExperimentSpec(editable.payload)
    plan={"spec_identity":spec.identity,"frozen_workflow":True}
    paths=MachinePaths.from_file(value["machine_profile"])
    run=paths.run_directory(value["logical_run_id"],value["attempt_id"])
    binding={"format":value["format"],"input_identity":canonical_fingerprint(value),
        "spec_identity":spec.identity,"workflow_identity":canonical_fingerprint(plan),
        "authority_ref":value["authority_ref"],"monitor_paths":value["monitor_paths"]}
    for name,payload in (("input_binding.json",binding),("experiment_spec.json",spec.to_dict()),("workflow_plan.json",plan)):
        target=run/"inputs"/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(canonical_json(payload).encode())
    write(run/"prepared/workflow_report.json",{"status":"PREPARED_FOR_PLATFORM","package_identity":"a"*64})
    return SimpleNamespace(**vars(editable),scientific_value=value,run=run,spec=spec)


def test_legacy_scientific_freeze_reuses_without_compiler_or_current_git(frozen_scientific,monkeypatch):
    case=frozen_scientific
    forbidden=Mock(side_effect=AssertionError("Existing run must not compile or inspect current Git"))
    monkeypatch.setattr("molgap.experiment_input._compile_scientific_input",forbidden)
    monkeypatch.setattr("molgap.experiment_package._git",forbidden)
    verify=Mock(return_value={"spec_identity":case.spec.identity,"package_identity":"a"*64})
    monkeypatch.setattr("molgap.experiment_package.verify_experiment_source_package",verify)
    before={str(p):p.read_bytes() for base in (case.run/"inputs",case.run/"prepared")
            for p in base.rglob("*") if p.is_file()}
    result=prepare_experiment_input(case.path,case.repo)
    assert result["reused_preparation"] and result["status"]=="PREPARED_FOR_PLATFORM"
    assert before=={str(p):p.read_bytes() for base in (case.run/"inputs",case.run/"prepared")
                   for p in base.rglob("*") if p.is_file()}
    assert "local_observation" not in result and "local_observation_status" not in result
    assert not (case.run/"local_observations").exists()
    assert not (case.run/"operations").exists()
    forbidden.assert_not_called();verify.assert_called_once()


def test_timing_observation_failure_does_not_invalidate_durable_preparation(frozen_scientific,monkeypatch):
    case=frozen_scientific
    from molgap import experiment_input as owner
    publish=owner.publish_immutable_bytes
    def interrupted(path,payload):
        if "local_observations" in path.parts:
            raise OSError("untrusted exception details must not be returned")
        return publish(path,payload)
    monkeypatch.setattr(owner,"publish_immutable_bytes",interrupted)
    result={"status":"PREPARED_FOR_PLATFORM","package_identity":"a"*64,
        "local_call_seconds":0.25,"source_commit":"frozen-source","reused_preparation":False}
    report=(case.run/"prepared/workflow_report.json").read_bytes()
    owner._observe_preparation(case.run,case.spec,result)
    assert result["status"]=="PREPARED_FOR_PLATFORM" and not result["reused_preparation"]
    assert result["local_observation_status"]=="MISSING" and "local_observation" not in result
    assert "untrusted" not in str(result) and not (case.run/"operations").exists()
    assert (case.run/"prepared/workflow_report.json").read_bytes()==report


@pytest.mark.parametrize("fault",["input","spec","workflow","authority","partial","published"])
def test_frozen_scientific_conflicts_and_uncertainty_do_not_compile(frozen_scientific,monkeypatch,fault):
    case=frozen_scientific
    forbidden=Mock(side_effect=AssertionError("No replacement identity"))
    monkeypatch.setattr("molgap.experiment_input._compile_scientific_input",forbidden)
    if fault=="input":
        value=dict(case.scientific_value,scientific_change="changed")
        write(case.path,value)
    elif fault=="spec":
        value=dict(case.payload,logical_run_id="other-run")
        write(case.run/"inputs/experiment_spec.json",value)
    elif fault=="workflow":write(case.run/"inputs/workflow_plan.json",{"spec_identity":"0"*64})
    elif fault=="authority":
        path=case.run/"inputs/input_binding.json";binding=json.loads(path.read_bytes());binding["authority_ref"]="different.json";write(path,binding)
    elif fault=="partial":(case.run/"inputs/workflow_plan.json").unlink()
    else:
        write(case.run/"prepared/workflow_report.json",{"status":"ERROR_REQUIRES_RECONCILIATION",
            "publication_intent":True,"prospective_published":"inspect_plan_result"})
        write(case.run/"operations/submission_state.json",{"phase":"SUBMIT_UNKNOWN"})
    before={str(p):p.read_bytes() for p in case.run.rglob("*") if p.is_file()}
    if fault=="published":assert prepare_experiment_input(case.path,case.repo)["status"]=="RECONCILIATION_REQUIRED"
    else:
        with pytest.raises(ValueError):prepare_experiment_input(case.path,case.repo)
    assert before=={str(p):p.read_bytes() for p in case.run.rglob("*") if p.is_file()}
    forbidden.assert_not_called()


def test_prepare_once_then_reuse_existing_owner_and_package(editable,monkeypatch):
    def original_owner(spec,root,plan,output,*,source_cache):
        assert root==editable.repo and source_cache.is_relative_to(editable.repo.parent/"artifact_root")
        write(output/"workflow_report.json",{"status":"PREPARED_FOR_PLATFORM","package_identity":"a"*64})
        return {"status":"PREPARED_FOR_PLATFORM"}
    owner=Mock(side_effect=original_owner)
    monkeypatch.setattr("molgap.experiment_workflow.prepare_workflow",owner)
    monkeypatch.setattr("molgap.experiment_workflow.check_workflow_readiness",Mock(return_value={"status":"WORKFLOW_INPUTS_READY"}))
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
    prep=Mock(return_value={"status":"PREPARED_FOR_PLATFORM","submitted":False,
        "package_identity":"a"*64,"full_owner_detail":{"retained":True}})
    backend=Mock(side_effect=AssertionError("no POST or SDK on preparation"))
    monkeypatch.setattr("molgap.experiment_input.prepare_experiment_input",prep)
    monkeypatch.setattr(platform_entry,"submit_prepared_workflow",backend)
    assert platform_entry.main(["--experiment","experiment.json"])==0
    compact=json.loads(capsys.readouterr().out)
    assert not compact["submitted"] and compact["package_identity"]=="a"*64
    assert "full_owner_detail" not in compact
    assert platform_entry.main(["--experiment","experiment.json","--verbose"])==0
    assert json.loads(capsys.readouterr().out)==prep.return_value
    backend.assert_not_called();assert prep.call_count==2
    blocker={"check":"family_recipes","error_type":"ValueError",
             "message":"Family validator was imported from another checkout"}
    prep.return_value={"status":"BLOCKED","stage":"workflow_readiness","blockers":[blocker],
                       "prepared":"uncreated-preparation","submitted":False}
    assert platform_entry.main(["--experiment","experiment.json"])==1
    blocked=json.loads(capsys.readouterr().out)
    assert blocked["blockers"]==[blocker]
    assert "report" not in blocked
    backend.assert_not_called()


def test_execute_requires_credentials_before_any_preparation(platform_entry,monkeypatch):
    prep=Mock(side_effect=AssertionError("preparation without credentials"))
    monkeypatch.setattr("molgap.experiment_input.prepare_experiment_input",prep)
    assert platform_entry.main(["--experiment","experiment.json","--execute"])==2
    prep.assert_not_called()


def test_execute_delegates_exact_frozen_bindings_to_original_journal(platform_entry,monkeypatch):
    from molgap.experiment_package import _PackageVerification
    verification = _PackageVerification()
    monkeypatch.setattr("molgap.experiment_package._PackageVerification", lambda: verification)
    values={"status":"PREPARED_FOR_PLATFORM","prepared":"prepared","operations":"operations",
        "authority_ref":"authority.json","monitor_paths":["STATUS.md"]}
    prepare=Mock(return_value=values)
    monkeypatch.setattr("molgap.experiment_input.prepare_experiment_input",prepare)
    backend=Mock(return_value={"status":"SUBMISSION_BOUND"})
    monkeypatch.setattr(platform_entry,"submit_prepared_workflow",backend)
    assert platform_entry.main(["--experiment","experiment.json","--credentials","private.json","--execute"])==0
    prepare.assert_called_once_with(Path("experiment.json"),Path.cwd(),_verification=verification)
    backend.assert_called_once_with(Path("prepared"),Path("private.json"),Path("operations"),
        authority_ref="authority.json",monitor_paths=["STATUS.md"],execute=True,
        recovery_authority=None,recovery_authority_sha256=None,source_version=1,kernel_version=1,
        _verification=verification)
