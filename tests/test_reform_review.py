"""Review regressions use local synthetic data and existing preparation owners."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from molgap import experiment_package as package
from molgap.experiment_input import freeze_experiment_input,prepare_experiment_input
from molgap.experiment_spec import ExperimentSpec
from test_reform_input import editable,platform_entry,write
from test_experiment_package import repo,payload


def test_prepublication_io_failure_recovers_same_identity_without_losing_failed_bytes(editable,monkeypatch):
    calls=[]
    def owner(spec,root,plan,output,**kwargs):
        calls.append(spec.identity)
        if len(calls)==1:
            write(output/"workflow_report.json",{"format":"molgap-workflow-preparation-v1",
                "spec_identity":spec.identity,"status":"ERROR_BEFORE_PUBLICATION",
                "prospective_published":False,"submitted":False,"publication_intent":False})
            (output/"failed-byte-evidence.txt").write_bytes(b"original failed attempt")
            raise OSError("injected local IO interruption")
        write(output/"workflow_report.json",{"status":"PREPARED_FOR_PLATFORM","package_identity":"a"*64})
        return {"status":"PREPARED_FOR_PLATFORM"}
    monkeypatch.setattr("molgap.experiment_workflow.prepare_workflow",owner)
    with pytest.raises(OSError):prepare_experiment_input(editable.path,editable.repo)
    result=prepare_experiment_input(editable.path,editable.repo)
    assert result["status"]=="PREPARED_FOR_PLATFORM" and len(set(calls))==1 and len(calls)==2
    retained=Path(result["prepared"]).parent/"local_preparation_failures"
    assert len(list(retained.iterdir()))==1
    assert next(retained.glob("*/failed-byte-evidence.txt")).read_bytes()==b"original failed attempt"
    assert not (Path(result["operations"])).exists()


@pytest.mark.parametrize("fault",["intent","published","operations","unknown","in_progress"])
def test_local_retry_never_crosses_published_or_platform_uncertainty(editable,monkeypatch,fault):
    frozen=freeze_experiment_input(editable.path,editable.repo)
    run=Path(frozen["run_directory"])
    report={"format":"molgap-workflow-preparation-v1","spec_identity":frozen["spec_identity"],
        "status":"ERROR_BEFORE_PUBLICATION","prospective_published":False,"submitted":False}
    if fault=="intent":report["publication_intent"]=True
    if fault=="published":report["prospective_published"]="inspect_plan_result"
    if fault=="operations":write(run/"operations/submission_state.json",{"phase":"SUBMIT_UNKNOWN"})
    if fault=="unknown":report={}
    if fault=="in_progress":report.update(status="PREPARING",publication_intent=False)
    write(run/"prepared/workflow_report.json",report)
    owner=Mock(side_effect=AssertionError("unsafe retry"))
    monkeypatch.setattr("molgap.experiment_workflow.prepare_workflow",owner)
    assert prepare_experiment_input(editable.path,editable.repo)["status"]=="RECONCILIATION_REQUIRED"
    owner.assert_not_called()


def test_legacy_partial_cache_is_retained_then_automatically_rebuilt(repo,payload,tmp_path):
    cache=tmp_path/"cache";spec=ExperimentSpec(payload)
    first=package.build_experiment_source_package(spec,repo,["README.md"],tmp_path/"first",source_cache=cache)
    cached=next(cache.iterdir())
    for name in ("SOURCE_COMMIT.txt","SOURCE_ARCHIVE_SHA256.txt","SOURCE_FILES.json"):(cached/name).unlink()
    second=package.build_experiment_source_package(spec,repo,["README.md"],tmp_path/"second",source_cache=cache)
    assert first==second
    retained=list(cache.glob(".incomplete-*"));assert len(retained)==1
    assert {p.name for p in retained[0].iterdir()}=={"source.tar.gz"}
    assert package.verify_experiment_source_package(tmp_path/"second",repo)==second


def test_interrupted_cache_build_never_publishes_a_partial_final_object(repo,payload,tmp_path,monkeypatch):
    cache=tmp_path/"cache";original=package.build_v4_source_bundle;calls=[]
    def interrupted(**kwargs):
        calls.append(kwargs)
        result=original(**kwargs)
        if len(calls)==1:raise OSError("interrupted before atomic directory publication")
        return result
    monkeypatch.setattr(package,"build_v4_source_bundle",interrupted)
    with pytest.raises(OSError):
        package.build_experiment_source_package(ExperimentSpec(payload),repo,["README.md"],tmp_path/"failed",source_cache=cache)
    assert not list(cache.iterdir())
    package.build_experiment_source_package(ExperimentSpec(payload),repo,["README.md"],tmp_path/"recovered",source_cache=cache)
    assert len(calls)==2 and len(list(cache.iterdir()))==1


def test_local_missing_initialization_is_actionable_and_exception_secrets_are_not_echoed(platform_entry,monkeypatch,capsys):
    monkeypatch.setattr("molgap.experiment_input.prepare_experiment_input",Mock(side_effect=ValueError("Missing frozen initialization file")))
    assert platform_entry.main(["--experiment","fixture.json"])==2
    value=json.loads(capsys.readouterr().out)
    assert value["stage"]=="local_preparation" and value["field"]=="arms.initial_state"
    assert "missing" in value["reason"]
    monkeypatch.setattr("molgap.experiment_input.prepare_experiment_input",Mock(side_effect=ValueError("Bearer untrusted-secret-token")))
    platform_entry.main(["--experiment","fixture.json"])
    assert "untrusted-secret-token" not in capsys.readouterr().out


def test_derived_package_pins_human_source_and_keeps_committed_v1_unchanged(repo,payload,tmp_path):
    import hashlib
    from test_experiment_package import git, dump
    from molgap.common.wire import canonical_fingerprint
    config_name="experiments/question/experiment.json"
    config=repo/config_name;config.parent.mkdir(parents=True)
    dump(config,{"format":"molgap-scientific-experiment-v1","experiment_id":payload["experiment_id"]})
    git(repo,"add",".");git(repo,"-c","commit.gpgsign=false","commit","-m","Human input fixture")
    source={"path":config_name,"sha256":hashlib.sha256(config.read_bytes()).hexdigest(),
            "compiler":"molgap-scientific-experiment-v1"}
    prefix=f"experiments/{payload['experiment_id']}/generated/"+canonical_fingerprint({
        "input":source["sha256"],"commit":git(repo,"rev-parse","HEAD")})
    name=prefix+"/recipe.json";derived=repo/name;derived.parent.mkdir(parents=True)
    derived.write_bytes(b'{"learning_rate":0.0005}')
    options={"generated_records":{name:derived.read_bytes()},"configuration_source":source}
    target=tmp_path/"derived"
    manifest=package.build_experiment_source_package(ExperimentSpec(payload),repo,
        ["README.md",config_name,name],target,source_cache=tmp_path/"cache",**options)
    assert manifest["format"]==package.GENERATED_PACKAGE_FORMAT
    assert package.verify_experiment_source_package(target,repo)==manifest
    assert name not in git(repo,"ls-files").splitlines()
    derived.write_bytes(b'{"learning_rate":0.0004}')
    with pytest.raises(ValueError,match="payload|Generated"):
        package.verify_experiment_source_package(target,repo)
    old=package.build_experiment_source_package(ExperimentSpec(payload),repo,["README.md"],tmp_path/"v1")
    assert old["format"]==package.PACKAGE_FORMAT and "generated_records" not in old


@pytest.mark.parametrize("bad",["src/changed.py","experiments/question/generated/evil.py","../escape.json"])
def test_generated_records_cannot_bypass_committed_python_or_path_confinement(repo,payload,tmp_path,bad):
    with pytest.raises(ValueError):
        package.build_experiment_source_package(ExperimentSpec(payload),repo,["README.md",bad],tmp_path/"bad",
            generated_records={bad:b'{}'},configuration_source={"path":"README.md","sha256":"a"*64,
            "compiler":"molgap-scientific-experiment-v1"})
