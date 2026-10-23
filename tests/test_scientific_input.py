"""Actual K1/GPTrans declaration preparation on temporary CPU fixtures only."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from molgap.common.declarations import ExecutionProfile, ExecutionQuestion
from molgap.experiment_input import resolve_experiment_input
from molgap.experiment_spec import ExperimentSpec
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_data import DatasetSelection
from molgap.family_training_runner import build_screen_recipe, family_arm_bindings
from molgap.v4_runtime import state_dict_sha256
from test_experiment_workflow import (_arm, _spec_payload, _write_strict_retained_reference_plan,
                                      _write_json, _git, _pointer)
from test_comparison_readiness import _assess, _prelaunch, _target_transform_asset
from test_v5_common import _envelope

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=[("neural_atom_k1", "3"), ("gptrans_t", "2")])
def scientific(tmp_path_factory, request):
    tmp_path = tmp_path_factory.mktemp("scientific")
    family, version = request.param
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / ".gitattributes").write_bytes(b"* -text -filter -ident -working-tree-encoding\n")
    # Copy active code, then commit the independent fixture repository. This
    # tests genuine isolated imports without locking live files to old history.
    for source in (ROOT / "src/molgap").rglob("*.py"):
        dest = repo / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    for name in ("platforms/kaggle/run_experiment.py", "platforms/kaggle/submit_workflow.py"):
        dest = repo / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    torch.manual_seed(42)
    if family == "neural_atom_k1":
        from molgap.qm9_neural_atom import make_encoder
        model = make_encoder(family)
    else:
        from molgap.family_model_ops import make_gptrans_backbone
        model = make_gptrans_backbone()
    initial = tmp_path / "initial.pt"
    torch.save(model.state_dict(), initial)
    state_sha = state_dict_sha256(model.state_dict())
    del model
    manifest = tmp_path / "data_root/synthetic-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(b"{}")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    recipe = FamilyRecipe(family, "reference", 42, state_sha,
        DatasetSelection("synthetic-manifest.json", manifest_sha, True),
        256, 3, 4, .001, .01, ScheduleSpec("cosine_epoch", 2),
        "best-development-live", "python-epoch-shuffle" if family == "neural_atom_k1" else "torch-pass-randperm",
        checkpoint_every=2, target_mean_eV=3., target_std_eV=2., torch_version=torch.__version__.split("+")[0])
    record = build_screen_recipe("reference", configuration=recipe, family=(family, version),
        source_idx_sha256="2" * 64, target_sha256="3" * 64,
        train_role_identity="c" * 64, development_role_identity="c" * 64)
    recipe_name = "experiments/reference/contract.json"
    _write_json(repo / recipe_name, record)
    arm = _arm(family, version, "reference", recipe_sha256=hashlib.sha256((repo / recipe_name).read_bytes()).hexdigest(),
               state_sha256=state_sha)
    arm["data"]["dataset"]["sha256"] = manifest_sha
    bindings = family_arm_bindings(recipe)
    arm["base"] = bindings["base"]
    arm["training"].update({k: bindings[k] for k in ("sampler", "transform", "objective")})
    arm["data"]["roles"].append({**arm["data"]["roles"][0], "role": "development"})
    spec = ExperimentSpec(_spec_payload([arm], platform="kaggle"))
    _write_json(repo / "experiments/reference/experiment_spec.json", spec.to_dict())
    plan_name = _write_strict_retained_reference_plan(repo, spec, record["acceptance_requirements"], {"reference": recipe_name})
    bundle_path = repo / "experiments/reference/reference_bundle.json"
    bundle = json.loads(bundle_path.read_bytes())
    transform = _target_transform_asset()
    transform.update(mean=3., std=2., ddof=1, variance_convention="sample")
    from molgap.comparison_readiness import target_transform_asset_digest
    transform["asset_sha256"] = target_transform_asset_digest(transform)
    _write_json(repo / bundle["target_transform_asset_ref"], transform)
    bundle["comparison_identity"]["target_transform_asset_sha256"] = transform["asset_sha256"]
    bundle["comparison_identity"].update(seed=42, precision="fp32", optimizer_steps=4,
        sample_presentations=512, ema_enabled=False, evaluation_weight_source="live")
    _write_json(bundle_path, bundle)
    prelaunch_path = repo / "experiments/reference/comparison_prelaunch.json"
    _write_json(prelaunch_path, _prelaunch(bundle))
    reference_plan = json.loads((repo / plan_name).read_bytes())
    reference_plan["arms"][0]["reference_bundle"] = _pointer(repo, bundle_path)
    reference_plan["arms"][0]["comparison_prelaunch"] = _pointer(repo, prelaunch_path)
    reference_plan["arms"][0]["reference_artifacts"]["target_transform_asset"] = _pointer(
        repo, repo / bundle["target_transform_asset_ref"])
    _write_json(repo / plan_name, reference_plan)
    authority_name = "experiments/question/authority.json"
    _write_json(repo / authority_name, {"user_authorized": True, "allocated_t4_hours_ceiling": 1,
                                      "arms": ["candidate"], "seeds": [42], "fixture_only": True})
    _write_json(repo / "experiments/support/v5_evidence.json", _envelope(authority_name))
    # A global rebuild would require this unrelated historical checkpoint.
    # Preparation retains its canonical metadata without hydrating its weights.
    historical = _assess()
    for side in ("candidate_artifact_bindings", "reference_artifact_bindings"):
        for name, binding in historical[side].items():
            binding["ref"] = f"experiments/history/{side}/{name}.bin"
            artifact = repo / binding["ref"]
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes((side + ":" + name).encode())
            binding["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    historical["reference_artifact_bindings"]["checkpoint"] = {
        "ref": "experiments/history/unavailable-checkpoint.pt", "sha256": "a" * 64}
    from molgap.comparison_readiness import validate_comparison_readiness
    from molgap.research_memory.paths import verify_bound_artifact
    with pytest.raises(ValueError, match="required repository pointer is missing.*unavailable-checkpoint"):
        validate_comparison_readiness(historical,
            evidence_verifier=lambda ref, sha: verify_bound_artifact(repo, ref, sha))
    _write_json(repo / "experiments/history/comparison_readiness.json", historical)
    platform_name = "experiments/question/platform.json"
    _write_json(repo / platform_name, {"platform": spec.to_dict()["platform"], "kaggle": {
        "account": "synthetic-account", "kernel": "synthetic-account/scientific-fixture",
        "title": "Scientific Fixture", "datasets": ["synthetic-account/source-data"],
        "source_dataset": "synthetic-account/source-data", "accelerator": "NvidiaTeslaT4"}})
    # Existing registered policy copied as an input, never a new production rule.
    policy_path = next(p for p in (ROOT / "research_memory/policies").glob("*.json")
                       if json.loads(p.read_bytes()).get("policy_type") == "research_action")
    policy = json.loads(policy_path.read_bytes())
    _write_json(repo / "research_memory/policies/fixture.json", policy)
    profile_path = tmp_path / "machine.json"
    _write_json(profile_path, {"format": "molgap-machine-paths-v1", "python": sys.executable,
        **{key: str(tmp_path / key) for key in ("data_root", "artifact_root", "scratch_root")}})
    question = ExecutionQuestion(policy["policy_id"], policy["action_rule"]["action"],
        "Does an optimizer parameter change alter a bounded screen?", "Untested optimizer sensitivity",
        "optimizer", ("evidence-common-test",), ("No effect",), ("fixture-closed",),
        "Review terminal evidence", "Retain baseline")
    profile = ExecutionProfile("desktop", "B", "kaggle", "CPU synthetic fixture, declared T4x2 only",
                               "Bounded same-contract screen", None, None)
    value = {"format": "molgap-scientific-experiment-v1", "experiment_id": "scientific-fixture",
        "logical_run_id": "scientific-fixture-run", "attempt_id": "v1", "machine_profile": str(profile_path),
        "authority_ref": authority_name, "monitor_paths": ["experiments/question/STATUS.md"],
        "timestamp": "2026-10-16T00:00:00Z", "question": asdict(question), "profile": asdict(profile),
        "platform_ref": platform_name, "role_ref": "experiments/reference/roles.json",
        "decision_ref": "experiments/reference/decision.md", "seeds": [42], "arms": [{
            "arm_id": "candidate", "model": {"family": family, "version": version, "addons": []},
            "recipe_ref": recipe_name, "data_ref": {"spec_ref": "experiments/reference/experiment_spec.json", "arm_id": "reference"},
            "initial_state_ref": str(initial), "training": {"learning_rate": .0008},
            "reference_ref": {"plan_ref": plan_name, "arm_id": "reference"}, "purpose": "optimizer_comparison"}]}
    path = repo / "experiments/question/experiment.json"
    _write_json(path, value)
    _git(repo, "init")
    _git(repo, "config", "user.name", "CPU fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Frozen synthetic scientific inputs")
    return SimpleNamespace(repo=repo, path=path, value=value, tmp=tmp_path)


def test_only_one_human_parameter_edit_compiles_and_recovers_actual_single_entry(scientific, record_property):
    case = scientific
    before = _git(case.repo, "rev-parse", "HEAD")
    case.value["arms"][0]["training"]["learning_rate"] = .0005
    _write_json(case.path, case.value)
    assert _git(case.repo, "diff", "--name-only").splitlines() == [case.path.relative_to(case.repo).as_posix()]
    _git(case.repo, "add", "--", case.path.relative_to(case.repo).as_posix())
    _git(case.repo, "-c", "commit.gpgsign=false", "commit", "-m", "Change only human learning rate")
    program = r'''
import contextlib, importlib.util, io, json, pathlib, sys
repo=pathlib.Path(sys.argv[1]);config=pathlib.Path(sys.argv[2]);sys.path.insert(0,str(repo/"src"))
from molgap import kaggle_workflow
from molgap.experiment_input import resolve_experiment_input
from molgap.experiment_package import verify_experiment_source_package
entry_spec=importlib.util.spec_from_file_location("platform_entry",repo/"platforms/kaggle/submit_workflow.py")
entry=importlib.util.module_from_spec(entry_spec);entry_spec.loader.exec_module(entry)
def forbidden(*a,**kw):raise AssertionError("No SDK or remote submission in CPU acceptance")
entry.submit_prepared_workflow=forbidden
original=kaggle_workflow.stage_inputs
def interrupted(**kw):raise OSError("simulated pre-publication staging interruption")
kaggle_workflow.stage_inputs=interrupted
out=io.StringIO()
with contextlib.redirect_stdout(out):code=entry.main(["--experiment",str(config),"--repo-root",str(repo)])
assert code==2,(code,out.getvalue())
value,spec,plan,paths=resolve_experiment_input(config,repo)
run=paths.run_directory(value["logical_run_id"],value["attempt_id"])
binding=(run/"inputs/input_binding.json").read_bytes()
failed=(run/"prepared/workflow_report.json").read_bytes()
assert json.loads(failed)["prospective_published"] is False
import subprocess
original_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
def document_commit(name):
    (repo/name).write_bytes(b"Unrelated CPU fixture documentation\n")
    subprocess.run(["git","add","--",name],cwd=repo,check=True,capture_output=True)
    subprocess.run(["git","-c","commit.gpgsign=false","commit","-m","Unrelated documentation"],
        cwd=repo,check=True,capture_output=True)
from molgap import experiment_input
experiment_input._compile_scientific_input=forbidden
document_commit("FAILURE_NOTES.md")
out=io.StringIO()
with contextlib.redirect_stdout(out):code=entry.main(["--experiment",str(config),"--repo-root",str(repo)])
blocked=json.loads(out.getvalue())
assert code==1 and blocked["status"]=="RECONCILIATION_REQUIRED",blocked
assert blocked["source_commit"]==original_commit
assert failed==(run/"prepared/workflow_report.json").read_bytes()
assert not (run/"local_preparation_failures").exists()
subprocess.run(["git","switch","--detach",original_commit],cwd=repo,check=True,capture_output=True)
kaggle_workflow.stage_inputs=original
out=io.StringIO()
with contextlib.redirect_stdout(out):code=entry.main(["--experiment",str(config),"--repo-root",str(repo)])
result=json.loads(out.getvalue());assert code==0,result
assert result["status"]=="PREPARED_FOR_PLATFORM" and result["submitted"] is False
workflow_report=json.loads((run/"prepared/workflow_report.json").read_bytes())
planned=next(row["result"] for row in workflow_report["stages"] if row["stage"]=="prospective")
assert planned["rml_rebuild_pending"] is True and planned["rml_rebuilt"] is False
assert not (repo/"research_memory/derived").exists()
assert not (repo/"experiments/history/unavailable-checkpoint.pt").exists()
trajectory_path=next(repo.glob("experiments/scientific-fixture/runs/*/prospective/*/trajectory.json"))
trajectory=json.loads(trajectory_path.read_bytes())
from molgap.research_memory.plan import resolve_decision_state
resolved=resolve_decision_state(repo,trajectory)
assert "evidence-common-test" in resolved["decision_state"]["known_evidence_ids"]
assert (repo/trajectory["decision_state"]["snapshot_ref"]).is_file()
assert binding==(run/"inputs/input_binding.json").read_bytes()
retained=list((run/"local_preparation_failures").glob("prepared-*.json"))
assert len(retained)==1
receipt=json.loads(retained[0].read_bytes())
assert receipt["workflow_report"]==json.loads(failed)
assert receipt["local_payload_retirement"]["status"]=="REMOVED"
assert not (retained[0].parent/retained[0].stem).exists()
out=io.StringIO()
with contextlib.redirect_stdout(out):code=entry.main(["--experiment",str(config),"--repo-root",str(repo)])
assert code==0 and json.loads(out.getvalue())["reused_preparation"]
assert len(list(repo.glob("experiments/scientific-fixture/runs/*/prospective/*/trajectory.json")))==1
assert not (run/"operations").exists()
manifest=verify_experiment_source_package(run/"prepared/package",repo)
assert manifest["format"]=="molgap-experiment-source-package-v2"
assert manifest["configuration_source"]["path"]==config.relative_to(repo).as_posix()
recipe=json.loads((repo/plan["arms"][0]["recipe"]).read_bytes())
assert recipe["learning_rate"]==.0005
assert len(manifest["generated_records"])==4
assert not set(manifest["generated_records"])&set(__import__("subprocess").check_output(["git","ls-files"],cwd=repo,text=True).splitlines())
from molgap.research_memory.trace import file_digest
snapshot={str(p):file_digest(p) for base in (run/"inputs",run/"prepared") for p in base.rglob("*") if p.is_file()}
document_commit("COMPLETED_NOTES.md")
renamed=[]
try:
    # Every unavailable input is retained within this synthetic temporary repo.
    sources=[repo/"experiments/reference",repo/"experiments/scientific-fixture/generated",
             pathlib.Path(value["arms"][0]["initial_state_ref"])]
    for source in sources:
        target=source.with_name(source.name+"-retained")
        assert source.resolve().is_relative_to(repo.parent) and target.resolve().is_relative_to(repo.parent)
        source.rename(target);renamed.append((source,target))
    out=io.StringIO()
    with contextlib.redirect_stdout(out):code=entry.main(["--experiment",str(config),"--repo-root",str(repo)])
    resumed=json.loads(out.getvalue());assert code==0 and resumed["reused_preparation"],resumed
    frozen=experiment_input.freeze_experiment_input(config,repo)
    assert frozen["spec_identity"]==spec.identity
    assert snapshot=={str(p):file_digest(p) for base in (run/"inputs",run/"prepared") for p in base.rglob("*") if p.is_file()}
    assert verify_experiment_source_package(run/"prepared/package")["source_commit"]==original_commit
finally:
    for source,target in reversed(renamed):target.rename(source)
print(json.dumps({"status":"CPU_SCENARIO_ACCEPTED","spec_identity":spec.identity,
 "package_identity":manifest["package_identity"],"family":spec.to_dict()["arms"][0]["family"],
 "same_frozen_binding":True,"prospective_records":1,"remote_requests":0,
 "cross_head_reuse":True,"development_inputs_unavailable":True,"compiler_called_on_continue":False,
 "unpublished_different_head_blocked":True,
 "native_qualification":"NOT_EVALUATED","scientific_acceptance":"NOT_EVALUATED"}))
'''
    result = subprocess.run([sys.executable, "-I", "-c", program, str(case.repo), str(case.path)],
        cwd=case.repo, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    observation = json.loads(result.stdout.splitlines()[-1])
    assert observation["status"] == "CPU_SCENARIO_ACCEPTED"
    assert before != _git(case.repo, "rev-parse", "HEAD")
    record_property("scientific_input_observation", json.dumps(observation))


@pytest.mark.parametrize("fault", ["seed", "batch", "missing_reference", "wrong_contract", "uncommitted", "data_context", "reference_checkpoint"])
def test_scientific_input_does_not_infer_authority_or_reference(scientific, fault):
    from copy import deepcopy
    case = scientific
    original = case.path.read_bytes()
    data_path = case.repo / case.value["arms"][0]["data_ref"]["spec_ref"]
    data_original = data_path.read_bytes()
    checkpoint = case.repo / "experiments/reference/checkpoint.pt"
    checkpoint_original = checkpoint.read_bytes()
    files = {str(p): p.stat().st_size for root in (case.tmp / "artifact_root", case.repo / "experiments/scientific-fixture")
             if root.exists() for p in root.rglob("*") if p.is_file()}
    value = deepcopy(case.value)
    if fault == "seed":value["seeds"] = [42, 43]
    elif fault == "batch":value["arms"][0]["training"]["batch_size"] = 256
    elif fault == "missing_reference":value["arms"][0]["reference_ref"]["plan_ref"] = "missing.json"
    elif fault == "wrong_contract":value["arms"][0]["recipe_ref"] = "experiments/reference/roles.json"
    elif fault == "reference_checkpoint":
        value["logical_run_id"] = "fresh-reference-checkpoint-run"
        checkpoint.write_bytes(checkpoint_original + b"tampered")
    elif fault == "data_context":
        value["logical_run_id"] = "fresh-data-context-run"
        data = json.loads(data_original)
        data["arms"][0]["data"]["roles"][0]["membership_sha256"] = "0" * 64
        _write_json(data_path, data)
    else:value["arms"][0]["training"]["learning_rate"] = .0004
    _write_json(case.path, value)
    try:
        expected = ("Retained reference context" if fault == "data_context" else
                    "Retained reference artifact bytes changed" if fault == "reference_checkpoint" else None)
        with pytest.raises((ValueError, KeyError, FileNotFoundError, RuntimeError), match=expected):
            resolve_experiment_input(case.path, case.repo)
    finally:
        case.path.write_bytes(original)
        data_path.write_bytes(data_original)
        checkpoint.write_bytes(checkpoint_original)
    after = {str(p): p.stat().st_size for root in (case.tmp / "artifact_root", case.repo / "experiments/scientific-fixture")
             if root.exists() for p in root.rglob("*") if p.is_file()}
    assert after == files


def test_two_arm_compilation_uses_independent_device_and_identity_bindings(scientific):
    from copy import deepcopy
    case = scientific
    value = deepcopy(case.value)
    value.update(logical_run_id="scientific-pair-run", authority_ref="experiments/pair/authority.json")
    second = deepcopy(value["arms"][0])
    second.update(arm_id="candidate_two", training={"learning_rate": .0006})
    value["arms"].append(second)
    authority = {"user_authorized": True, "allocated_t4_hours_ceiling": 1,
                 "arms": ["candidate", "candidate_two"], "seeds": [42], "fixture_only": True}
    path = case.repo / "experiments/pair/experiment.json"
    _write_json(path, value)
    _write_json(case.repo / value["authority_ref"], authority)
    _git(case.repo, "add", "--", path.relative_to(case.repo).as_posix(), value["authority_ref"])
    _git(case.repo, "-c", "commit.gpgsign=false", "commit", "-m", "Explicit synthetic two-arm inputs")
    _, spec, plan, _ = resolve_experiment_input(path, case.repo)
    assert [a["device"] for a in plan["arms"]] == [0, 1]
    assert len({a["recipe"] for a in plan["arms"]}) == 2
    assert len({b["trajectory_id"] for b in spec.to_dict()["prospective"]["arms"]}) == 2
    assert [a["scientific_role"] for a in spec.to_dict()["arms"]] == ["candidate", "candidate"]
    assert "same_run_replay" not in spec.to_dict()["prospective"]
    assert len(plan["generated_records"]) == 7


def test_ordinary_addon_runs_complete_scientific_entry(scientific, record_property):
    from copy import deepcopy
    case = scientific
    value = deepcopy(case.value)
    value.update(experiment_id="addon-fixture", logical_run_id="addon-fixture-run")
    arm = value["arms"][0]
    name = "k1_readout_scale" if arm["model"]["family"] == "neural_atom_k1" else "gptrans_readout_scale"
    arm.update(training={}, purpose="architecture_comparison")
    arm["model"]["addons"] = [{"name": name, "version": "1", "config": None}]
    value["question"].update(changed_mechanism="readout-scale", question="Does this ordinary addon integrate faithfully?")
    path = case.repo / "experiments/addon-fixture/experiment.json"
    _write_json(path, value)
    _git(case.repo, "add", "--", path.relative_to(case.repo).as_posix())
    _git(case.repo, "-c", "commit.gpgsign=false", "commit", "-m", "One registered ordinary addon input")
    program = r'''
import contextlib, importlib.util, io, json, pathlib, sys
repo=pathlib.Path(sys.argv[1]);config=pathlib.Path(sys.argv[2]);sys.path.insert(0,str(repo/"src"))
from molgap.experiment_input import resolve_experiment_input
from molgap.experiment_package import verify_experiment_source_package,read_packaged_text
from molgap.experiment_execution import build_addon_declaration
entry_spec=importlib.util.spec_from_file_location("entry",repo/"platforms/kaggle/submit_workflow.py")
entry=importlib.util.module_from_spec(entry_spec);entry_spec.loader.exec_module(entry)
def forbidden(*a,**kw):raise AssertionError("No platform operations in CPU addon acceptance")
entry.submit_prepared_workflow=forbidden
out=io.StringIO()
with contextlib.redirect_stdout(out):code=entry.main(["--experiment",str(config),"--repo-root",str(repo)])
result=json.loads(out.getvalue());assert code==0,result
value,spec,plan,paths=resolve_experiment_input(config,repo)
arm=spec.to_dict()["arms"][0];family=(arm["family"]["name"],arm["family"]["version"])
expected=build_addon_declaration(family,arm["addons"][0]["name"],repo_root=repo)
assert arm["addons"]==[expected]
package=pathlib.Path(result["prepared"])/"package"
manifest=verify_experiment_source_package(package,repo)
recipe=json.loads(read_packaged_text(package,plan["arms"][0]["recipe"]))
assert recipe["mode"]=="composed" and recipe["addons"]==arm["addons"]
acceptance=json.loads(read_packaged_text(package,plan["acceptance_plan"]))["arms"][0]
comparison=json.loads(read_packaged_text(package,acceptance["comparison_prelaunch"]["path"]))
assert comparison["prelaunch_ready"] and comparison["experiment_purpose"]=="architecture_comparison"
assert comparison["declared_intervention_fields"]==["architecture_config_identity"]
release=json.loads((pathlib.Path(result["prepared"])/"release_report.json").read_bytes())
assert release["errors"]==[],release
assert not (paths.run_directory(value["logical_run_id"],value["attempt_id"])/"operations").exists()
print(json.dumps({"status":"CPU_ADDON_ENTRY_ACCEPTED","family":arm["family"],"addon":expected,
 "package_identity":manifest["package_identity"],"spec_identity":spec.identity,
 "initialization_release_checked":True,"prelaunch_purpose":"architecture_comparison",
 "platform_requests":0,"native_qualification":"NOT_EVALUATED"}))
'''
    result = subprocess.run([sys.executable, "-I", "-c", program, str(case.repo), str(path)], cwd=case.repo,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    record_property("ordinary_addon_observation", result.stdout.splitlines()[-1])


def test_ordinary_addon_alias_does_not_waive_real_optimizer_confound(scientific):
    from copy import deepcopy
    case = scientific
    value = deepcopy(case.value)
    value.update(experiment_id="mixed-confound", logical_run_id="mixed-confound-run")
    arm = value["arms"][0]
    name = "k1_readout_scale" if arm["model"]["family"] == "neural_atom_k1" else "gptrans_readout_scale"
    arm.update(training={"learning_rate": .0005}, purpose="architecture_comparison")
    arm["model"]["addons"] = [{"name": name, "version": "1", "config": None}]
    path = case.repo / "experiments/mixed-confound/experiment.json"
    _write_json(path, value)
    _git(case.repo, "add", "--", path.relative_to(case.repo).as_posix())
    _git(case.repo, "-c", "commit.gpgsign=false", "commit", "-m", "Deliberately confounded CPU declaration")
    with pytest.raises(ValueError, match="outside one logical group.*optimizer_identity"):
        resolve_experiment_input(path, case.repo)
    assert not (case.repo / "experiments/mixed-confound/generated").exists()
