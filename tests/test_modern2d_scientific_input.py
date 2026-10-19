"""Configured model front door: actual offline packages, no real data or SDK."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
import subprocess
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from molgap.common.declarations import ExecutionProfile, ExecutionQuestion
from molgap.common.wire import canonical_fingerprint
from molgap.experiment_input import resolve_experiment_input, prepare_experiment_input
from molgap.experiment_spec import ExperimentSpec
from molgap.family_training_runner import build_screen_recipe, family_arm_bindings, read_recipe
from molgap.modern2d_training import Modern2DFormalTrainingProfile
from molgap.modern2d_training_release import CONFIGURED_AUTHORITY_FORMAT, CONFIGURED_PLAN_FORMAT, check_plan
from molgap.modern2d_feature_binding import indexed_transform_identity
from molgap.research_memory.trace import file_digest
from test_experiment_workflow import _arm, _spec_payload, _write_json, _git, _pointer
from test_modern2d_formal_training import case
from test_modern2d_qualification import permit_record
from test_v5_common import _envelope

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configured(case, tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    for source in (ROOT / "src/molgap").rglob("*.py"):
        target = repo / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    for name in ("platforms/kaggle/run_experiment.py", "platforms/kaggle/submit_workflow.py"):
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    train, train_record = permit_record("train_100k")
    dev, dev_record = permit_record("common_dev")
    binding_ref = "experiments/reference/training_input.json"
    _write_json(repo / binding_ref, dict(format="molgap-indexed-training-input-v1", data_root="absent",
        scope="ACCEPTED_INDEXED_INPUT_V1", train=train_record, development=dev_record))
    # Real accepted manifests predate the compiler and are not canonical JSON.
    # Packaging must retain their original pin, not treat them as derived bytes.
    binding_path = repo / binding_ref
    binding_path.write_text(json.dumps(json.loads(binding_path.read_bytes()), indent=2) + "\n", encoding="utf-8")
    recipe = replace(case.recipe, train_rows=100000, development_rows=50000, optimizer_steps=46860,
        dataset=replace(case.recipe.dataset, manifest="training_input.json", manifest_sha256=file_digest(repo / binding_ref)))
    family = recipe.interface
    record = build_screen_recipe("reference", configuration=recipe, family=family,
        source_idx_sha256="2" * 64, target_sha256="3" * 64,
        train_role_identity="c" * 64, development_role_identity="c" * 64)
    recipe_ref = "experiments/reference/contract.json"
    _write_json(repo / recipe_ref, record)
    arm = _arm(*family, "reference", recipe_sha256=file_digest(repo / recipe_ref), state_sha256=recipe.initialization_sha256)
    bindings = family_arm_bindings(recipe)
    arm["base"] = bindings["base"]
    arm["training"].update({k: bindings[k] for k in ("sampler", "transform", "objective")})
    arm["data"]["dataset"]["sha256"] = recipe.dataset.manifest_sha256
    arm["data"]["roles"] = [{"role": r, "usage_sha256": "c" * 64,
        "membership_sha256": "2" * 64, "row_order_sha256": "2" * 64} for r in ("train", "development")]
    source = _spec_payload([arm], platform="kaggle")
    source["platform"]["accelerator"] = "NvidiaTeslaT4"
    spec = ExperimentSpec(source)
    spec_ref = "experiments/reference/experiment_spec.json"
    _write_json(repo / spec_ref, spec.to_dict())
    features = {permit.role: dict(base_dataset_manifest_sha=permit.manifest_sha256,
        source_row_sha="2" * 64, source_topology_sha="4" * 64,
        feature_schema_version=arm["data"]["feature_schema"],
        transform_code_sha=indexed_transform_identity(family),
        transform_config_sha=canonical_fingerprint(record["model_config"]), derived_feature_sha=None,
        role_history_sha=permit.role_history_sha256, role=permit.role, scope=permit.scope) for permit in (train, dev)}
    metadata_ref = "experiments/reference/training_inputs.json"
    _write_json(repo / metadata_ref, dict(format="molgap-modern2d-formal-input-metadata-v1",
        scientific_training_authorized=True, target_mean_eV=recipe.target_mean_eV, target_std_eV=recipe.target_std_eV,
        training_input=dict(path="training_input.json", sha256=recipe.dataset.manifest_sha256),
        models=dict(reference=dict(family=list(family), model_config=record["model_config"],
            feature_bindings=features, feature_binding_sha256={k: canonical_fingerprint(v) for k, v in features.items()}))))
    target_ref = "experiments/reference/targets.json"
    _write_json(repo / target_ref, dict(development_source_idx_sha256="2" * 64, development_target_sha256="3" * 64,
        target_encoding="little-endian-float64-contiguous-raw-bytes"))
    plan_ref = "experiments/reference/acceptance_plan.json"
    _write_json(repo / plan_ref, dict(format="molgap-modern2d-formal-reference-plan-v1", spec_identity=spec.identity,
        inputs=_pointer(repo, repo / metadata_ref), arms=[dict(arm_id="reference",
            contract=_pointer(repo, repo / recipe_ref), target_manifest=_pointer(repo, repo / target_ref))]))
    authority_ref = "experiments/question/authority.json"
    authority = dict(format=CONFIGURED_AUTHORITY_FORMAT, owner="server", purpose="CONFIGURED_SCREEN",
        user_authorized=True, scientific_training_authorized=True, logical_run_id="configured-fixture-run",
        arms=["candidate"], interfaces=[list(family)], seeds=[42], physical_batch_size=128,
        roles=["train_100k", "common_dev"], scale_500k_authorized=False,
        protected_role_access_authorized=False, extra_seed_authorized=False, automatic_successor_authorized=False,
        full_training_authorized=False, allocated_t4_hours_ceiling=1, single_arm_reason="CPU fixture for one family",
        recipes=dict(candidate=dict(horizon_passes=8, horizon_steps=6248)),
        run_reference="fixture/configured", gpu=dict(owner="fixture", accelerator="NvidiaTeslaT4",
            allocated_devices=2, wall_ceiling_seconds=1800, allocated_t4_hours_ceiling=1))
    _write_json(repo / authority_ref, authority)
    _write_json(repo / "experiments/reference/roles.json", {"roles": ["train_100k", "common_dev"]})
    (repo / "experiments/reference/decision.md").write_text("CPU fixture only.\n")
    _write_json(repo / "experiments/support/v5_evidence.json", _envelope(authority_ref))
    platform_ref = "experiments/question/platform.json"
    _write_json(repo / platform_ref, dict(platform=source["platform"], kaggle=dict(account="fixture",
        kernel="fixture/configured", title="Configured", datasets=["fixture/source-data"],
        source_dataset="fixture/source-data", accelerator="NvidiaTeslaT4")))
    policy_path = next(p for p in (ROOT / "research_memory/policies").glob("*.json")
        if json.loads(p.read_bytes()).get("policy_type") == "research_action")
    policy = json.loads(policy_path.read_bytes())
    _write_json(repo / "research_memory/policies/fixture.json", policy)
    machine = tmp_path / "machine.json"
    _write_json(machine, {"format": "molgap-machine-paths-v1", "python": sys.executable,
        **{key: str(tmp_path / key) for key in ("artifact_root", "scratch_root", "data_root")}})
    question = ExecutionQuestion(policy["policy_id"], policy["action_rule"]["action"],
        "Check model workflow", "Unverified workflow", "readout", ("evidence-common-test",),
        ("No effect",), ("fixture-closed",), "Review evidence", "Retain evidence")
    value = dict(format="molgap-scientific-experiment-v1", experiment_id="configured-fixture",
        logical_run_id="configured-fixture-run", attempt_id="v1", machine_profile=str(machine),
        authority_ref=authority_ref, monitor_paths=["experiments/question/STATUS.md"],
        timestamp="2026-10-19T00:00:00Z", question=asdict(question),
        profile=asdict(ExecutionProfile("server", "C", "kaggle", "CPU synthetic fixture", "metadata only", None, None)),
        platform_ref=platform_ref, role_ref="experiments/reference/roles.json", decision_ref="experiments/reference/decision.md",
        seeds=[42], arms=[dict(arm_id="candidate", model=dict(family=family[0], version=family[1], addons=[
            dict(name=("ppgt" if family[0] == "ppgt_small" else "esa") + "_readout_scale", version="1", config=None)]),
            recipe_ref=recipe_ref, data_ref=dict(spec_ref=spec_ref, arm_id="reference"), initial_state_ref=str(case.initial),
            training=dict(horizon_passes=8, learning_rate=.0008), reference_ref=dict(plan_ref=plan_ref, arm_id="reference"),
            purpose="context_only")])
    path = repo / "experiments/question/experiment.json"
    _write_json(path, value)
    _git(repo, "init")
    _git(repo, "config", "user.name", "CPU fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic configured fixture")
    return SimpleNamespace(repo=repo, path=path, value=value, authority=authority, initial=case.initial)


def test_configured_compile_prepare_and_cross_head_reuse(configured, monkeypatch):
    c = configured
    value, spec, workflow, paths = resolve_experiment_input(c.path, c.repo)
    plan = json.loads((c.repo / workflow["acceptance_plan"]).read_bytes())
    assert plan["format"] == CONFIGURED_PLAN_FORMAT
    assert check_plan(spec, c.repo, plan)["comparison_scope"] == "CONTEXT_ONLY"
    _, recipe = read_recipe(c.repo / workflow["arms"][0]["recipe"])
    assert recipe.mode == "composed" and recipe.optimizer_steps == 6248
    assert recipe.learning_rate == .0008 and recipe.formal_training.configured
    # Source identity checks require imports from the fixture checkout itself.
    program = r'''
import sys, pathlib, subprocess, json
repo=pathlib.Path(sys.argv[1]); path=pathlib.Path(sys.argv[2]); sys.path.insert(0,str(repo/'src'))
from molgap.experiment_input import resolve_experiment_input, prepare_experiment_input
from molgap.family_training_runner import read_recipe
from molgap.experiment_family_workflow import RunContext
from molgap.modern2d_training_release import build_formal_context, execution_extension
import molgap.kaggle_submission as platform
def forbidden(*a,**k): raise AssertionError('remote call or recompile')
platform.submit_prepared_workflow=forbidden
value,spec,workflow,paths=resolve_experiment_input(path,repo)
_,recipe=read_recipe(repo/workflow['arms'][0]['recipe'])
prepared=prepare_experiment_input(path,repo)
assert prepared['status']=='PREPARED_FOR_PLATFORM',prepared
assert prepared['submitted'] is False
context=RunContext.for_training(spec,pathlib.Path(prepared['prepared'])/'package',
    expected_package_identity=prepared['package_identity'],arm_id='candidate',account='fixture',run_reference='fixture/configured')
native,authority_path=build_formal_context(spec=spec,job=workflow['arms'][0],source_root=repo,
    staged_root=pathlib.Path(prepared['prepared']),input_root=repo/'experiments/reference',run_context=context)
native.validate_before_payload(recipe=recipe,context=context,spec=spec,authority_path=authority_path,
    input_root=repo/'experiments/reference')
assert native.configured and native.max_wall_seconds==1800
assert execution_extension(spec,repo).wall_ceiling_seconds==1800
(repo/'unrelated.md').write_text('unrelated\n')
subprocess.run(['git','add','unrelated.md'],cwd=repo,check=True,capture_output=True)
subprocess.run(['git','-c','commit.gpgsign=false','commit','-m','Unrelated docs'],cwd=repo,check=True,capture_output=True)
import molgap.experiment_input as owner
owner._compile_scientific_input=forbidden
initial=pathlib.Path(value['arms'][0]['initial_state_ref'])
initial.rename(initial.with_suffix('.retained'))
reused=prepare_experiment_input(path,repo)
assert reused['reused_preparation'] and reused['package_identity']==prepared['package_identity']
print(json.dumps({'status':'CPU_ACCEPTED','package_identity':prepared['package_identity'],'remote_calls':0}))
'''
    result = subprocess.run([sys.executable, "-I", "-c", program, str(c.repo), str(c.path)], cwd=c.repo,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def test_configured_authority_rejects_scope_changes(configured):
    from molgap.modern2d_training_release import validate_configured_authority
    c = configured
    _, spec, workflow, _ = resolve_experiment_input(c.path, c.repo)
    recipes = {j["arm_id"]: read_recipe(c.repo / j["recipe"])[1] for j in workflow["arms"]}
    for field, value in [("protected_role_access_authorized", True), ("allocated_t4_hours_ceiling", .5),
            ("format", "molgap-modern2d-formal-training-authority-v1"),
            ("roles", ["train_500k", "common_dev"]), ("recipes", {})]:
        with pytest.raises(ValueError):
            validate_configured_authority({**c.authority, field: value}, spec.to_dict(), recipes)
    # A new scale is an explicit contract, never a decision from the 100K MAE.
    original = recipes["candidate"]
    steps = 8 * (500000 // 128)
    larger = replace(original, train_rows=500000, optimizer_steps=steps,
        schedule=replace(original.schedule, total_units=steps),
        formal_training=replace(original.formal_training, horizon_steps=steps))
    authority = {**c.authority, "roles": ["train_500k", "common_dev"], "scale_500k_authorized": True,
        "recipes": {"candidate": {"horizon_passes": 8, "horizon_steps": steps}}}
    assert validate_configured_authority(authority, spec.to_dict(), {"candidate": larger}) == 1800


def test_configured_metadata_tamper_is_rejected(configured):
    c = configured
    _, spec, workflow, _ = resolve_experiment_input(c.path, c.repo)
    plan = json.loads((c.repo / workflow["acceptance_plan"]).read_bytes())
    metadata = c.repo / plan["inputs"]["path"]
    metadata.write_bytes(metadata.read_bytes() + b" ")
    with pytest.raises(ValueError, match="metadata changed"):
        check_plan(spec, c.repo, plan)


def test_configured_two_independent_arms(configured):
    c = configured
    value = deepcopy(c.value)
    second = deepcopy(value["arms"][0])
    second.update(arm_id="second", training={"learning_rate": .0006, "horizon_passes": 8})
    value["arms"].append(second)
    authority = deepcopy(c.authority)
    authority["arms"].append("second")
    authority["interfaces"].append(deepcopy(authority["interfaces"][0]))
    authority["recipes"]["second"] = deepcopy(authority["recipes"]["candidate"])
    _write_json(c.repo / value["authority_ref"], authority)
    _write_json(c.path, value)
    _git(c.repo, "add", ".")
    _git(c.repo, "-c", "commit.gpgsign=false", "commit", "-m", "Independent paired arms")
    _, spec, workflow, _ = resolve_experiment_input(c.path, c.repo)
    assert [job["device"] for job in workflow["arms"]] == [0, 1]
    assert len({job["recipe"] for job in workflow["arms"]}) == 2
    identities = [read_recipe(c.repo / job["recipe"])[1].identity for job in workflow["arms"]]
    assert len(set(identities)) == 2
    plan = json.loads((c.repo / workflow["acceptance_plan"]).read_bytes())
    assert len(check_plan(spec, c.repo, plan)["arms"]) == 2
