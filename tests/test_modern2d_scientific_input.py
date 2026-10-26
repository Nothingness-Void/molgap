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
    for name in ("platforms/kaggle/run_experiment.py", "platforms/kaggle/submit_workflow.py",
                 "experiments/.gitattributes", "research_memory/policies/.gitattributes"):
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
    recipe = replace(case.recipe, train_rows=100000, development_rows=50000, optimizer_steps=15620,
        schedule=replace(case.recipe.schedule, total_units=15620, reference_epochs=20),
        formal_training=Modern2DFormalTrainingProfile(purpose="configured-screen",
            horizon_passes=20, horizon_steps=15620),
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
        same_run_reference_arm_id="reference", arms=["reference", "candidate"],
        interfaces=[list(family), list(family)], seeds=[42], physical_batch_size=128,
        roles=["train_100k", "common_dev"], scale_500k_authorized=False,
        protected_role_access_authorized=False, extra_seed_authorized=False, automatic_successor_authorized=False,
        full_training_authorized=False, allocated_t4_hours_ceiling=1, single_arm_reason="CPU fixture for one family",
        recipes=dict(reference=dict(horizon_passes=20, horizon_steps=15620),
            candidate=dict(horizon_passes=20, horizon_steps=15620)),
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
        seeds=[42], same_run=dict(reference_arm_id="reference", target_manifest_ref=target_ref),
        arms=[dict(arm_id="reference", model=dict(family=family[0], version=family[1], addons=[]),
            recipe_ref=recipe_ref, data_ref=dict(spec_ref=spec_ref, arm_id="reference"), initial_state_ref=str(case.initial),
            training={}, reference_ref=dict(plan_ref=plan_ref, arm_id="reference"),
            purpose="architecture_comparison"),
            dict(arm_id="candidate", model=dict(family=family[0], version=family[1], addons=[
                dict(name=("ppgt" if family[0] == "ppgt_small" else "esa") + "_readout_scale", version="1", config=None)]),
            recipe_ref=recipe_ref, data_ref=dict(spec_ref=spec_ref, arm_id="reference"), initial_state_ref=str(case.initial),
            training=dict(horizon_passes=20, learning_rate=.001), reference_ref=dict(plan_ref=plan_ref, arm_id="reference"),
            purpose="architecture_comparison")])
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
    checked = check_plan(spec, c.repo, plan)
    assert checked["comparison_scope"] == "STRICT_CAUSAL" and not checked["noReplay"]
    assert checked["reference_status"] == "SAME_RUN_REFERENCE_PLANNED"
    assert plan["comparison"]["declared_intervention_fields"] == ["architecture_config_identity"]
    _, recipe = read_recipe(c.repo / workflow["arms"][0]["recipe"])
    assert recipe.mode == "reference" and recipe.optimizer_steps == 15620
    assert recipe.learning_rate == .001 and recipe.formal_training.configured
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
job=next(item for item in workflow['arms'] if item['arm_id']=='candidate')
_,recipe=read_recipe(repo/job['recipe'])
prepared=prepare_experiment_input(path,repo)
assert prepared['status']=='PREPARED_FOR_PLATFORM',prepared
assert prepared['submitted'] is False
context=RunContext.for_training(spec,pathlib.Path(prepared['prepared'])/'package',
    expected_package_identity=prepared['package_identity'],arm_id='candidate',account='fixture',run_reference='fixture/configured')
native,authority_path=build_formal_context(spec=spec,job=job,source_root=repo,
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
    steps = 8 * (500000 // 128)
    larger = {name: replace(recipe, train_rows=500000, optimizer_steps=steps,
        schedule=replace(recipe.schedule, total_units=steps, reference_epochs=8),
        formal_training=replace(recipe.formal_training, horizon_passes=8, horizon_steps=steps))
        for name, recipe in recipes.items()}
    authority = {**c.authority, "roles": ["train_500k", "common_dev"], "scale_500k_authorized": True,
        "recipes": {name: {"horizon_passes": 8, "horizon_steps": steps} for name in larger}}
    assert validate_configured_authority(authority, spec.to_dict(), larger) == 1800


def test_configured_metadata_tamper_is_rejected(configured):
    c = configured
    _, spec, workflow, _ = resolve_experiment_input(c.path, c.repo)
    plan = json.loads((c.repo / workflow["acceptance_plan"]).read_bytes())
    metadata = c.repo / plan["inputs"]["path"]
    metadata.write_bytes(metadata.read_bytes() + b" ")
    with pytest.raises(ValueError, match="metadata changed"):
        check_plan(spec, c.repo, plan)


def test_configured_independent_arms_require_a_frozen_pair(configured):
    c = configured
    value = deepcopy(c.value)
    value.pop("same_run")
    _write_json(c.path, value)
    with pytest.raises(ValueError, match="same-run or retained frozen reference"):
        resolve_experiment_input(c.path, c.repo)


def test_transfer_study_uses_exact_source_clock_without_causal_pair(configured):
    c = configured
    value = deepcopy(c.value)
    value.pop("same_run")
    value["arms"] = [value["arms"][0]]
    value["arms"][0].update(purpose="transfer_study",
        training=dict(horizon_passes=20, horizon_steps=15620, evaluation_every_steps=781))
    old_path = c.repo / value["arms"][0]["recipe_ref"]
    old_record, old = read_recipe(old_path)
    source_spec_copy = c.repo / "experiments/reference/source100k_spec.json"
    _write_json(source_spec_copy, json.loads((c.repo / value["arms"][0]["data_ref"]["spec_ref"]).read_bytes()))
    old_copy = c.repo / "experiments/reference/source100k.json"
    _write_json(old_copy, old_record)
    study_path = c.repo / "experiments/question/transfer.json"
    _write_json(study_path, dict(format="molgap-modern2d-transfer-plan-v1", experiment_purpose="transfer_study",
        arms={"reference": dict(source_contract=_pointer(c.repo, old_copy))}))
    binding_path = c.repo / "experiments/reference/training_input.json"
    binding = json.loads(binding_path.read_bytes())
    train, train_record = permit_record("train_500k")
    binding["train"] = train_record
    _write_json(binding_path, binding)
    new = replace(old, train_rows=500000,
        dataset=replace(old.dataset, manifest_sha256=file_digest(binding_path)),
        formal_training=replace(old.formal_training, horizon_passes=4, evaluation_every_steps=781))
    record = build_screen_recipe(new.mode, configuration=new, family=new.interface,
        source_idx_sha256="2" * 64, target_sha256="3" * 64,
        train_role_identity="c" * 64, development_role_identity="c" * 64)
    _write_json(old_path, record)
    spec_path = c.repo / value["arms"][0]["data_ref"]["spec_ref"]
    payload = json.loads(spec_path.read_bytes())
    arm = payload["arms"][0]
    arm["data"]["dataset"]["sha256"] = new.dataset.manifest_sha256
    arm["training"]["recipe"]["sha256"] = file_digest(old_path)
    bindings = family_arm_bindings(new)
    arm["base"] = bindings["base"]
    arm["training"].update({k: bindings[k] for k in ("sampler", "transform", "objective")})
    source = ExperimentSpec(payload)
    _write_json(spec_path, source.to_dict())
    metadata_path = c.repo / "experiments/reference/training_inputs.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata["training_input"]["sha256"] = new.dataset.manifest_sha256
    model = metadata["models"]["reference"]
    feature = model["feature_bindings"].pop("train_100k")
    model["feature_binding_sha256"].pop("train_100k")
    feature.update(role="train_500k", base_dataset_manifest_sha=train.manifest_sha256,
        role_history_sha=train.role_history_sha256)
    model["feature_bindings"]["train_500k"] = feature
    model["feature_binding_sha256"]["train_500k"] = canonical_fingerprint(feature)
    arm["data"]["feature_sha256"] = canonical_fingerprint(model["feature_binding_sha256"])
    _write_json(spec_path, ExperimentSpec(payload).to_dict())
    _write_json(metadata_path, metadata)
    prior_path = c.repo / value["arms"][0]["reference_ref"]["plan_ref"]
    prior = json.loads(prior_path.read_bytes())
    prior.update(spec_identity=ExperimentSpec(payload).identity, inputs=_pointer(c.repo, metadata_path))
    prior["arms"][0]["contract"] = _pointer(c.repo, old_path)
    _write_json(prior_path, prior)
    authority = {**c.authority, "purpose": "CONFIGURED_TRANSFER_STUDY", "arms": ["reference"],
        "interfaces": [list(new.interface)], "roles": ["train_500k", "common_dev"],
        "scale_500k_authorized": True, "recipes": {"reference": dict(horizon_passes=4, horizon_steps=15620)},
        "transfer_study_ref": _pointer(c.repo, study_path)}
    authority.pop("same_run_reference_arm_id")
    _write_json(c.repo / value["authority_ref"], authority)
    _write_json(c.path, value)
    _git(c.repo, "add", ".")
    _git(c.repo, "-c", "commit.gpgsign=false", "commit", "-m", "Freeze noncausal scale fixture")
    _, spec, workflow, _ = resolve_experiment_input(c.path, c.repo)
    plan = json.loads((c.repo / workflow["acceptance_plan"]).read_bytes())
    result = check_plan(spec, c.repo, plan)
    assert result["comparison_scope"] == "CONTEXT_ONLY" and result["noReplay"]
    assert "comparison" not in plan and "same_run_replay" not in spec.to_dict()["prospective"]
    _, compiled = read_recipe(c.repo / plan["arms"][0]["contract"]["path"])
    assert compiled.optimizer_steps == old.optimizer_steps
    assert compiled.schedule == old.schedule
    assert compiled.formal_training.evaluation_every_steps == 781
    assert (compiled.loop_plan.steps_per_pass, plan["arms"][0]["expected"]["next_batch"]) == (3906, 3902)
    # The scale anchor uses this same compiler contract; it never enrolls a
    # trained reference or reads the fake fixture's absent graph mount.
    from molgap.modern2d_research_prepare import freeze_scale_anchor
    qualification = deepcopy(metadata)
    qualification.update(status="INPUT_METADATA_PREPARED",
        roles={role: dict(rows=rows, source_idx_tensor_sha256="2" * 64)
               for role, rows in (("train_500k", 500000), ("common_dev", 50000))})
    model_id = "ppgt_author_v2" if new.family == "ppgt_small" else "esa_native_v3"
    qualification["models"] = {model_id: qualification["models"]["reference"]}
    qualification_path = c.repo / "experiments/reference/qualification.json"
    _write_json(qualification_path, qualification)
    anchored = freeze_scale_anchor(c.repo, base="experiments/scale_anchor",
        source_spec_ref=source_spec_copy.relative_to(c.repo).as_posix(),
        source_contract_ref=old_copy.relative_to(c.repo).as_posix(), source_arm_id="reference",
        accepted_metadata_ref=qualification_path.relative_to(c.repo).as_posix(),
        accepted_input_ref=binding_path.relative_to(c.repo).as_posix(),
        target_manifest=prior["arms"][0]["target_manifest"])
    assert not anchored["graph_payload_read"] and not anchored["scientific_reference_accepted"]
    anchor_metadata = json.loads((c.repo / anchored["pins"]["training_inputs.json"]["path"]).read_bytes())
    assert anchor_metadata["training_input"] == anchored["pins"]["training_input.json"]
    _, anchor_recipe = read_recipe(c.repo / anchored["pins"]["contract.json"]["path"])
    assert anchor_recipe.train_rows == 500000 and anchor_recipe.schedule == old.schedule
    # Exercise the owning platform's final release helper against the actual
    # prepared archive, rather than treating compiler success as submission.
    # As in the configured journey above, package validators must import from
    # the actual checkout being released, not the parent pytest checkout.
    program = r'''
import sys, pathlib, json
repo=pathlib.Path(sys.argv[1]); path=pathlib.Path(sys.argv[2]); sys.path.insert(0,str(repo/'src'))
from molgap.experiment_input import prepare_experiment_input
from molgap.experiment_spec import ExperimentSpec
from molgap.kaggle_submission import _check_scientific_release
from molgap.experiment_package import verify_experiment_source_package
result=prepare_experiment_input(path,repo)
assert result['status']=='PREPARED_FOR_PLATFORM',result
prepared=pathlib.Path(result['prepared'])
manifest=verify_experiment_source_package(prepared/'package')
spec=ExperimentSpec.from_json((prepared/'package/experiment_spec.json').read_text(encoding='utf-8'))
authority=json.loads((repo/'experiments/question/authority.json').read_bytes())
_check_scientific_release(spec,prepared/'package',prepared/'source_dataset',authority,manifest)
# Execute the actual cloud entry's stdlib admission against the extracted
# frozen archive: local release alone does not exercise startup routing.
import runpy
from molgap.experiment_preflight import _unpack
source=prepared/'bootstrap_source'
source.mkdir()
_unpack(prepared/'package',source)
startup=runpy.run_path(str(prepared/'kernel/run.py'))['_modern2d_wall_ceiling']
assert startup(spec.to_dict(),source)==authority['gpu']['wall_ceiling_seconds']
forged={**authority,'transfer_study_ref':{**authority['transfer_study_ref'],'sha256':'0'*64}}
try:
    _check_scientific_release(spec,prepared/'package',prepared/'source_dataset',forged,manifest)
except ValueError:
    pass
else:
    raise AssertionError('Forged transfer pointer passed release')
print(json.dumps({'status':'CPU_ACCEPTED','package_identity':manifest['package_identity'],'remote_calls':0}))
'''
    result = subprocess.run([sys.executable, "-I", "-c", program, str(c.repo), str(c.path)], cwd=c.repo,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def test_execution_repair_reissues_only_source_pin(configured):
    from molgap.experiment_input import _read_scientific_recipe
    c = configured
    path = c.repo / c.value["arms"][0]["recipe_ref"]
    original = json.loads(path.read_bytes())
    family = (original["family"], original["family_version"])
    original["execution_policy"]["feature_transform_source_sha256"] = "0" * 64
    _write_json(path, original)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Execution policy differs"):
        read_recipe(path)
    authority = dict(purpose="CONFIGURED_EXECUTION_REPAIR", scientific_training_authorized=False)
    retained, reissued = _read_scientific_recipe(path, family, authority)
    assert retained == original and path.read_bytes() == before
    assert reissued.execution_policy["feature_transform_source_sha256"] == indexed_transform_identity(
        family, feature_only=True)
    for denied in ({**authority, "purpose": "CONFIGURED_SCREEN"},
                   {**authority, "scientific_training_authorized": True},
                   {"purpose": "CONFIGURED_EXECUTION_REPAIR"}):
        with pytest.raises(ValueError, match="Execution policy differs"):
            _read_scientific_recipe(path, family, denied)
    # Source reissuance must not repair a second, unrelated policy discrepancy.
    original["execution_policy"]["unreviewed_policy"] = True
    _write_json(path, original)
    with pytest.raises(ValueError, match="Execution policy differs"):
        _read_scientific_recipe(path, family, authority)
