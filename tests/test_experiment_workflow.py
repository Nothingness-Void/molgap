"""Shared workflow orchestration checks using synthetic, local-only inputs."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from molgap import experiment_cli as cli
from molgap import experiment_prospective as prospective
from molgap import experiment_preflight as preflight
from molgap import experiment_source_inventory as source_inventory
from molgap import experiment_workflow as workflow
from molgap import kaggle_workflow as kaggle_backend
from molgap.experiment_execution import (
    build_family_recipe,
    check_family_recipes,
    validate_execution_plan,
    validate_staged_trajectory,
)
from molgap.experiment_family_workflow import (
    RunContext,
    build_incomplete_terminal_descriptor,
)
from molgap.experiment_launch import canonical_json
from molgap.experiment_preflight import check_workflow_binding
from molgap.experiment_spec import (
    FAMILIES,
    ExperimentSpec,
    SCHEMA_VERSION_V2,
    TERMINAL_PROTOCOL,
)
from molgap.screen_policy import canonical_fingerprint
from molgap.v4_runtime import normalized_source_sha256


ROOT = Path(__file__).resolve().parents[1]
HEX_A, HEX_B, HEX_C = "a" * 64, "b" * 64, "c" * 64
SOURCE_IDX_SHA256, TARGET_SHA256 = "2" * 64, "3" * 64


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return canonical_json(value).encode("utf-8")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _ref(name: str, digest: str = HEX_A) -> dict:
    return {"name": name, "version": "1", "sha256": digest}


def _arm(
    family_name: str,
    version: str,
    arm_id: str,
    *,
    recipe_sha256: str = HEX_A,
    state_sha256: str = HEX_C,
    sampler_sha256: str = HEX_B,
    addon: str | None = None,
    addon_sha256: str = HEX_C,
) -> dict:
    contract = FAMILIES[(family_name, version)]
    addons = []
    if addon == "k1_joint_aggregation":
        config = {
            "layer": 6,
            "latent_channels": 64,
            "kappa": 4,
            "seed": 42,
            "degree_policy": "original-sum-above-four",
        }
        addons = [{"name": addon, "version": "1", "config": config,
                   "source_sha256": addon_sha256}]
    elif addon is not None:
        addons = [{"name": addon, "version": "1", "config": {},
                   "source_sha256": addon_sha256}]
    return {
        "arm_id": arm_id,
        "scientific_role": "candidate",
        "family": {"name": family_name, "version": version},
        "base": _ref("synthetic-base"),
        "initialization": {
            "kind": "frozen_state", "seed": 42, "state_sha256": state_sha256,
        },
        "data": {
            "dataset": _ref("pcqm4mv2"),
            "split": _ref("synthetic-split"),
            "roles": [
                {"role": role, "membership_sha256": HEX_A,
                 "row_order_sha256": HEX_B, "usage_sha256": HEX_C}
                for role in contract.roles
            ],
            "feature_schema": contract.feature_schema,
            "feature_sha256": HEX_A,
            "target": "pcqm4mv2-gap-eV-direct",
        },
        "training": {
            "recipe": _ref(contract.recipe, recipe_sha256),
            "overrides": {},
            "objective": _ref("normalized-gap-l1"),
            "sampler": _ref(contract.sampler, sampler_sha256),
            "transform": _ref(contract.transform),
        },
        "addons": addons,
        "addon_semantics": "ordered" if addons else "baseline",
    }


def _spec_payload(arms: list[dict], *, platform: str = "local") -> dict:
    return {
        "schema_version": SCHEMA_VERSION_V2,
        "experiment_id": "synthetic-workflow-question",
        "logical_run_id": "synthetic-workflow-run",
        "arms": arms,
        "platform": {
            "name": platform,
            "accelerator": "Tesla T4" if platform == "kaggle" else "synthetic-cpu",
            "device_count": 2 if platform == "kaggle" else 1,
            "cpu_cores": 8,
            "memory_gib": 32,
            "atomic_checkpoints": True,
            "retrievable_chunks": True,
        },
        "prospective": {
            "arms": [
                {
                    "arm_id": arm["arm_id"],
                    "trajectory_id": "trajectory-" + arm["arm_id"],
                    "plan_spec_ref": "plans/" + arm["arm_id"] + ".json",
                    "plan_spec_sha256": HEX_C,
                    "output": "experiments/prospective-" + arm["arm_id"],
                }
                for arm in arms
            ]
        },
        "evidence": {
            "policy": _ref("molgap-v5"),
            "required_artifacts": [
                "v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact",
            ],
        },
        "terminal_protocol": TERMINAL_PROTOCOL,
    }


def _candidate_pair_spec() -> ExperimentSpec:
    arms = [
        _arm("neural_atom_k1", "2", "k1_candidate",
             addon="k1_joint_aggregation"),
        _arm("gptrans_t", "1", "gptrans_candidate", addon="pair_prenorm"),
    ]
    return ExperimentSpec(_spec_payload(arms, platform="kaggle"))


def _jobs(spec: ExperimentSpec) -> list[dict]:
    bindings = {row["arm_id"]: row for row in spec.to_dict()["prospective"]["arms"]}
    return [
        {
            "arm_id": arm["arm_id"],
            "device": index,
            "recipe": f"recipes/{arm['arm_id']}.json",
            "initial_state": f"initial_states/{arm['arm_id']}.pt",
            "trajectory_id": bindings[arm["arm_id"]]["trajectory_id"],
        }
        for index, arm in enumerate(spec.to_dict()["arms"])
    ]


def test_v2_two_candidate_arms_bind_supported_k1_and_gptrans_adapters():
    spec = _candidate_pair_spec()
    declaration = spec.to_dict()

    assert [arm["scientific_role"] for arm in declaration["arms"]] == ["candidate", "candidate"]
    assert "same_run_replay" not in declaration["prospective"]
    assert len(declaration["prospective"]["arms"]) == 2

    selected = validate_execution_plan(spec, _jobs(spec))

    assert [(job["module"], job["mode"]) for job in selected] == [
        ("molgap.k1_screen_training", "ssma"),
        ("molgap.gptrans_screen_workflow", "pair_prenorm"),
    ]


@pytest.mark.parametrize(
    ("family", "version"),
    [("neural_atom_k1", "1"), ("edge_state_gps", "1")],
)
def test_declared_only_k1_v1_and_edgestate_fail_closed(family, version):
    spec = ExperimentSpec(_spec_payload([_arm(family, version, "declared_candidate")]))

    with pytest.raises(ValueError, match="no training adapter"):
        validate_execution_plan(spec, _jobs(spec))


def test_execution_plan_rejects_duplicate_jobs_and_escaped_paths():
    spec = _candidate_pair_spec()
    jobs = _jobs(spec)
    duplicate = [jobs[0], copy.deepcopy(jobs[0])]
    with pytest.raises(ValueError, match="Unknown or duplicate arm"):
        validate_execution_plan(spec, duplicate)

    for field in ("recipe", "initial_state"):
        escaped = _jobs(spec)
        escaped[0][field] = "../outside.pt"
        with pytest.raises(ValueError, match="confined relative POSIX execution path"):
            validate_execution_plan(spec, escaped)


def test_execution_plan_rejects_wrong_trajectory_binding():
    spec = _candidate_pair_spec()
    jobs = _jobs(spec)
    jobs[1]["trajectory_id"] = "trajectory-from-another-arm"

    with pytest.raises(ValueError, match="trajectory mismatch"):
        validate_execution_plan(spec, jobs)


def test_staged_trajectory_identity_mismatch_fails_before_schema_translation(tmp_path):
    spec = _candidate_pair_spec()
    arm = spec.to_dict()["arms"][0]
    path = tmp_path / "prospective" / arm["arm_id"] / "trajectory.json"
    _write_json(path, {
        "record_mode": "prospective",
        "trajectory_id": "wrong-trajectory",
        "state_at_start": {"source_config_identity": canonical_fingerprint(arm)},
    })

    with pytest.raises(ValueError, match="trajectory/arm identity mismatch"):
        validate_staged_trajectory(spec, arm["arm_id"], tmp_path)


def _process_spec(family: str, version: str, mode: str, owner, recipe: dict) -> ExperimentSpec:
    addon = None if mode == "reference" else (
        "k1_joint_aggregation" if family == "neural_atom_k1" else mode
    )
    source_sha = "d" * 64
    if addon:
        module = (
            "k1_joint_aggregation.py" if family == "neural_atom_k1"
            else owner.VARIANTS[mode]
        )
        source_sha = normalized_source_sha256(Path(owner.__file__).with_name(module))
    state_sha = (
        owner.INITIAL_STATE_SHA256 if family == "neural_atom_k1"
        else owner.owner.EXPECTED_INITIAL_MODEL_SHA256
    )
    arm = _arm(
        family, version, "recipe_candidate",
        recipe_sha256=_sha(_json_bytes(recipe)),
        state_sha256=state_sha,
        sampler_sha256=recipe["row_order_fingerprint"],
        addon=addon,
        addon_sha256=source_sha,
    )
    return ExperimentSpec(_spec_payload([arm]))


@pytest.mark.parametrize(
    ("family", "version", "mode"),
    [
        ("neural_atom_k1", "2", "reference"),
        ("neural_atom_k1", "2", "ssma"),
        ("gptrans_t", "1", "reference"),
        ("gptrans_t", "1", "pair_prenorm"),
        ("gptrans_t", "1", "centered_logits"),
        ("gptrans_t", "1", "memory_value"),
        ("gptrans_t", "1", "memory_message"),
    ],
)
def test_family_recipe_builder_and_validator_are_metadata_only(family, version, mode):
    family_key = (family, version)
    addon = None if mode == "reference" else (
        "k1_joint_aggregation" if family == "neural_atom_k1" else mode
    )
    recipe = build_family_recipe(
        family_key, addon=addon,
        source_idx_sha256=SOURCE_IDX_SHA256, target_sha256=TARGET_SHA256,
    )
    spec = _process_spec(family, version, mode,
        __import__("molgap.k1_screen_training" if family == "neural_atom_k1"
                   else "molgap.gptrans_screen_workflow", fromlist=["*"]), recipe)
    owner = __import__(
        "molgap.k1_screen_training" if family == "neural_atom_k1"
        else "molgap.gptrans_screen_workflow", fromlist=["*"],
    )

    result = owner.validate_screen_recipe(spec, "recipe_candidate", recipe)

    assert result["recipe"] == "frozen_recipe_verified"
    assert result["mode"] == mode


def test_check_family_recipes_rejects_legacy_gptrans_contract_and_hash_tampering():
    recipe_path = ROOT / "experiments" / "pcqm_gptrans_t_100k_v4" / "training_contract.json"
    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    from molgap import gptrans_screen_workflow as gptrans

    arm = _arm(
        "gptrans_t", "1", "frozen_gptrans",
        recipe_sha256=normalized_source_sha256(recipe_path),
        state_sha256=gptrans.owner.EXPECTED_INITIAL_MODEL_SHA256,
        sampler_sha256=recipe["row_order_fingerprint"],
    )
    spec = ExperimentSpec(_spec_payload([arm]))
    relative_recipe = recipe_path.resolve().relative_to(ROOT.resolve()).as_posix()

    with pytest.raises(ValueError, match="requires frozen acceptance_requirements"):
        check_family_recipes(spec, ROOT, {"frozen_gptrans": relative_recipe})

    changed = spec.to_dict()
    changed["arms"][0]["training"]["recipe"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recipe/Spec hash mismatch"):
        check_family_recipes(ExperimentSpec(changed), ROOT,
                             {"frozen_gptrans": relative_recipe})


@pytest.fixture
def workflow_case(tmp_path):
    repo = tmp_path / "synthetic-repo"
    repo.mkdir()
    source_names = sorted(source_inventory.SHARED_SOURCE_FILES)
    for name in source_names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            b"EXPECTED_LAUNCH_SHA256 = None\n"
            if name == "platforms/kaggle/run_experiment.py"
            else b"# synthetic source package input\n"
        )
        path.write_bytes(content)

    expected = {
        "epochs": 2,
        "optimizer_steps": 4,
        "sample_presentations": 8,
        "development_rows": 10,
        "source_idx_sha256": SOURCE_IDX_SHA256,
        "target_sha256": TARGET_SHA256,
        "precision": "fp32",
    }
    recipes, initial_states, arms = {}, {}, []
    for index, (family, version, arm_id, addon) in enumerate([
        ("neural_atom_k1", "2", "k1_candidate", "k1_joint_aggregation"),
        ("gptrans_t", "1", "gptrans_candidate", "pair_prenorm"),
    ]):
        recipe_path = repo / "recipes" / f"{arm_id}.json"
        recipe = {
            "format": "synthetic-screen-recipe-v1",
            "development_role_identity": "synthetic-development-role",
            "acceptance_requirements": expected,
        }
        recipe_bytes = _json_bytes(recipe)
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_path.write_bytes(recipe_bytes)
        recipes[arm_id] = recipe_path.relative_to(repo).as_posix()

        state_path = tmp_path / f"{arm_id}.pt"
        torch.save({"weight": torch.tensor([float(index + 1)])}, state_path)
        state_sha = _sha(state_path.read_bytes())
        initial_states[arm_id] = state_path
        arms.append(_arm(
            family, version, arm_id,
            recipe_sha256=_sha(recipe_bytes), state_sha256=state_sha,
            addon=addon,
        ))

    _git(repo, "init")
    _git(repo, "config", "user.email", "synthetic@example.invalid")
    _git(repo, "config", "user.name", "Synthetic workflow fixture")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", "--", *source_names, *recipes.values())
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic workflow sources")

    payload = _spec_payload(arms, platform="kaggle")
    for arm, binding in zip(payload["arms"], payload["prospective"]["arms"]):
        plan_value = {
            "trajectory": {
                "trajectory_id": binding["trajectory_id"],
                "state_at_start": {"source_config_identity": canonical_fingerprint(arm)},
            },
            "decision_state": {},
        }
        plan_path = repo / binding["plan_spec_ref"]
        plan_bytes = _json_bytes(plan_value)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_bytes(plan_bytes)
        binding["plan_spec_sha256"] = _sha(plan_bytes)
    spec = ExperimentSpec(payload)

    acceptance_plan = _write_strict_retained_reference_plan(repo, spec, expected, recipes)
    plan = {
        "format": "molgap-experiment-workflow-v1",
        "spec_identity": spec.identity,
        "source_files": [],
        "arms": [
            {
                "arm_id": arm["arm_id"],
                "device": index,
                "recipe": recipes[arm["arm_id"]],
                "initial_state": str(initial_states[arm["arm_id"]]),
            }
            for index, arm in enumerate(spec.to_dict()["arms"])
        ],
        "acceptance_plan": acceptance_plan,
        "kaggle": {
            "account": "synthetic-account",
            "kernel": "synthetic-account/synthetic-pair-run",
            "title": "Synthetic Pair Run",
            "datasets": ["synthetic-account/source-data"],
            "source_dataset": "synthetic-account/source-data",
            "accelerator": "NvidiaTeslaT4",
        },
    }
    return {
        "repo": repo,
        "spec": spec,
        "plan": plan,
        "expected": expected,
        "recipes": recipes,
        "initial_states": initial_states,
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


def _pointer(root: Path, path: Path) -> dict:
    return {"path": path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": _sha(path.read_bytes())}


def _write_strict_retained_reference_plan(repo: Path, spec: ExperimentSpec,
                                          expected: dict, recipes: dict) -> str:
    from test_comparison_readiness import _prelaunch, _reference_bundle
    from molgap.comparison_readiness import REQUIRED_REFERENCE_ARTIFACTS

    bundle = copy.deepcopy(_reference_bundle())
    prediction_bytes = _json_bytes({
        "synthetic": True,
        "source_idx_sha256": expected["source_idx_sha256"],
        "target_sha256": expected["target_sha256"],
        "rows": expected["development_rows"],
    })
    bundle["prediction_manifest"].update({
        "prediction_sha256": _sha(prediction_bytes),
        "source_idx_sha256": expected["source_idx_sha256"],
        "target_sha256": expected["target_sha256"],
        "row_count": expected["development_rows"],
        "unique_source_idx": expected["development_rows"],
    })

    artifact_paths = {}
    owner_fields = {
        "runtime_certificate": "runtime_certificate_ref",
        "row_manifest": "row_manifest_ref",
        "target_manifest": "target_manifest_ref",
        "trace_manifest": "trace_manifest_ref",
        "role_history": "role_history_ref",
        "target_transform_asset": "target_transform_asset_ref",
        "cost_records": "cost_records_ref",
        "acceptance": "acceptance_ref",
        "decision": "decision_ref",
    }
    for name, field in owner_fields.items():
        path = repo / bundle[field]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic retained reference decision\n" if name == "decision"
                         else _json_bytes({"synthetic": True, "artifact": name}))
        artifact_paths[name] = _pointer(repo, path)

    checkpoint = repo / "experiments/reference/checkpoint.pt"
    checkpoint.write_bytes(b"synthetic retained checkpoint\n")
    artifact_paths["checkpoint"] = _pointer(repo, checkpoint)
    prediction_manifest = repo / "experiments/reference/prediction_manifest.json"
    _write_json(prediction_manifest, bundle["prediction_manifest"])
    artifact_paths["prediction_manifest"] = _pointer(repo, prediction_manifest)
    predictions = repo / "experiments/reference/predictions.json"
    predictions.write_bytes(prediction_bytes)
    artifact_paths["predictions"] = _pointer(repo, predictions)
    assert set(artifact_paths) == REQUIRED_REFERENCE_ARTIFACTS | {"predictions"}

    bundle_path = repo / "experiments/reference/reference_bundle.json"
    _write_json(bundle_path, bundle)
    prelaunch_path = repo / "experiments/reference/comparison_prelaunch.json"
    prelaunch = _prelaunch(bundle)
    assert prelaunch["prelaunch_ready"] is True
    _write_json(prelaunch_path, prelaunch)

    adapters = {"neural_atom_k1": "k1-screen-v1", "gptrans_t": "gptrans-v1"}
    entries = []
    for arm in spec.to_dict()["arms"]:
        recipe_path = repo / recipes[arm["arm_id"]]
        entries.append({
            "arm_id": arm["arm_id"],
            "adapter": adapters[arm["family"]["name"]],
            "expected": expected,
            "contract": _pointer(repo, recipe_path),
            "comparison_prelaunch": _pointer(repo, prelaunch_path),
            "reference_bundle": _pointer(repo, bundle_path),
            "reference_artifacts": artifact_paths,
        })
    path = repo / "experiments/workflow/acceptance_plan.json"
    _write_json(path, {
        "format": "molgap-family-acceptance-plan-v1",
        "spec_identity": spec.identity,
        "arms": entries,
    })
    return path.relative_to(repo).as_posix()


def test_stage_acceptance_inputs_preserves_plan_and_all_pinned_bytes(workflow_case, tmp_path):
    from molgap.experiment_family_workflow import TargetIdentityBinding

    repo, spec = workflow_case["repo"], workflow_case["spec"]
    plan_path = workflow_case["plan"]["acceptance_plan"]
    destination = tmp_path / "staged-acceptance"
    binding = kaggle_backend.stage_acceptance_inputs(spec, repo, plan_path, destination)
    plan = json.loads((repo / plan_path).read_bytes())
    pointers = {plan_path: _sha((repo / plan_path).read_bytes())}
    for arm in plan["arms"]:
        for pointer in [arm["contract"], arm["comparison_prelaunch"], arm["reference_bundle"],
                        *arm["reference_artifacts"].values()]:
            pointers[pointer["path"]] = pointer["sha256"]
    assert binding == {"plan_path": plan_path, "plan_sha256": pointers[plan_path]}
    assert {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()} == set(pointers)
    for relative, digest in pointers.items():
        assert (destination / relative).read_bytes() == (repo / relative).read_bytes()
        assert _sha((destination / relative).read_bytes()) == digest
    assert TargetIdentityBinding.from_acceptance_plan(
        spec, destination, plan_path, plan_sha256=binding["plan_sha256"]
    ).repo_root == destination.absolute()


@pytest.mark.parametrize("phase", ["preflight", "train"])
@pytest.mark.parametrize("binding_mode", ["pinned", "omitted", "malformed", "changed_plan"])
def test_worker_validates_staged_binding_before_dispatch(
    workflow_case, tmp_path, monkeypatch, phase, binding_mode
):
    import molgap.experiment_execution as execution
    from molgap.experiment_family_workflow import TargetIdentityBinding
    from molgap.experiment_training_worker import main

    repo, spec = workflow_case["repo"], workflow_case["spec"]
    staged = tmp_path / "worker-stage"
    staged.mkdir()
    (staged / "experiment_spec.json").write_text(spec.to_json(), encoding="utf-8")
    binding = kaggle_backend.stage_acceptance_inputs(
        spec, repo, workflow_case["plan"]["acceptance_plan"], staged / "acceptance"
    )
    arm_id = spec.to_dict()["arms"][0]["arm_id"]
    config = {"jobs": [{"arm_id": arm_id}], "expected_package_identity": HEX_A,
              "account": "synthetic-account", "run_reference": "synthetic-account/workload-1",
              "prospective_sha256": {arm_id: HEX_B}}
    if binding_mode != "omitted":
        config["target_identity"] = dict(binding)
    if binding_mode == "malformed":
        config["target_identity"]["inferred_encoding"] = "float32"
    elif binding_mode == "changed_plan":
        path = staged / "acceptance" / binding["plan_path"]
        path.write_bytes(path.read_bytes() + b"\n")
    launch_path = staged / "experiment_launch.json"
    _write_json(launch_path, config)
    monkeypatch.setattr(execution, "validate_execution_plan", lambda _spec, jobs: jobs)
    dispatch = Mock(return_value={"status": "MECHANICALLY_VERIFIED" if phase == "train" else "accepted"})
    monkeypatch.setattr(execution, "execute_training_phase", dispatch)
    argv = ["--source-root", str(repo), "--package-dir", str(staged), "--input-root", str(staged),
            "--output", str(tmp_path / "output"), "--launch", str(launch_path),
            "--arm", arm_id, "--phase", phase]
    if binding_mode in {"malformed", "changed_plan"}:
        with pytest.raises(ValueError, match="binding|hash mismatch"):
            main(argv)
        dispatch.assert_not_called()
    else:
        main(argv)
        forwarded = dispatch.call_args.kwargs["target_identity"]
        if binding_mode == "omitted":
            assert forwarded is None
        else:
            assert isinstance(forwarded, TargetIdentityBinding)
            assert forwarded.repo_root == (staged / "acceptance").absolute()
            assert forwarded.plan_path == binding["plan_path"]
            assert forwarded.plan_sha256 == binding["plan_sha256"]


def test_prepare_workflow_packages_then_releases_then_plans_synthetic_candidates(
    workflow_case, tmp_path, monkeypatch,
):
    repo, spec, plan = workflow_case["repo"], workflow_case["spec"], workflow_case["plan"]
    output = tmp_path / "prepared"
    events = []
    real_package = workflow.build_experiment_source_package
    package_manifest = {}

    def package(*args, **kwargs):
        events.append("package")
        result = real_package(*args, **kwargs)
        package_manifest.update(result)
        return result

    monkeypatch.setattr(workflow, "build_experiment_source_package", package)
    monkeypatch.setattr(workflow, "check_family_recipes",
        lambda *_args: {"status": "synthetic_family_recipe_stub"})

    def release(release_spec, package_dir, **kwargs):
        events.append("release")
        assert release_spec == spec
        assert (package_dir / "package_manifest.json").is_file()
        assert "molgap.pcqm_wedge" in kwargs["required_modules"]
        assert "molgap.experiment_training_worker" in kwargs["required_modules"]
        for arm_id, recipe_path in kwargs["recipe_files"].items():
            assert _sha((repo / recipe_path).read_bytes()) == next(
                arm["training"]["recipe"]["sha256"]
                for arm in spec.to_dict()["arms"] if arm["arm_id"] == arm_id
            )
        for arm_id, state_path in kwargs["initial_states"].items():
            digest = _sha(state_path.read_bytes())
            assert digest == next(
                arm["initialization"]["state_sha256"]
                for arm in spec.to_dict()["arms"] if arm["arm_id"] == arm_id
            )
        return {
            "status": "LOCAL_RELEASE_INPUTS_VERIFIED",
            "errors": [], "inputs": {}, "checks": {}, "limitations": [],
        }

    monkeypatch.setattr(workflow, "check_release_inputs", release)
    monkeypatch.setattr(kaggle_backend, "bind_release",
        lambda *_args, **_kwargs: events.append("final_binding") or {"status": "bound"})

    def plan_many(root, plans):
        events.append("prospective")
        records = []
        for item in plans:
            trajectory = item["spec"]["trajectory"]
            destination = Path(root) / item["output"]
            destination.mkdir(parents=True)
            _write_json(destination / "trajectory.json", {
                "record_mode": "prospective",
                "trajectory_id": trajectory["trajectory_id"],
                "state_at_start": trajectory["state_at_start"],
            })
            records.append({
                "trajectory_id": trajectory["trajectory_id"],
                "status": "PLANNED", "path": item["output"],
            })
        return {"status": "PLANNED", "results": records, "batch": {"size": len(records)}}

    monkeypatch.setattr(prospective, "plan_many", plan_many)
    monkeypatch.setattr(prospective, "rebuild_research_memory",
        lambda _root: events.append("rebuild") or {"index.json": b"{}"})

    result = workflow.prepare_workflow(spec, repo, plan, output)

    assert result["status"] == "PREPARED_FOR_PLATFORM", result
    assert result["submitted"] is False
    assert events.index("package") < events.index("release") < events.index("prospective")
    assert events.index("prospective") < events.index("rebuild") < events.index("final_binding")
    assert "src/molgap/pcqm_wedge.py" in package_manifest["relative_allowlist"]
    assert (Path(result["source_dataset_dir"]) / "experiment_launch.json").is_file()
    assert (Path(result["kernel_dir"]) / "run.py").is_file()
    assert result["prospective_published"] is True


def test_release_static_failure_never_starts_prospective_planning(
    workflow_case, tmp_path, monkeypatch,
):
    repo, spec, plan = workflow_case["repo"], workflow_case["spec"], workflow_case["plan"]
    monkeypatch.setattr(workflow, "check_family_recipes", lambda *_args: {"status": "stub"})
    monkeypatch.setattr(workflow, "check_release_inputs", lambda *_args, **_kwargs: {
        "status": "RELEASE_INPUTS_FAILED", "errors": [{"check": "synthetic", "item": "fixture"}],
        "inputs": {}, "checks": {}, "limitations": [],
    })
    prospective_call = Mock()
    handoff = Mock()
    monkeypatch.setattr(workflow, "plan_prospective", prospective_call)
    monkeypatch.setattr(workflow, "build_launch_receipt", handoff)

    result = workflow.prepare_workflow(spec, repo, plan, tmp_path / "blocked")

    assert result["status"] == "BLOCKED"
    assert result["stage"] == "check_release"
    assert result["prospective_published"] is False
    assert result["submitted"] is False
    prospective_call.assert_not_called()
    handoff.assert_not_called()


def test_partial_prospective_plan_returns_without_platform_handoff(
    workflow_case, tmp_path, monkeypatch,
):
    repo, spec, plan = workflow_case["repo"], workflow_case["spec"], workflow_case["plan"]
    monkeypatch.setattr(workflow, "check_family_recipes", lambda *_args: {"status": "stub"})
    monkeypatch.setattr(workflow, "check_release_inputs", lambda *_args, **_kwargs: {
        "status": "LOCAL_RELEASE_INPUTS_VERIFIED", "errors": [],
        "inputs": {}, "checks": {}, "limitations": [],
    })
    partial = {"status": "PARTIAL_PROSPECTIVE_PLAN_REQUIRES_RECONCILIATION",
               "completed_records": [], "rml_rebuilt": False}
    monkeypatch.setattr(workflow, "plan_prospective", Mock(return_value=(partial, 1)))
    handoff = Mock()
    monkeypatch.setattr(workflow, "build_launch_receipt", handoff)

    result = workflow.prepare_workflow(spec, repo, plan, tmp_path / "partial")

    assert result["status"] == "RECONCILIATION_REQUIRED"
    assert result["stage"] == "prospective"
    assert result["prospective_published"] == "inspect_plan_result"
    assert result["submitted"] is False
    handoff.assert_not_called()


def test_accept_workflow_blocks_all_arm_closure_when_one_output_is_missing(
    tmp_path, monkeypatch,
):
    spec = _candidate_pair_spec()
    arm_ids = [arm["arm_id"] for arm in spec.to_dict()["arms"]]
    existing = tmp_path / "existing-output"
    existing.mkdir()
    missing = tmp_path / "missing-output"
    supplied_outputs = {
        arm_ids[0]: {"context": None, "output_dir": existing, "expected": {}},
        arm_ids[1]: {"context": None, "output_dir": missing, "expected": {}},
    }
    monkeypatch.setattr(workflow, "prepare_terminal_outputs",
        lambda *_args, **_kwargs: supplied_outputs)

    observed_bindings = {}

    def inspect(output_dir, **kwargs):
        observed_bindings[Path(output_dir)] = kwargs.get("target_identity")
        return {"status": "MECHANICALLY_VERIFIED" if Path(output_dir).is_dir() else "BLOCKED",
                "blockers": [] if Path(output_dir).is_dir() else ["output missing"]}

    from molgap import experiment_family_workflow as family_workflow
    monkeypatch.setattr(family_workflow, "inspect_output", inspect)
    descriptor, close = Mock(), Mock()
    monkeypatch.setattr(workflow, "build_verified_terminal_descriptor", descriptor)
    monkeypatch.setattr(workflow, "close_verified_outputs", close)

    target_identity = object()
    result = workflow.accept_workflow(
        spec, tmp_path, {}, receipt_path=tmp_path / "receipt.json",
        package_dir=tmp_path / "package", expected_package_identity=HEX_A,
        locations={arm_id: {} for arm_id in arm_ids}, execute=True,
        target_identities={arm_ids[0]: target_identity},
    )

    assert result["status"] == "BLOCKED"
    assert result["executed"] is False
    assert result["arms"][arm_ids[0]]["status"] == "MECHANICALLY_VERIFIED"
    assert result["arms"][arm_ids[1]]["status"] == "BLOCKED"
    assert observed_bindings[existing] is target_identity
    assert observed_bindings[missing] is None
    descriptor.assert_not_called()
    close.assert_not_called()


def _contexts(spec: ExperimentSpec) -> dict[str, RunContext]:
    return {
        arm["arm_id"]: RunContext(
            experiment_id=spec.to_dict()["experiment_id"],
            logical_run_id=spec.to_dict()["logical_run_id"],
            arm_id=arm["arm_id"],
            arm_identity=canonical_fingerprint(arm),
            spec_identity=spec.identity,
            family_name=arm["family"]["name"],
            family_version=arm["family"]["version"],
            source_commit="1" * 40,
            source_archive_sha256="2" * 64,
            package_identity="3" * 64,
            training_recipe_sha256=arm["training"]["recipe"]["sha256"],
            platform=spec.to_dict()["platform"]["name"],
            account="synthetic-account",
            run_reference="synthetic-account/synthetic-run",
            platform_version=None,
        )
        for arm in spec.to_dict()["arms"]
    }


def _incomplete_locations(spec: ExperimentSpec) -> dict:
    return {
        arm["arm_id"]: {
            "trajectory_id": "trajectory-" + arm["arm_id"],
            "run_id": "run-" + arm["arm_id"],
            "trajectory": f"experiments/incomplete/{arm['arm_id']}/trajectory.json",
            "terminal": f"experiments/incomplete/{arm['arm_id']}/terminal.json",
        }
        for arm in spec.to_dict()["arms"]
    }


def test_incomplete_terminal_descriptor_keeps_failed_and_cancelled_unknowns_typed():
    spec = _candidate_pair_spec()
    arm_ids = [arm["arm_id"] for arm in spec.to_dict()["arms"]]
    observations = {
        arm_ids[0]: {"status": "failed", "exit_reason": "preflight_worker_failed"},
        arm_ids[1]: {"status": "cancelled", "exit_reason": "peer_cancelled_after_failure"},
    }

    descriptor = build_incomplete_terminal_descriptor(
        spec, contexts=_contexts(spec), locations=_incomplete_locations(spec),
        observations=observations,
    )
    entries = {arm["arm_id"]: arm["observed"] for arm in descriptor.to_dict()["arms"]}

    for arm_id, status in zip(arm_ids, ("failed", "cancelled")):
        observed = entries[arm_id]
        assert observed["terminal"]["status"] == {"value": status, "missing_reason": None}
        assert all(observed["progress"][key]["value"] is None for key in ("epoch", "step", "samples"))
        assert all(observed["progress"][key]["missing_reason"] for key in ("epoch", "step", "samples"))
        assert observed["costs"][0]["status"] == "measurement_missing"
        assert observed["costs"][0]["value"] is None
        assert observed["costs"][0]["reason"]


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_incomplete_terminal_descriptor_preserves_explicit_zero_observations(status):
    spec = _candidate_pair_spec()
    observations = {
        arm["arm_id"]: {
            "status": status,
            "exit_reason": "no_work_completed",
            "progress": {"epoch": 0, "step": 0, "samples": 0},
            "costs": [{"metric": "wall_seconds", "unit": "seconds", "value": 0,
                       "status": "measured", "reason": None}],
        }
        for arm in spec.to_dict()["arms"]
    }

    descriptor = build_incomplete_terminal_descriptor(
        spec, contexts=_contexts(spec), locations=_incomplete_locations(spec),
        observations=observations,
    )

    for arm in descriptor.to_dict()["arms"]:
        observed = arm["observed"]
        assert {key: observed["progress"][key]["value"] for key in ("epoch", "step", "samples")} == {
            "epoch": 0, "step": 0, "samples": 0,
        }
        assert observed["costs"] == [{
            "metric": "wall_seconds", "unit": "seconds", "value": 0,
            "status": "measured", "reason": None,
        }]


def test_pair_runtime_preflight_failure_terminates_peer_and_never_starts_training(
    tmp_path, monkeypatch,
):
    from molgap import kaggle_pair_runtime as pair_runtime

    spec = _candidate_pair_spec()
    source_root = tmp_path / "source"
    package_dir = tmp_path / "package"
    input_root = tmp_path / "input"
    output = tmp_path / "output"
    for path in (source_root, package_dir, input_root):
        path.mkdir()
    (package_dir / "experiment_spec.json").write_bytes(spec.to_json().encode())
    jobs = _jobs(spec)
    launch_path = input_root / "experiment_launch.json"
    _write_json(launch_path, {
        "format": "molgap-execution-launch-v1",
        "spec_identity": spec.identity,
        "expected_package_identity": HEX_A,
        "expected_source_archive_sha256": HEX_B,
        "account": "synthetic-account",
        "run_reference": "synthetic-account/synthetic-pair-run",
        "jobs": jobs,
    })
    monkeypatch.setattr(pair_runtime.subprocess, "check_output",
        lambda *_args, **_kwargs: "NVIDIA Tesla T4\nNVIDIA Tesla T4\n")

    processes = {}
    commands = []

    class FakeProcess:
        def __init__(self, command):
            self.command = command
            self.arm_id = command[command.index("--arm") + 1]
            self.phase = command[command.index("--phase") + 1]
            self.returncode = None
            self.terminated = False

        def poll(self):
            if self.phase == "preflight" and self.arm_id == jobs[0]["arm_id"]:
                self.returncode = 17
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

        def kill(self):
            self.returncode = -9

    def fake_popen(command, **_kwargs):
        commands.append(command)
        process = FakeProcess(command)
        processes[process.arm_id, process.phase] = process
        return process

    monkeypatch.setattr(pair_runtime.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="Arm failed during preflight"):
        pair_runtime.run_two_phase_pair(
            source_root=source_root, package_dir=package_dir, input_root=input_root,
            launch_path=launch_path, output=output,
        )

    assert {command[command.index("--phase") + 1] for command in commands} == {"preflight"}
    failed = processes[jobs[0]["arm_id"], "preflight"]
    peer = processes[jobs[1]["arm_id"], "preflight"]
    assert failed.returncode == 17
    assert peer.terminated is True
    state = json.loads((output / "pair_state.json").read_bytes())
    assert state["status"] == "failed"
    phase = state["phases"][0]
    assert phase["phase"] == "preflight" and phase["status"] == "failed"
    assert {row["arm_id"] for row in phase["workers"]} == set(arm["arm_id"] for arm in spec.to_dict()["arms"])
    assert {row["returncode"] for row in phase["workers"]} == {17, -15}
    assert all((output / arm["arm_id"] / "preflight.log").is_file()
               for arm in spec.to_dict()["arms"])
    assert all(not arm["training_started"] for arm in state["arms"].values())
    from molgap.experiment_family_workflow import incomplete_observations_from_execution
    observations = incomplete_observations_from_execution(spec, state)
    assert all(observed["progress"] == {key: 0 for key in ("epoch", "step", "samples")}
               for observed in observations.values())

    wrong_hardware = copy.deepcopy(state)
    wrong_hardware["hardware"][0] = "NVIDIA A100"
    with pytest.raises(ValueError, match="matching observed T4 allocation"):
        incomplete_observations_from_execution(spec, wrong_hardware)
    duplicate_device = copy.deepcopy(state)
    arm_ids = [arm["arm_id"] for arm in spec.to_dict()["arms"]]
    duplicate_device["arms"][arm_ids[1]]["device"] = duplicate_device["arms"][arm_ids[0]]["device"]
    with pytest.raises(ValueError, match="distinct observed arm device assignment"):
        incomplete_observations_from_execution(spec, duplicate_device)
    out_of_range_device = copy.deepcopy(state)
    out_of_range_device["arms"][arm_ids[1]]["device"] = spec.to_dict()["platform"]["device_count"]
    with pytest.raises(ValueError, match="distinct observed arm device assignment"):
        incomplete_observations_from_execution(spec, out_of_range_device)

    # A persisted training-start marker means progress is unknown after a
    # crash; only a preflight failure proves that formal training stayed at 0.
    uncertain = copy.deepcopy(state)
    first_arm = spec.to_dict()["arms"][0]["arm_id"]
    uncertain["arms"][first_arm]["training_started"] = True
    unknown = incomplete_observations_from_execution(spec, uncertain)[first_arm]
    assert unknown["progress"] == {key: None for key in ("epoch", "step", "samples")}
    assert "training_progress_not_retained" in unknown["missing_evidence"]


def test_pair_runtime_train_failure_drains_peer_and_freezes_per_worker_wall(
    tmp_path, monkeypatch,
):
    from molgap import kaggle_pair_runtime as pair_runtime

    spec = _candidate_pair_spec()
    source_root = tmp_path / "source"
    package_dir = tmp_path / "package"
    input_root = tmp_path / "input"
    output = tmp_path / "output"
    for path in (source_root, package_dir, input_root):
        path.mkdir()
    (package_dir / "experiment_spec.json").write_bytes(spec.to_json().encode())
    jobs = _jobs(spec)
    launch_path = input_root / "experiment_launch.json"
    _write_json(launch_path, {
        "format": "molgap-execution-launch-v1",
        "spec_identity": spec.identity,
        "expected_package_identity": HEX_A,
        "expected_source_archive_sha256": HEX_B,
        "account": "synthetic-account",
        "run_reference": "synthetic-account/synthetic-pair-run",
        "jobs": jobs,
    })
    monkeypatch.setattr(pair_runtime.subprocess, "check_output",
        lambda *_args, **_kwargs: "NVIDIA Tesla T4\nNVIDIA Tesla T4\n")

    clock = [0.0]
    monkeypatch.setattr(pair_runtime.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(pair_runtime.time, "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    processes = {}
    commands = []

    class FakeProcess:
        def __init__(self, command):
            self.command = command
            self.arm_id = command[command.index("--arm") + 1]
            self.phase = command[command.index("--phase") + 1]
            self.returncode = None
            self.terminated = False
            self.poll_count = 0

        def poll(self):
            if self.returncode is not None:
                return self.returncode
            if self.phase == "preflight":
                self.returncode = 0
            else:
                self.poll_count += 1
                if self.arm_id == jobs[0]["arm_id"]:
                    self.returncode = 23
                elif self.poll_count >= 4:
                    self.returncode = 0
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

        def kill(self):
            self.returncode = -9

    def fake_popen(command, **_kwargs):
        commands.append(command)
        process = FakeProcess(command)
        processes[process.arm_id, process.phase] = process
        return process

    monkeypatch.setattr(pair_runtime.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="Arm failed during train"):
        pair_runtime.run_two_phase_pair(
            source_root=source_root, package_dir=package_dir, input_root=input_root,
            launch_path=launch_path, output=output,
        )

    assert {command[command.index("--phase") + 1] for command in commands} == {
        "preflight", "train",
    }
    failed = processes[jobs[0]["arm_id"], "train"]
    peer = processes[jobs[1]["arm_id"], "train"]
    assert failed.returncode == 23
    assert peer.returncode == 0
    assert peer.terminated is False

    state = json.loads((output / "pair_state.json").read_bytes())
    assert state["status"] == "failed"
    assert all(arm["started"] and arm["training_started"] for arm in state["arms"].values())
    assert state["arms"][jobs[0]["arm_id"]]["terminal_status"] == "failed"
    assert state["arms"][jobs[1]["arm_id"]]["terminal_status"] == "complete"
    assert state["arms"][jobs[0]["arm_id"]]["exit_reason"] == "train_worker_exit_23"
    assert state["arms"][jobs[1]["arm_id"]]["exit_reason"] == "train_worker_exit_0"
    assert state["arms"][jobs[0]["arm_id"]]["worker_wall_seconds"] == 0.0
    assert state["arms"][jobs[1]["arm_id"]]["worker_wall_seconds"] == 1.5
    train_phase = next(phase for phase in state["phases"] if phase["phase"] == "train")
    assert train_phase["status"] == "failed"
    assert {row["terminal_status"] for row in train_phase["workers"]} == {"failed", "complete"}
    assert {row["started"] for row in train_phase["workers"]} == {True}


def test_pair_runtime_budget_terminates_workers_without_claiming_endpoint(tmp_path, monkeypatch):
    from molgap import kaggle_pair_runtime as pair_runtime

    spec = _candidate_pair_spec()
    package = tmp_path / "package"
    package.mkdir()
    (package / "experiment_spec.json").write_bytes(spec.to_json().encode())
    launch = tmp_path / "launch.json"
    _write_json(launch, {"jobs": _jobs(spec)})
    monkeypatch.setattr(pair_runtime.subprocess, "check_output",
                        lambda *args, **kwargs: "Tesla T4\nTesla T4\n")
    clock = [0.0]
    monkeypatch.setattr(pair_runtime.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(pair_runtime.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    processes = []

    class Process:
        def __init__(self, command):
            self.phase = command[command.index("--phase") + 1]
            self.returncode = None
            self.terminated = False

        def poll(self):
            return 0 if self.phase == "preflight" else self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    def spawn(command, **kwargs):
        process = Process(command)
        processes.append(process)
        return process

    monkeypatch.setattr(pair_runtime.subprocess, "Popen", spawn)
    with pytest.raises(TimeoutError, match="allocation ceiling"):
        pair_runtime.run_two_phase_pair(source_root=tmp_path, package_dir=package,
            input_root=tmp_path, launch_path=launch, output=tmp_path / "output", maximum_wall_seconds=1)
    state = json.loads((tmp_path / "output/pair_state.json").read_text())
    assert state["status"] == "failed"
    assert state["stop_reason"] == "STOP_FOR_COST"
    assert all(process.terminated for process in processes if process.phase == "train")
    assert all(arm["terminal_status"] == "cancelled" for arm in state["arms"].values())


def test_check_workflow_binding_rejects_tampered_final_prospective_identity(tmp_path):
    spec = _candidate_pair_spec()
    staged = tmp_path / "source-dataset"
    staged.mkdir()
    metadata_path = tmp_path / "kernel-metadata.json"
    entry_path = tmp_path / "run.py"
    arm_ids = [arm["arm_id"] for arm in spec.to_dict()["arms"]]
    prospective_hashes = {}
    for arm_id in arm_ids:
        trajectory = staged / "prospective" / arm_id / "trajectory.json"
        trajectory.parent.mkdir(parents=True)
        _write_json(trajectory, {"synthetic": True, "arm_id": arm_id})
        prospective_hashes[arm_id] = _sha(trajectory.read_bytes())
    metadata = {
        "id": "synthetic-account/synthetic-pair-run",
        "dataset_sources": ["synthetic-account/source-data"],
        "enable_gpu": True,
        "enable_tpu": False,
    }
    _write_json(metadata_path, metadata)
    launch = {
        "spec_identity": spec.identity,
        "run_reference": metadata["id"],
        "account": "synthetic-account",
        "dataset_sources": metadata["dataset_sources"],
        "accelerator": "NvidiaTeslaT4",
        "device_count": 2,
        "prospective_sha256": prospective_hashes,
    }
    launch_path = staged / "experiment_launch.json"
    _write_json(launch_path, launch)
    entry_path.write_text(
        "EXPECTED_LAUNCH_SHA256 = " + repr(_sha(launch_path.read_bytes())) + "\n",
        encoding="utf-8",
    )

    checked = check_workflow_binding(
        spec, launch_config=launch_path, kernel_metadata=metadata_path,
        entry_script=entry_path,
    )
    assert checked["prospective_sha256"] == prospective_hashes

    changed = staged / "prospective" / arm_ids[1] / "trajectory.json"
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Published prospective record changed"):
        check_workflow_binding(
            spec, launch_config=launch_path, kernel_metadata=metadata_path,
            entry_script=entry_path,
        )


def test_real_shared_source_inventory_clean_imports_training_owners(tmp_path):
    source_root = tmp_path / "source"
    for relative in sorted(source_inventory.SHARED_SOURCE_FILES):
        source, destination = ROOT / relative, source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    modules = [
        "molgap.k1_screen_training",
        "molgap.gptrans_screen_workflow",
        "molgap.experiment_training_worker",
        "molgap.pcqm_wedge",
    ]
    bootstrap = tmp_path / "bootstrap.py"
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    shutil.copyfile(preflight.__file__, bootstrap)
    preflight._atomic(request, {
        "mode": "release-imports",
        "source_root": str(source_root),
        "modules": modules,
        "pickle_globals": [["molgap.pcqm_wedge", "WedgeData"]],
        "dependency_paths": sorted({
            sysconfig.get_path("purelib"), sysconfig.get_path("platlib"),
        }),
    })
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith("PYTHON")}
    env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    subprocess.run(
        [sys.executable, "-I", "-S", str(bootstrap), str(request), str(response)],
        cwd=tmp_path, env=env, check=True, capture_output=True, timeout=60,
    )

    result = preflight._load(response.read_bytes())
    assert result["errors"] == []
    origins = result["import_origins"]
    assert set(modules) <= set(origins)
    assert all((source_root / relative).is_file() for relative in origins.values())
    assert all(relative.startswith("src/molgap/") for relative in origins.values())


def test_workflow_navigation_docs_point_to_lifecycle_and_cli():
    guide = (ROOT / "docs/operations/EXPERIMENT_ADDON_GUIDE.md").read_text(encoding="utf-8")
    cli_doc = (ROOT / "docs/operations/EXPERIMENT_CLI.md").read_text(encoding="utf-8")

    assert "EXPERIMENT_WORKFLOW.md" in guide
    assert "EXPERIMENT_CLI.md" in guide
    for command in ("prepare-workflow", "accept-workflow", "build-terminal"):
        assert command in cli_doc


@pytest.mark.parametrize("command", ["prepare-workflow", "accept-workflow", "build-terminal"])
def test_new_workflow_commands_emit_json_help(capsys, command):
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    value = json.loads(output)
    assert output == canonical_json(value) + "\n"
    assert "usage:" in value["help"]


def _cli_json(capsys, args, expected_code):
    code = cli.main([str(arg) for arg in args])
    output = capsys.readouterr().out
    result = json.loads(output)
    assert code == expected_code
    assert output == canonical_json(result) + "\n"
    return result


def test_prepare_and_accept_workflow_cli_json_dispatch(tmp_path, capsys, monkeypatch):
    spec = _candidate_pair_spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(spec.to_json().encode())
    repo = tmp_path / "repo"
    repo.mkdir()
    plan_path = tmp_path / "plan.json"
    outputs_path = tmp_path / "outputs.json"
    locations_path = tmp_path / "locations.json"
    _write_json(plan_path, {"synthetic": True})
    _write_json(outputs_path, {"synthetic": True})
    _write_json(locations_path, {"synthetic": True})

    prepare = Mock(return_value={"status": "PREPARED_FOR_PLATFORM", "submitted": False})
    accept = Mock(return_value={"status": "BLOCKED", "arms": {}, "executed": False})
    monkeypatch.setattr(workflow, "prepare_workflow", prepare)
    monkeypatch.setattr(workflow, "accept_workflow", accept)

    prepared = _cli_json(capsys, [
        "prepare-workflow", "--spec", spec_path, "--repo-root", repo,
        "--plan", plan_path, "--output", tmp_path / "prepared",
    ], 0)
    assert prepared == {"status": "PREPARED_FOR_PLATFORM", "submitted": False}
    prepare.assert_called_once_with(spec, repo, {"synthetic": True}, tmp_path / "prepared")

    blocked = _cli_json(capsys, [
        "accept-workflow", "--spec", spec_path, "--repo-root", repo,
        "--package", tmp_path / "package", "--expected-package-identity", HEX_A,
        "--receipt", tmp_path / "receipt.json", "--outputs", outputs_path,
        "--locations", locations_path,
    ], 1)
    assert blocked == {"status": "BLOCKED", "arms": {}, "executed": False}
    accept.assert_called_once_with(
        spec, repo, {"synthetic": True}, receipt_path=tmp_path / "receipt.json",
        package_dir=tmp_path / "package", expected_package_identity=HEX_A,
        locations={"synthetic": True}, execute=False,
    )
