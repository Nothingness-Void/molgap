"""Actual-owner integration for the registered local workflow.

The fixture is intentionally synthetic.  It exercises source packaging,
prospective planning, Kaggle staging, release binding, launch reconciliation,
and retained family-output acceptance without constructing a model or calling a
platform API.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

pytest_plugins = ["test_experiment_workflow"]

from molgap import experiment_execution as execution
from molgap import experiment_launch as launch
from molgap.experiment_family_workflow import (
    FamilyOutputSession,
    RunContext,
    _metric_semantics,
    tensor_digest,
)
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_workflow import accept_workflow, prepare_workflow
from molgap.research_memory.trace import file_digest, json_bytes
from molgap.screen_policy import canonical_fingerprint
from test_experiment_workflow import (
    _write_json,
    _write_strict_retained_reference_plan,
)
from test_comparison_readiness import _target_transform_asset
from test_research_memory_plan_batch import _item
from test_terminal_trace_closure import create_candidate_arm, setup_mock_repo


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _synthetic_policy(repo: Path) -> dict:
    source = Path(__file__).resolve().parents[1] / (
        "research_memory/policies/k1-relation-resolution-bounded-research.v1.json"
    )
    policy = json.loads(source.read_bytes())
    target = repo / "research_memory/policies/synthetic.v1.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(json_bytes(policy))
    return policy


def _prepare_reference_inputs(repo: Path) -> None:
    """Add the local pointers used by prospective planning."""
    setup_mock_repo(repo)
    ref = repo / "experiments/ref_exp"
    (ref / "role_plan.json").write_bytes(json_bytes({"schema": "role-plan-v1", "synthetic": True}))
    (ref / "budget.json").write_bytes(json_bytes({"schema": "budget-v1", "synthetic": True}))
    _synthetic_policy(repo)


def _plan_input(repo: Path, arm: dict, binding: dict, index: int, policy: dict) -> dict:
    """Build a complete synthetic RML plan input from the real plan schema."""
    label = f"registered-{index}"
    source = _item(label)
    trajectory = source["spec"]["trajectory"]
    action_id = "act-1"
    run_id = f"synthetic-workflow-run:{binding['arm_id']}:downstream"
    cost_id = f"cost-{binding['arm_id']}"
    trajectory.update(
        trajectory_id=binding["trajectory_id"],
        question=f"Synthetic registered workflow question {binding['arm_id']}",
    )
    trajectory["hypothesis"].update(
        hypothesis_id=f"H-{binding['arm_id']}",
        supporting_evidence_ids=["ev-ref-1"],
        expected_native_cost_ref=cost_id,
    )
    trajectory["state_at_start"].update(
        source_config_identity=canonical_fingerprint(arm),
        contract_refs=["experiments/ref_exp/contract.json"],
        reference_ids=["ev-ref-1"],
        prior_evidence_ids=["ev-ref-1"],
        role_snapshot_refs=["experiments/ref_exp/role_plan.json"],
        budget_snapshot_ref="experiments/ref_exp/budget.json",
    )
    trajectory["actions"][0].update(
        action_id=action_id,
        source_commit=_git(repo, "rev-parse", "HEAD"),
        run_ids=[run_id],
        evidence_refs=["experiments/ref_exp/v5_evidence.json"],
        cost_event_ids=[cost_id],
    )
    trajectory["decision"].update(
        decision_ref="experiments/ref_exp/decision.md",
        next_allowed_actions=[action_id],
    )
    trajectory["result"] = {"evidence_ids": [], "evidence_refs": []}

    cost = copy.deepcopy(source["spec"]["costs"][0])
    cost.update(
        cost_event_id=cost_id,
        trajectory_id=binding["trajectory_id"],
        action_id=action_id,
        run_id=run_id,
        attempt_id=f"attempt-{binding['arm_id']}",
        platform="kaggle",
        hardware="synthetic-cpu",
        evidence_ref="experiments/ref_exp/v5_evidence.json",
    )
    decision_state = {
        "available_actions": [action_id],
        "chosen_action": action_id,
        "policy_id": policy["policy_id"],
        "policy_version": policy["version"],
        "state_timestamp": "2026-10-02T00:00:00Z",
    }
    return {"trajectory": trajectory, "decision_state": decision_state, "costs": [cost]}


def _install_synthetic_owner(repo: Path, monkeypatch):
    """Install a temp-repo module and a registry entry for schema-only checks."""
    module_path = repo / "src/molgap/synthetic_trainer.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        "def validate_screen_recipe(spec, arm_id, recipe):\n"
        "    if recipe.get('format') != 'synthetic-screen-recipe-v1':\n"
        "        raise ValueError('synthetic recipe format mismatch')\n"
        "    return {'recipe': 'synthetic_recipe_verified', 'arm_id': arm_id}\n",
        encoding="utf-8",
    )
    _git(repo, "add", "src/molgap/synthetic_trainer.py")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic registered owner")

    module_spec = importlib.util.spec_from_file_location(
        "molgap.synthetic_trainer", module_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, "molgap.synthetic_trainer", module)
    module_spec.loader.exec_module(module)

    from molgap.experiment_execution import TrainingAdapter, TrainingAddon

    registry = {
        ("neural_atom_k1", "2"): TrainingAdapter(
            ("neural_atom_k1", "2"),
            "molgap.synthetic_trainer",
            "k1-screen-v1",
            (TrainingAddon("k1_joint_aggregation", "1", "ssma"),),
            source_files=("src/molgap/synthetic_trainer.py",),
        ),
        ("gptrans_t", "1"): TrainingAdapter(
            ("gptrans_t", "1"),
            "molgap.synthetic_trainer",
            "gptrans-v1",
            (TrainingAddon("pair_prenorm", "1", "pair_prenorm"),),
            source_files=("src/molgap/synthetic_trainer.py",),
        ),
    }
    monkeypatch.setattr(execution, "TRAINING_ADAPTERS", MappingProxyType(registry))
    return module_path


def _rewrite_case(case, repo: Path, module_path: Path) -> tuple[ExperimentSpec, dict, dict, dict]:
    """Bind recipe/plan identities after adding the synthetic owner module."""
    declaration = case["spec"].to_dict()
    source_idx = torch.arange(100, 110, dtype=torch.int64)
    target = torch.zeros(10, dtype=torch.float64)
    expected = copy.deepcopy(case["expected"])
    expected.update(
        source_idx_sha256=tensor_digest(source_idx, role="source_idx"),
        target_sha256=tensor_digest(target, role="target"),
        development_rows=10,
    )
    from molgap.v4_runtime import state_dict_sha256

    recipes = {}
    for arm in declaration["arms"]:
        arm_id = arm["arm_id"]
        initial = torch.load(case["initial_states"][arm_id], map_location="cpu", weights_only=True)
        arm["initialization"]["state_sha256"] = state_dict_sha256(initial)
        recipe_path = repo / f"recipes/{arm_id}.json"
        recipe = json.loads(recipe_path.read_bytes())
        recipe["acceptance_requirements"] = expected
        recipe["development_role_identity"] = "synthetic-development"
        recipe["metric_semantics"] = {
            "live_train_metric": {
                "metric": "MAE", "unit": "eV", "target": "Gap", "weights": "live",
                "direction": "minimize", "role_identity": "synthetic-train",
            },
            "live_dev_metric": {
                "metric": "MAE", "unit": "eV", "target": "Gap", "weights": "live",
                "direction": "minimize", "role_identity": "synthetic-development",
            },
            "ema_dev_metric": {
                "metric": "MAE", "unit": "eV", "target": "Gap", "weights": "ema",
                "direction": "minimize", "role_identity": "synthetic-development",
            },
        }
        recipe_bytes = json_bytes(recipe)
        recipe_path.write_bytes(recipe_bytes)
        recipes[arm_id] = recipe_path.relative_to(repo).as_posix()
        arm["training"]["recipe"]["sha256"] = _sha(recipe_bytes)

    # Recipes are packaged source, so the preparation owner must see their
    # exact bytes in the committed source snapshot used by the package.
    _git(repo, "add", "recipes")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic registered recipes")

    for arm, binding in zip(declaration["arms"], declaration["prospective"]["arms"]):
        plan_value = _plan_input(repo, arm, binding, len(recipes), _synthetic_policy(repo))
        plan_path = repo / binding["plan_spec_ref"]
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_bytes = json_bytes(plan_value)
        plan_path.write_bytes(plan_bytes)
        binding["plan_spec_sha256"] = _sha(plan_bytes)

    spec = ExperimentSpec(declaration)
    acceptance_plan = _write_strict_retained_reference_plan(repo, spec, expected, recipes)
    # The strict reference helper intentionally uses compact placeholder
    # artifacts.  The actual compiler validates this one asset's schema, so
    # replace only that placeholder and rebind the acceptance pointer.
    transform_path = repo / "experiments/reference/target_transform.json"
    (repo / "experiments/reference/contract.json").write_bytes(
        json_bytes({"schema": "contract-v1", "synthetic": True})
    )
    transform_path.write_bytes(json_bytes(_target_transform_asset()))
    acceptance_path = repo / acceptance_plan
    acceptance = json.loads(acceptance_path.read_bytes())
    for entry in acceptance["arms"]:
        entry["reference_artifacts"]["target_transform_asset"]["sha256"] = file_digest(transform_path)
    acceptance_path.write_bytes(json_bytes(acceptance))
    plan = copy.deepcopy(case["plan"])
    plan["spec_identity"] = spec.identity
    plan["acceptance_plan"] = acceptance_plan
    for arm in plan["arms"]:
        arm["recipe"] = recipes[arm["arm_id"]]
    return spec, plan, expected, recipes


def _launch_receipt(spec, package_dir: Path, package_identity: str, repo: Path) -> Path:
    reference = "synthetic-account/synthetic-pair-run"
    binding = launch.build_launch_receipt(
        spec, package_dir, expected_package_identity=package_identity
    )["binding"]
    response = {
        "format": launch.RESPONSE_FORMAT,
        "version": launch.VERSION,
        "mode": "observed",
        "outcome": "accepted",
        "conflict_kind": None,
        "binding": binding,
        "canonical_platform_reference": {"value": reference, "missing_reason": None},
        "platform_version": {"value": "synthetic-platform-v1", "missing_reason": None},
        "physical_runs": {"value": [{
            "run_identity": "synthetic-physical-run",
            "canonical_reference": reference,
            "platform_version": {"value": "synthetic-platform-v1", "missing_reason": None},
            "arm_ids": [arm["arm_id"] for arm in spec.to_dict()["arms"]],
        }], "missing_reason": None},
        "timestamp": {"value": None, "missing_reason": "not_observed"},
        "monitor_paths": {"value": None, "missing_reason": "not_observed"},
    }
    raw = launch.canonical_json(launch.reconcile_platform_response(
        spec, package_dir, launch.canonical_json(response),
        expected_package_identity=package_identity,
    ))
    receipt_dir = repo / "receipts"
    receipt_dir.mkdir()
    return launch.write_launch_receipt(
        raw, receipt_dir, spec, package_dir,
        expected_package_identity=package_identity,
    )


def _write_family_output(repo: Path, spec: ExperimentSpec, package_dir: Path,
                         package_identity: str, receipt_path: Path, expected: dict,
                         arm_id: str, recipe_path: str) -> Path:
    context = RunContext.from_launch(
        spec, receipt_path, package_dir,
        expected_package_identity=package_identity, arm_id=arm_id,
    )
    output = repo / f"retained/family_outputs/{arm_id}"
    recipe = repo / recipe_path
    adapter = "gptrans-v1" if context.family_name == "gptrans_t" else "k1-screen-v1"
    session = FamilyOutputSession(
        output, context, adapter=adapter, contract=recipe,
        trajectory_id=next(row["trajectory_id"] for row in spec.to_dict()["prospective"]["arms"]
                            if row["arm_id"] == arm_id),
        metric_semantics=_metric_semantics(json.loads(recipe.read_bytes())),
    )
    source_idx = torch.arange(100, 110, dtype=torch.int64)
    target = torch.zeros(10, dtype=torch.float64)
    prediction = torch.ones(10, dtype=torch.float64)
    for epoch in range(1, 3):
        model = {"weight": torch.tensor([0.5])}
        optimizer = {"state": {}, "param_groups": [{"params": [0], "lr": 0.001}]}
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": [],
        }
        session.epoch_finished(
            epoch=epoch, optimizer_step=epoch * 2,
            sample_presentations=epoch * 4,
            live_train_metric=1.0, live_dev_metric=1.0, ema_dev_metric=1.0,
        )
        if epoch == 2:
            session.selected(
                model_state=model, epoch=epoch, optimizer_step=epoch * 2,
                weights="ema" if adapter == "gptrans-v1" else "live",
                prediction_eV=prediction, target_eV=target, source_idx=source_idx,
            )
        resume = {"model": model, "optimizer": optimizer, "rng_state": rng,
                  "cursor": {"epoch": epoch, "next_batch": 0,
                             "sampler_order_sha256": "a" * 64}}
        if adapter == "gptrans-v1":
            resume["ema"] = model
        else:
            resume["scheduler"] = {"last_epoch": epoch}
        session.checkpoint(
            model_state=resume["model"], optimizer_state=resume["optimizer"],
            rng_state=resume["rng_state"], cursor=resume["cursor"],
            optimizer_step=epoch * 2, sample_presentations=epoch * 4,
            scheduler_state=resume.get("scheduler"), ema_state=resume.get("ema"),
        )
    report = session.complete(runtime={
        "platform": context.platform, "account": context.account,
        "precision": "fp32", "source_commit": context.source_commit,
        "source_archive_sha256": context.source_archive_sha256,
    }, hardware="synthetic-cpu")
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    assert report["observed"]["progress"]["sample_presentations"] == expected["sample_presentations"]
    return output


def _terminal_locations(repo: Path, spec: ExperimentSpec, output_roots: dict[str, Path],
                        source_commit: str) -> dict:
    locations = {}
    for arm in spec.to_dict()["arms"]:
        arm_id = arm["arm_id"]
        binding = next(row for row in spec.to_dict()["prospective"]["arms"]
                       if row["arm_id"] == arm_id)
        run_id = f"synthetic-workflow-run:{arm_id}:downstream"
        candidate = create_candidate_arm(
            repo, f"authority-{arm_id}", binding["trajectory_id"], run_id,
            f"ev-authority-{arm_id}",
        )
        # The planned trajectory is the one authoritative prospective record.
        candidate["traj_path"].unlink()
        terminal = json.loads(candidate["terminal_path"].read_bytes())
        trace = output_roots[arm_id] / "canonical_trace.json"
        relative_trace = trace.resolve().relative_to(repo.resolve()).as_posix()
        artifact = next(row for row in terminal["evidence"]["artifacts"]
                        if row["name"] == "training_trace")
        terminal["artifact_hashes"].pop(artifact["locator"])
        artifact.update(locator=relative_trace, sha256=file_digest(trace))
        terminal["artifact_hashes"][relative_trace] = artifact["sha256"]
        candidate["terminal_path"].write_bytes(json_bytes(terminal))
        locations[arm_id] = {
            "trajectory_id": binding["trajectory_id"],
            "run_id": run_id,
            "trajectory": binding["output"] + "/trajectory.json",
            "terminal": candidate["terminal_path"].resolve().relative_to(repo.resolve()).as_posix(),
            "trace": relative_trace,
        }
    return locations


def test_registered_workflow_prepares_stages_rechecks_and_accepts_synthetic_outputs(
    tmp_path, workflow_case, monkeypatch,
):
    repo = workflow_case["repo"]
    _prepare_reference_inputs(repo)
    module_path = _install_synthetic_owner(repo, monkeypatch)
    spec, plan, expected, recipes = _rewrite_case(workflow_case, repo, module_path)

    result = prepare_workflow(spec, repo, plan, tmp_path / "prepared")

    assert result["status"] == "PREPARED_FOR_PLATFORM", json.dumps(result, indent=2)
    assert result["submitted"] is False
    assert result["prospective_published"] is True
    prepared = tmp_path / "prepared"
    package_dir = prepared / "package"
    source_dataset = prepared / "source_dataset"
    kernel_dir = prepared / "kernel"
    assert (source_dataset / "experiment_launch.json").is_file()
    assert (source_dataset / "prospective" / "k1_candidate" / "trajectory.json").is_file()
    assert (kernel_dir / "run.py").is_file()
    assert result["release_report"] == str(prepared / "release_report.json")
    assert result["timings"]["total_local_preparation_seconds"] >= 0
    assert "src/molgap/synthetic_trainer.py" in [
        entry["path"] for entry in json.loads((package_dir / "SOURCE_FILES.json").read_bytes())["files"]
    ]

    # Repeat the same release check a platform adapter uses immediately before
    # POST, proving the frozen staged launch remains independently verifiable.
    from molgap.experiment_preflight import check_release_inputs
    recheck = check_release_inputs(
        spec, package_dir, expected_package_identity=result["package_identity"],
        recipe_files=recipes,
        initial_states={arm_id: source_dataset / "initial_states" / f"{arm_id}.pt"
                        for arm_id in recipes},
        required_modules=["molgap.synthetic_trainer", "molgap.experiment_training_worker"],
        input_root=source_dataset,
        entry_script=kernel_dir / "run.py",
        launch_config=source_dataset / "experiment_launch.json",
        kernel_metadata=kernel_dir / "kernel-metadata.json",
    )
    assert recheck["status"] == "LOCAL_RELEASE_INPUTS_VERIFIED", recheck
    assert recheck["errors"] == []

    receipt = _launch_receipt(spec, package_dir, result["package_identity"], repo)
    output_roots = {
        arm_id: _write_family_output(
            repo, spec, package_dir, result["package_identity"], receipt,
            expected, arm_id, recipes[arm_id],
        )
        for arm_id in recipes
    }
    locations = _terminal_locations(repo, spec, output_roots, result["source_commit"])
    supplied = {
        arm_id: {
            "output_dir": output_roots[arm_id].resolve().relative_to(repo.resolve()).as_posix(),
            "expected": expected,
        }
        for arm_id in recipes
    }

    accepted = accept_workflow(
        spec, repo, supplied, receipt_path=receipt, package_dir=package_dir,
        expected_package_identity=result["package_identity"], locations=locations,
        execute=False,
    )
    assert accepted["status"] == "MECHANICALLY_VERIFIED", accepted
    assert accepted["executed"] is False
    assert set(accepted["arms"]) == set(recipes)
    assert all(report["status"] == "MECHANICALLY_VERIFIED"
               for report in accepted["arms"].values())
