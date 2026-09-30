"""Synthetic family output validation; no model, protected role, or remote run."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import molgap.experiment_launch as launch
from molgap.experiment_family_workflow import (
    FamilyOutputSession,
    RunContext,
    StageRecorder,
    build_verified_terminal_descriptor,
    check_acceptance_plan,
    close_verified_outputs,
    inspect_output,
    tensor_digest,
    write_resume_state,
    write_output_manifest,
    write_selected_state,
)
from molgap.experiment_package import (
    build_experiment_source_package,
    verify_experiment_source_package,
)
from molgap.experiment_spec import ExperimentSpec
from molgap.comparison_readiness import REQUIRED_REFERENCE_ARTIFACTS
from molgap.research_memory.terminal_wiring import close_terminal_arm
from molgap.research_memory.compiler import frozen_differences
from molgap.research_memory.finalize import verified_receipt
from molgap.research_memory.trace import (
    canonicalize_trace, file_digest, json_bytes, load_canonical_trace,
)
from test_comparison_readiness import _prelaunch, _reference_bundle
from test_terminal_trace_closure import create_candidate_arm, setup_mock_repo
from test_experiment_package import repo  # Synthetic local Git/package fixture.
from test_experiment_spec import payload


FAMILY_CASES = [
    ("gptrans_t", "gptrans-v1", "ema"),
    ("neural_atom_k1", "k1-v1", "live"),
]


def known(value):
    return {"value": value, "missing_reason": None}


def unknown(reason="not_observed"):
    return {"value": None, "missing_reason": reason}


def metric(weight, role):
    return {
        "metric": "MAE",
        "unit": "eV",
        "target": "Gap",
        "role_identity": role,
        "weights": weight,
        "direction": "minimize",
    }


METRIC_SEMANTICS = {
    "live_train_metric": metric("live", "train"),
    "live_dev_metric": metric("live", "dev"),
    "ema_dev_metric": metric("ema", "dev"),
}

SOURCE_IDX = torch.tensor([101, 103, 108], dtype=torch.int64)
TARGET = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)


def _expected_requirements():
    return {
        "epochs": 2,
        "optimizer_steps": 4,
        "sample_presentations": 8,
        "development_rows": 3,
        "source_idx_sha256": tensor_digest(SOURCE_IDX, role="source_idx"),
        "target_sha256": tensor_digest(TARGET, role="target"),
        "precision": "fp32",
    }


def _recipe(family_name):
    return {
        "format": "synthetic-family-recipe-v1",
        "family": {"name": family_name, "version": "1"},
        "acceptance_requirements": _expected_requirements(),
    }


@pytest.fixture
def launch_contexts(tmp_path, repo, payload):
    recipes = []
    for arm in payload["arms"]:
        family_name = arm["family"]["name"]
        recipe = _recipe(family_name)
        recipe_path = repo / "recipes" / f"{family_name}.json"
        recipe_path.parent.mkdir(exist_ok=True)
        recipe_bytes = json_bytes(recipe)
        recipe_path.write_bytes(recipe_bytes)
        arm["training"]["recipe"]["sha256"] = hashlib.sha256(recipe_bytes).hexdigest()
        recipes.append(recipe_path.relative_to(repo).as_posix())
    subprocess.run(["git", "add", "recipes"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic family recipes"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    spec = ExperimentSpec(payload)
    package_dir = tmp_path / "family-package"
    build_experiment_source_package(
        spec,
        repo,
        ["src/example.py", "README.md", *recipes],
        package_dir,
    )
    package_identity = verify_experiment_source_package(package_dir)["package_identity"]
    binding = launch.build_launch_receipt(
        spec, package_dir, expected_package_identity=package_identity
    )["binding"]
    arm_ids = [arm["arm_id"] for arm in spec.to_dict()["arms"]]
    canonical_reference = "synthetic-account/workload-1"
    physical_runs = [{
        "run_identity": "synthetic-physical-run",
        "canonical_reference": canonical_reference,
        "platform_version": known("synthetic-platform-v1"),
        "arm_ids": arm_ids,
    }]
    response = {
        "format": launch.RESPONSE_FORMAT,
        "version": launch.VERSION,
        "mode": "observed",
        "outcome": "accepted",
        "conflict_kind": None,
        "binding": binding,
        "canonical_platform_reference": known(canonical_reference),
        "platform_version": known("synthetic-platform-v1"),
        "physical_runs": known(physical_runs),
        "timestamp": unknown(),
        "monitor_paths": unknown(),
    }
    receipt_json = launch.canonical_json(launch.reconcile_platform_response(
        spec,
        package_dir,
        launch.canonical_json(response),
        expected_package_identity=package_identity,
    ))
    receipt_dir = tmp_path / "launch-receipts"
    receipt_dir.mkdir()
    receipt_path = launch.write_launch_receipt(
        receipt_json,
        receipt_dir,
        spec,
        package_dir,
        expected_package_identity=package_identity,
    )
    contexts = {
        arm_id: RunContext.from_launch(
            spec,
            receipt_path,
            package_dir,
            expected_package_identity=package_identity,
            arm_id=arm_id,
        )
        for arm_id in arm_ids
    }
    return spec, package_dir, package_identity, receipt_path, contexts


def _runtime(context, *, precision="fp32", **overrides):
    value = {
        "platform": context.platform,
        "account": context.account,
        "precision": precision,
        "source_commit": context.source_commit,
        "source_archive_sha256": context.source_archive_sha256,
    }
    value.update(overrides)
    return value


def _costs(*, status="measured"):
    return [{
        "metric": "wall_seconds",
        "unit": "seconds",
        "value": 12.0,
        "status": status,
        "semantics": "process_wall",
        "hardware": "synthetic-cpu",
    }]


def _write_trace(path, context, *, partial=False):
    trajectory_id = f"{context.experiment_id}-{context.arm_id}"
    run_id = f"{context.logical_run_id}:{context.arm_id}:downstream"
    if partial:
        trace = canonicalize_trace({
            "trajectory_id": trajectory_id,
            "run_id": run_id,
            "metric_semantics": copy.deepcopy(METRIC_SEMANTICS),
            "observations": [{
                "epoch_or_pass": 1,
                "optimizer_step": None,
                "sample_presentations": 4,
                "live_train_metric": 1.1,
                "live_dev_metric": 1.0,
                "ema_dev_metric": 1.0,
            }],
        })
        path.write_bytes(json_bytes(trace))
        return

    recorder = StageRecorder(path, context, "downstream", METRIC_SEMANTICS)
    recorder.observe(
        optimizer_step=2,
        sample_presentations=4,
        epoch_or_pass=1,
        learning_rate=0.001,
        live_train_metric=1.2,
        live_dev_metric=2.0,
        ema_dev_metric=2.0,
    )
    recorder.observe(
        optimizer_step=4,
        sample_presentations=8,
        epoch_or_pass=2,
        learning_rate=0.0005,
        live_train_metric=1.1,
        live_dev_metric=1.0,
        ema_dev_metric=1.0,
    )


def _write_output(
    tmp_path,
    context,
    adapter,
    *,
    variant=None,
    runtime_overrides=None,
    progress_overrides=None,
    resume_missing=None,
    selected_overrides=None,
    costs=None,
    publish=True,
):
    root = tmp_path / f"output-{context.arm_id}"
    root.mkdir()

    source_idx = torch.tensor([101, 103, 108], dtype=torch.int64)
    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    prediction = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    protected = {}
    if variant == "wrong_rows":
        source_idx = source_idx + 1
    elif variant == "wrong_targets":
        target = target + 1.0
    elif variant == "nan_target":
        target[0] = float("nan")
    elif variant == "nan_prediction":
        prediction[0] = float("nan")
    elif variant == "wrong_shape":
        prediction = prediction[:2]
    elif variant == "duplicate_rows":
        source_idx[1] = source_idx[0]
    elif variant == "protected_role":
        protected["official_validation_used"] = True

    prediction_path = root / "predictions.pt"
    torch.save({
        "context": context.to_dict(),
        "source_idx": source_idx,
        "target_eV": target,
        "prediction_eV": prediction,
        **protected,
    }, prediction_path)
    (root / "contract.json").write_bytes(json_bytes(_recipe(context.family_name)))

    selected_weights = "ema" if adapter == "gptrans-v1" else "live"
    selected = {
        "epoch": 2,
        "optimizer_step": 4,
        "weights": selected_weights,
        "development_mae_eV": 1.0,
    }
    if selected_overrides:
        selected.update(selected_overrides)
    selected_path = root / "selected_model.pt"
    if selected["weights"] in {"live", "ema"}:
        write_selected_state(
            selected_path,
            context,
            model_state={"weight": torch.tensor([0.5])},
            epoch=selected["epoch"],
            optimizer_step=selected["optimizer_step"],
            weights=selected["weights"],
            development_mae_eV=selected["development_mae_eV"],
        )
    else:
        torch.save({
            "context": context.to_dict(),
            "model": {"weight": torch.tensor([0.5])},
            **selected,
        }, selected_path)

    resume = {
        "context": context.to_dict(),
        "model": {"weight": torch.tensor([0.5])},
        "optimizer": {"state": {}, "param_groups": [{"params": [0], "lr": 0.001}]},
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": [],
        },
        "optimizer_step": 4,
        "sample_presentations": 8,
        "cursor": {
            "epoch": 2,
            "next_batch": 0,
            "sampler_order_sha256": "a" * 64,
        },
    }
    if adapter == "k1-v1":
        resume["scheduler"] = {"last_epoch": 2}
    else:
        resume["ema"] = {"weight": torch.tensor([0.5])}
    resume_path = root / "resume.pt"
    write_resume_state(
        resume_path,
        context,
        adapter=adapter,
        model_state=resume["model"],
        optimizer_state=resume["optimizer"],
        rng_state=resume["rng_state"],
        cursor=resume["cursor"],
        optimizer_step=resume["optimizer_step"],
        sample_presentations=resume["sample_presentations"],
        scheduler_state=resume.get("scheduler"),
        ema_state=resume.get("ema"),
    )
    if resume_missing:
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        for key in resume_missing:
            saved.pop(key, None)
        torch.save(saved, resume_path)

    trace_path = root / "trace.json"
    _write_trace(trace_path, context, partial=(variant == "partial_trace"))
    artifacts = {
        "contract": "contract.json",
        "predictions": "predictions.pt",
        "selected_model": "selected_model.pt",
        "resume": "resume.pt",
        "trace": "trace.json",
    }
    progress = {
        "epochs": 2,
        "optimizer_steps": 4,
        "sample_presentations": 8,
    }
    if progress_overrides:
        progress.update(progress_overrides)
    expected = {
        "epochs": 2,
        "optimizer_steps": 4,
        "sample_presentations": 8,
        "development_rows": 3,
        "source_idx_sha256": tensor_digest(SOURCE_IDX, role="source_idx"),
        "target_sha256": tensor_digest(TARGET, role="target"),
        "precision": "fp32",
    }
    if publish:
        write_output_manifest(
            root,
            context,
            adapter=adapter,
            artifacts=artifacts,
            progress=progress,
            runtime=_runtime(context, **(runtime_overrides or {})),
            costs=_costs() if costs is None else costs,
        )
    return root, expected, artifacts, progress


@pytest.mark.parametrize("arm_id,adapter,_weights", FAMILY_CASES)
def test_verified_output_is_bound_to_launch_arm_and_source(
    tmp_path, launch_contexts, arm_id, adapter, _weights
):
    context = launch_contexts[4][arm_id]
    root, expected, _, _ = _write_output(tmp_path, context, adapter)

    result = inspect_output(root, context=context, expected=expected)

    assert result["status"] == "MECHANICALLY_VERIFIED", result
    assert result["blockers"] == []
    assert result["context"] == context.to_dict()
    assert result["scientific_acceptance"] == "NOT_EVALUATED"
    assert result["replay_readiness"] == "NOT_EVALUATED"
    assert result["observed"]["progress"] == {
        "epochs": 2,
        "optimizer_steps": 4,
        "sample_presentations": 8,
    }


@pytest.mark.parametrize(
    "variant,match",
    [
        ("wrong_rows", "Development row identity mismatch"),
        ("wrong_targets", "Development target identity mismatch"),
        ("nan_target", "target must be finite"),
        ("nan_prediction", "Nonfinite or nonfloating predictions"),
        ("wrong_shape", "shape mismatch"),
        ("duplicate_rows", "duplicate source rows"),
        ("protected_role", "Protected role consumption"),
        ("partial_trace", "Trace missing observed exposure"),
    ],
)
def test_invalid_prediction_and_trace_evidence_blocks(
    tmp_path, launch_contexts, variant, match
):
    context = launch_contexts[4]["gptrans_t"]
    root, expected, _, _ = _write_output(
        tmp_path, context, "gptrans-v1", variant=variant
    )

    result = inspect_output(root, context=context, expected=expected)

    assert result["status"] == "BLOCKED"
    assert any(match in blocker for blocker in result["blockers"])


@pytest.mark.parametrize(
    "runtime_overrides,match",
    [
        ({"source_commit": "f" * 40}, "Runtime identity mismatch: source_commit"),
        ({"source_archive_sha256": "f" * 64}, "Runtime identity mismatch: source_archive_sha256"),
        ({"account": "legacy-account"}, "Runtime identity mismatch: account"),
        ({"platform": "ims"}, "Runtime identity mismatch: platform"),
        ({"precision": "bf16"}, "Runtime precision mismatch"),
    ],
)
def test_old_platform_account_precision_or_source_labels_are_not_normalized(
    tmp_path, launch_contexts, runtime_overrides, match
):
    context = launch_contexts[4]["neural_atom_k1"]
    root, expected, _, _ = _write_output(
        tmp_path,
        context,
        "k1-v1",
        runtime_overrides=runtime_overrides,
    )

    result = inspect_output(root, context=context, expected=expected)

    assert result["status"] == "BLOCKED"
    assert any(match in blocker for blocker in result["blockers"])


@pytest.mark.parametrize(
    "arm_id,adapter,missing",
    [
        ("neural_atom_k1", "k1-v1", ["scheduler"]),
        ("neural_atom_k1", "k1-v1", ["rng_state"]),
        ("gptrans_t", "gptrans-v1", ["ema"]),
        ("gptrans_t", "gptrans-v1", ["rng_state"]),
        ("gptrans_t", "gptrans-v1", ["cursor"]),
    ],
)
def test_incomplete_resume_state_is_blocked(
    tmp_path, launch_contexts, arm_id, adapter, missing
):
    context = launch_contexts[4][arm_id]
    root, expected, _, _ = _write_output(
        tmp_path, context, adapter, resume_missing=missing
    )

    result = inspect_output(root, context=context, expected=expected)

    assert result["status"] == "BLOCKED"
    assert any(
        "Incomplete checkpoint/resume state" in blocker or "sampler cursor" in blocker
        for blocker in result["blockers"]
    )


def test_output_from_another_arm_cannot_be_inspected_as_this_arm(tmp_path, launch_contexts):
    contexts = launch_contexts[4]
    root, expected, _, _ = _write_output(
        tmp_path, contexts["gptrans_t"], "gptrans-v1"
    )

    result = inspect_output(root, context=contexts["neural_atom_k1"], expected=expected)

    assert result["status"] == "BLOCKED"
    assert any("Output context/source/arm mismatch" in blocker for blocker in result["blockers"])


def test_artifact_hash_tampering_is_detected(tmp_path, launch_contexts):
    context = launch_contexts[4]["gptrans_t"]
    root, expected, _, _ = _write_output(tmp_path, context, "gptrans-v1")
    with (root / "predictions.pt").open("ab") as stream:
        stream.write(b"tamper")

    result = inspect_output(root, context=context, expected=expected)

    assert result["status"] == "BLOCKED"
    assert any("Artifact hash mismatch: predictions" in blocker for blocker in result["blockers"])


@pytest.mark.parametrize(
    "arm_id,adapter,invalid_weights",
    [
        ("neural_atom_k1", "k1-v1", "ema"),
        ("gptrans_t", "gptrans-v1", "shadow"),
    ],
)
def test_selected_endpoint_must_bind_declared_weights_and_retained_trace(
    tmp_path, launch_contexts, arm_id, adapter, invalid_weights
):
    context = launch_contexts[4][arm_id]
    root, expected, _, _ = _write_output(
        tmp_path,
        context,
        adapter,
        selected_overrides={"weights": invalid_weights},
    )

    result = inspect_output(root, context=context, expected=expected)

    assert result["status"] == "BLOCKED"
    assert result["blockers"]


@pytest.mark.parametrize(
    "selected_overrides",
    [
        {"development_mae_eV": 1.5},
        {"epoch": 1, "optimizer_step": 2},
    ],
)
def test_selected_metric_must_match_predictions_at_selected_trace_row(
    tmp_path, launch_contexts, selected_overrides
):
    context = launch_contexts[4]["gptrans_t"]
    root, expected, _, _ = _write_output(
        tmp_path,
        context,
        "gptrans-v1",
        selected_overrides=selected_overrides,
    )

    result = inspect_output(root, context=context, expected=expected)

    assert result["status"] == "BLOCKED"
    assert any("Selected trace/checkpoint/prediction metric mismatch" in blocker
               for blocker in result["blockers"])


def test_stage_recorder_requires_real_increasing_counts(tmp_path, launch_contexts):
    context = launch_contexts[4]["gptrans_t"]
    path = tmp_path / "stage-trace.json"
    recorder = StageRecorder(path, context, "downstream", METRIC_SEMANTICS)
    recorder.observe(
        optimizer_step=2,
        sample_presentations=4,
        epoch_or_pass=1,
        ema_dev_metric=1.0,
    )

    with pytest.raises(ValueError, match="duplicate x-axis observation|non-monotonic"):
        recorder.observe(
            optimizer_step=2,
            sample_presentations=6,
            epoch_or_pass=2,
            ema_dev_metric=1.0,
        )
    with pytest.raises(ValueError, match="duplicate x-axis observation|non-monotonic"):
        recorder.observe(
            optimizer_step=1,
            sample_presentations=3,
            epoch_or_pass=2,
            ema_dev_metric=1.0,
        )

    trace = load_canonical_trace(path)
    assert len(trace["observations"]) == 1
    assert trace["observations"][0]["optimizer_step"] == 2
    assert trace["observations"][0]["sample_presentations"] == 4


def test_missing_step_cannot_be_silently_fabricated(tmp_path, launch_contexts):
    context = launch_contexts[4]["gptrans_t"]
    path = tmp_path / "stage-trace.json"
    recorder = StageRecorder(path, context, "downstream", METRIC_SEMANTICS)

    with pytest.raises(TypeError, match="optimizer_step"):
        recorder.observe(
            sample_presentations=4,
            epoch_or_pass=1,
            ema_dev_metric=1.0,
        )

    trace = load_canonical_trace(path)
    assert trace["observations"] == []


def test_unknown_adapter_and_family_mismatch_fail_closed(tmp_path, launch_contexts):
    context = launch_contexts[4]["gptrans_t"]
    root, _, artifacts, progress = _write_output(
        tmp_path, context, "gptrans-v1", publish=False
    )
    with pytest.raises(ValueError, match="Unsupported artifact adapter"):
        write_output_manifest(
            root,
            context,
            adapter="arbitrary-v1",
            artifacts=artifacts,
            progress=progress,
            runtime=_runtime(context),
            costs=_costs(),
        )
    with pytest.raises(ValueError, match="adapter/family mismatch"):
        write_output_manifest(
            root,
            context,
            adapter="k1-v1",
            artifacts=artifacts,
            progress=progress,
            runtime=_runtime(context),
            costs=_costs(),
        )
    assert not (root / "output_manifest.json").exists()


def test_unknown_output_format_does_not_fall_back_to_legacy_availability(
    tmp_path, launch_contexts
):
    context = launch_contexts[4]["gptrans_t"]
    root, expected, _, _ = _write_output(tmp_path, context, "gptrans-v1")
    manifest_path = root / "output_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["format"] = "molgap-family-output-v999"
    manifest_path.write_bytes(json_bytes(manifest))

    result = inspect_output(root, context=context, expected=expected)

    assert result["status"] == "BLOCKED"
    assert any("Unsupported output manifest" in blocker for blocker in result["blockers"])


def test_manifest_publication_is_lf_stable_and_conflicts_are_rejected(
    tmp_path, launch_contexts
):
    context = launch_contexts[4]["neural_atom_k1"]
    root, _, artifacts, progress = _write_output(
        tmp_path, context, "k1-v1", publish=False
    )

    path = write_output_manifest(
        root,
        context,
        adapter="k1-v1",
        artifacts=artifacts,
        progress=progress,
        runtime=_runtime(context),
        costs=_costs(),
    )
    first_bytes = path.read_bytes()
    first_mtime = path.stat().st_mtime_ns
    assert b"\r" not in first_bytes
    assert first_bytes.endswith(b"\n")

    assert write_output_manifest(
        root,
        context,
        adapter="k1-v1",
        artifacts=artifacts,
        progress=progress,
        runtime=_runtime(context),
        costs=_costs(),
    ) == path
    assert path.read_bytes() == first_bytes
    assert path.stat().st_mtime_ns == first_mtime

    changed_costs = _costs()
    changed_costs[0]["value"] = 13.0
    with pytest.raises(ValueError, match="conflict|immutable|different"):
        write_output_manifest(
            root,
            context,
            adapter="k1-v1",
            artifacts=artifacts,
            progress=progress,
            runtime=_runtime(context),
            costs=changed_costs,
        )
    assert path.read_bytes() == first_bytes


def test_progress_must_match_expected_rows_and_exposure(tmp_path, launch_contexts):
    context = launch_contexts[4]["gptrans_t"]
    root, expected, _, _ = _write_output(
        tmp_path,
        context,
        "gptrans-v1",
        progress_overrides={"optimizer_steps": 5},
    )

    result = inspect_output(root, context=context, expected=expected)

    assert result["status"] == "BLOCKED"
    assert any("Progress mismatch: optimizer_steps" in blocker for blocker in result["blockers"])


def test_expectations_must_match_hash_pinned_recipe_contract(tmp_path, launch_contexts):
    context = launch_contexts[4]["gptrans_t"]
    root, expected, _, _ = _write_output(tmp_path, context, "gptrans-v1")
    changed = copy.deepcopy(expected)
    changed["epochs"] = 1

    result = inspect_output(root, context=context, expected=changed)

    assert result["status"] == "BLOCKED"
    assert any("frozen recipe contract" in blocker for blocker in result["blockers"])


def test_acceptance_plan_blocks_missing_strict_reference_inputs(
    tmp_path, repo, launch_contexts
):
    spec = launch_contexts[0]

    missing = {"path": "experiments/missing/reference.json", "sha256": "a" * 64}
    expected = _expected_requirements()
    adapters = {"gptrans_t": "gptrans-v1", "neural_atom_k1": "k1-v1"}
    arms = [
        {
            "arm_id": arm["arm_id"],
            "adapter": adapters[arm["arm_id"]],
            "expected": copy.deepcopy(expected),
            "contract": dict(missing),
            "comparison_prelaunch": dict(missing),
            "reference_bundle": dict(missing),
            "reference_artifacts": {
                key: dict(missing)
                for key in REQUIRED_REFERENCE_ARTIFACTS | {"predictions"}
            },
        }
        for arm in spec.to_dict()["arms"]
    ]
    plan = {
        "format": "molgap-family-acceptance-plan-v1",
        "spec_identity": spec.identity,
        "arms": arms,
    }

    result = check_acceptance_plan(spec, repo, plan)

    assert result["status"] == "BLOCKED"
    assert result["trainer_execution"] == "NOT_VERIFIED"
    assert all(arm["status"] == "BLOCKED" for arm in result["arms"])
    assert all(arm["blockers"] for arm in result["arms"])


def test_acceptance_plan_accepts_complete_synthetic_strict_reference_inputs(
    repo, launch_contexts
):
    spec = launch_contexts[0]
    expected = _expected_requirements()
    bundle = copy.deepcopy(_reference_bundle())
    prediction_path = "experiments/reference/predictions.pt"
    prediction_bytes = json_bytes({
        "synthetic": True,
        "source_idx_sha256": expected["source_idx_sha256"],
        "target_sha256": expected["target_sha256"],
        "rows": expected["development_rows"],
    })
    bundle["prediction_manifest"].update({
        "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "source_idx_sha256": expected["source_idx_sha256"],
        "target_sha256": expected["target_sha256"],
        "row_count": expected["development_rows"],
        "unique_source_idx": expected["development_rows"],
    })

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
    artifact_paths = {}
    for name, field in owner_fields.items():
        pointer = bundle[field]
        artifact_path = repo / pointer
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            b"# Synthetic retained reference decision\n"
            if name == "decision"
            else json_bytes({"synthetic": True, "artifact": name})
        )
        artifact_path.write_bytes(content)
        artifact_paths[name] = {
            "path": pointer,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    for name, content in {
        "checkpoint": b"synthetic checkpoint artifact\n",
        "prediction_manifest": json_bytes(bundle["prediction_manifest"]),
    }.items():
        pointer = f"experiments/reference/{name}.json"
        artifact_path = repo / pointer
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(content)
        artifact_paths[name] = {
            "path": pointer,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    prediction_file = repo / prediction_path
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    prediction_file.write_bytes(prediction_bytes)
    artifact_paths["predictions"] = {
        "path": prediction_path,
        "sha256": bundle["prediction_manifest"]["prediction_sha256"],
    }

    plan_root = repo / "family-plan"
    plan_root.mkdir()
    bundle_path = plan_root / "reference_bundle.json"
    prelaunch_path = plan_root / "comparison_prelaunch.json"
    bundle_path.write_bytes(json_bytes(bundle))
    prelaunch = _prelaunch(bundle)
    assert prelaunch["prelaunch_ready"] is True
    prelaunch_path.write_bytes(json_bytes(prelaunch))
    pointer = lambda path: {
        "path": path.resolve().relative_to(repo.resolve()).as_posix(),
        "sha256": file_digest(path),
    }
    adapters = {"gptrans_t": "gptrans-v1", "neural_atom_k1": "k1-v1"}
    arms = []
    for arm in spec.to_dict()["arms"]:
        recipe_path = repo / "recipes" / f"{arm['family']['name']}.json"
        arms.append({
            "arm_id": arm["arm_id"],
            "adapter": adapters[arm["arm_id"]],
            "expected": copy.deepcopy(expected),
            "contract": pointer(recipe_path),
            "comparison_prelaunch": pointer(prelaunch_path),
            "reference_bundle": pointer(bundle_path),
            "reference_artifacts": copy.deepcopy(artifact_paths),
        })
    plan = {
        "format": "molgap-family-acceptance-plan-v1",
        "spec_identity": spec.identity,
        "arms": arms,
    }

    result = check_acceptance_plan(spec, repo, plan)

    assert result["status"] == "ACCEPTANCE_INPUTS_AVAILABLE", result
    assert result["trainer_execution"] == "NOT_VERIFIED"
    assert all(arm["status"] == "AVAILABLE" and arm["blockers"] == []
               for arm in result["arms"])


def _closure_descriptor(tmp_path, repo, launch_contexts, *, alternate_trace_for=None):
    setup_mock_repo(repo)
    spec = launch_contexts[0]
    contexts = launch_contexts[4]
    output_parent = repo / "family-output"
    output_parent.mkdir()
    outputs = {}
    locations = {}
    for arm in spec.to_dict()["arms"]:
        arm_id = arm["arm_id"]
        context = contexts[arm_id]
        adapter = "gptrans-v1" if arm_id == "gptrans_t" else "k1-v1"
        output_dir, expected, _, _ = _write_output(output_parent, context, adapter)
        trajectory_id = f"{context.experiment_id}-{arm_id}"
        run_id = f"{context.logical_run_id}:{arm_id}:downstream"
        folder = f"candidate-{arm_id}"
        candidate = create_candidate_arm(
            repo,
            folder,
            trajectory_id,
            run_id,
            f"ev-family-{arm_id}",
        )
        trajectory = json.loads(candidate["traj_path"].read_bytes())
        trajectory["state_at_start"]["source_commit"] = context.source_commit
        trajectory["actions"][0]["source_commit"] = context.source_commit
        trajectory["actions"][0]["attempt_ids"] = [
            f"{arm_id}-v{context.platform_version}"
        ]
        candidate["traj_path"].write_text(
            json.dumps(trajectory, indent=2) + "\n", encoding="utf-8"
        )
        relative = lambda path: path.resolve().relative_to(repo.resolve()).as_posix()
        terminal = json.loads(candidate["terminal_path"].read_bytes())
        trace_pointer = relative(output_dir / "trace.json")
        trace_sha256 = file_digest(output_dir / "trace.json")
        trace_artifact = next(
            artifact for artifact in terminal["evidence"]["artifacts"]
            if artifact["name"] == "training_trace"
        )
        old_trace_pointer = trace_artifact["locator"]
        trace_artifact["locator"] = trace_pointer
        trace_artifact["sha256"] = trace_sha256
        terminal["artifact_hashes"].pop(old_trace_pointer)
        terminal["artifact_hashes"][trace_pointer] = trace_sha256
        candidate["terminal_path"].write_bytes(json_bytes(terminal))
        outputs[arm_id] = {
            "context": context,
            "output_dir": output_dir,
            "expected": expected,
        }
        locations[arm_id] = {
            "trajectory_id": trajectory_id,
            "run_id": run_id,
            "trajectory": relative(candidate["traj_path"]),
            "terminal": relative(candidate["terminal_path"]),
            "trace": relative(output_dir / "trace.json"),
        }
    if alternate_trace_for is not None:
        alternate_root = outputs[alternate_trace_for]["output_dir"]
        alternate = load_canonical_trace(alternate_root / "trace.json")
        alternate["observations"][-1]["live_train_metric"] = 9.0
        alternate_path = repo / "alternate-same-identity-trace.json"
        alternate_path.write_bytes(json_bytes(alternate))
        locations[alternate_trace_for]["trace"] = relative(alternate_path)
    descriptor = build_verified_terminal_descriptor(
        repo,
        spec,
        outputs=outputs,
        locations=locations,
    )
    return spec, descriptor, outputs


def test_descriptor_rejects_a_different_trace_with_the_same_ids(
    tmp_path, repo, launch_contexts
):
    with pytest.raises(ValueError, match="Descriptor closure trace differs"):
        _closure_descriptor(
            tmp_path,
            repo,
            launch_contexts,
            alternate_trace_for="gptrans_t",
        )


def test_one_blocked_arm_prevents_any_terminal_closure(
    tmp_path, repo, launch_contexts, monkeypatch
):
    spec, descriptor, outputs = _closure_descriptor(tmp_path, repo, launch_contexts)
    outputs["gptrans_t"]["expected"] = copy.deepcopy(outputs["gptrans_t"]["expected"])
    outputs["gptrans_t"]["expected"]["source_idx_sha256"] = "0" * 64
    import molgap.experiment_terminal as terminal

    close = Mock(
        side_effect=AssertionError("terminal closure must not start")
    )
    monkeypatch.setattr(terminal, "execute_terminal_descriptor", close)

    result = close_verified_outputs(repo, spec, descriptor, outputs=outputs)

    assert result["status"] == "BLOCKED"
    assert result["executed"] is False
    assert result["arms"]["gptrans_t"]["status"] == "BLOCKED"
    assert result["arms"]["neural_atom_k1"]["status"] == "MECHANICALLY_VERIFIED"
    close.assert_not_called()


def test_verified_outputs_close_two_independent_rml_trajectories(
    tmp_path, repo, launch_contexts
):
    spec, descriptor, outputs = _closure_descriptor(tmp_path, repo, launch_contexts)
    descriptor_arms = {row["arm_id"]: row for row in descriptor.to_dict()["arms"]}
    original_snapshots = {
        arm_id: (repo / descriptor_arms[arm_id]["trajectory"]).read_bytes()
        for arm_id in outputs
    }

    dry_run = close_verified_outputs(repo, spec, descriptor, outputs=outputs, execute=False)

    assert dry_run["status"] == "MECHANICALLY_VERIFIED", dry_run
    assert dry_run["executed"] is False
    assert all(report["status"] == "MECHANICALLY_VERIFIED"
               for report in dry_run["arms"].values())
    assert all(
        not ((repo / descriptor_arms[arm_id]["trajectory"]).parent / "rml_finalized").exists()
        for arm_id in outputs
    )

    result = close_verified_outputs(repo, spec, descriptor, outputs=outputs)

    assert result["status"] == "COMPLETE", result
    assert result["executed"] is True
    expected_trajectory_ids = {
        arm["trajectory_id"] for arm in descriptor.to_dict()["arms"]
    }
    assert {arm["trajectory_id"] for arm in result["results"]} == expected_trajectory_ids
    assert len(expected_trajectory_ids) == 2
    results_by_trajectory = {arm["trajectory_id"]: arm for arm in result["results"]}
    for arm_id, item in outputs.items():
        descriptor_arm = descriptor_arms[arm_id]
        arm_result = results_by_trajectory[descriptor_arm["trajectory_id"]]
        assert arm_result["pipeline_status"] == "COMPLETE", arm_result
        assert arm_result["validation_status"] == "VALID", arm_result
        trajectory_path = repo / descriptor_arm["trajectory"]
        finalized = trajectory_path.parent / "rml_finalized"
        receipt = verified_receipt(finalized)
        assert receipt["trace_status"] == "available"
        assert (finalized / "prospective_snapshot.json").read_bytes() == original_snapshots[arm_id]
        assert json.loads((finalized / "trajectory.json").read_bytes())["decision"]["final"] is True
        trace_manifest = json.loads((finalized / "trace_manifest.json").read_bytes())
        assert trace_manifest["backtest_eligibility"]["eligible"] is False
        assert (repo / "research_memory/derived").is_dir()
    assert frozen_differences(repo) == []


def test_cli_inspect_output_emits_one_blocked_json_object_and_exit_one(
    tmp_path, launch_contexts
):
    spec, package_dir, package_identity, receipt_path, contexts = launch_contexts
    arm_id = "gptrans_t"
    output_dir, expected, _, _ = _write_output(
        tmp_path, contexts[arm_id], "gptrans-v1"
    )
    expectations_path = tmp_path / "blocked-expectations.json"
    blocked_expected = copy.deepcopy(expected)
    blocked_expected["epochs"] = 1
    expectations_path.write_bytes(json_bytes(blocked_expected))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.to_json(), encoding="utf-8")
    environment = os.environ.copy()
    repo_root = Path(__file__).parents[1]
    environment["PYTHONPATH"] = str(repo_root / "src")

    completed = subprocess.run(
        [
            sys.executable, "-m", "molgap.experiment_cli", "inspect-output",
            "--spec", str(spec_path),
            "--package", str(package_dir),
            "--expected-package-identity", package_identity,
            "--receipt", str(receipt_path),
            "--arm", arm_id,
            "--artifact-root", str(output_dir),
            "--expectations", str(expectations_path),
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1, completed.stderr
    response = json.loads(completed.stdout)
    assert response["status"] == "BLOCKED"
    assert any("frozen recipe contract" in blocker for blocker in response["blockers"])


@pytest.mark.parametrize("arm_id,adapter,weights", FAMILY_CASES)
def test_output_session_round_trips_tensor_events_and_native_unknown_cost(
    tmp_path, launch_contexts, arm_id, adapter, weights
):
    context = launch_contexts[4][arm_id]
    root = tmp_path / f"session-{arm_id}"
    contract = tmp_path / f"contract-{arm_id}.json"
    contract.write_bytes(json_bytes(_recipe(context.family_name)))
    session = FamilyOutputSession(
        root,
        context,
        adapter=adapter,
        contract=contract,
        trajectory_id=f"{context.experiment_id}-{arm_id}",
        metric_semantics=METRIC_SEMANTICS,
    )
    session.epoch_finished(
        epoch=1,
        optimizer_step=2,
        sample_presentations=4,
        live_train_metric=2.0,
        live_dev_metric=2.0,
        ema_dev_metric=2.0,
    )
    session.epoch_finished(
        epoch=2,
        optimizer_step=4,
        sample_presentations=8,
        live_train_metric=1.1,
        live_dev_metric=1.0,
        ema_dev_metric=1.0,
    )
    session.selected(
        model_state={"weight": torch.tensor([0.5])},
        epoch=2,
        optimizer_step=4,
        weights=weights,
        prediction_eV=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        target_eV=TARGET,
        source_idx=SOURCE_IDX,
    )
    session.checkpoint(
        model_state={"weight": torch.tensor([0.5])},
        optimizer_state={"state": {}, "param_groups": [{"params": [0], "lr": 0.001}]},
        cursor={"epoch": 2, "next_batch": 0, "sampler_order_sha256": "a" * 64},
        optimizer_step=4,
        sample_presentations=8,
        rng_state={
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": [],
        },
        scheduler_state={"last_epoch": 2} if adapter == "k1-v1" else None,
        ema_state={"weight": torch.tensor([0.5])} if adapter == "gptrans-v1" else None,
    )

    result = session.complete(runtime=_runtime(context), hardware="synthetic-cpu")

    assert result["status"] == "MECHANICALLY_VERIFIED", result
    manifest = json.loads((root / "output_manifest.json").read_bytes())
    wall, device = manifest["costs"]
    assert (wall["metric"], wall["status"], wall["semantics"]) == (
        "wall_seconds", "measured", "process_wall"
    )
    assert wall["value"] >= 0
    assert (device["metric"], device["status"], device["value"]) == (
        "device_seconds", "missing", None
    )


def test_repeated_rml_finalization_reports_existing_receipt_without_overwrite(tmp_path):
    root = tmp_path / "mock-rml"
    root.mkdir()
    setup_mock_repo(root)
    arm = create_candidate_arm(
        root,
        "exp_repeat_finalize",
        "TB-repeat-finalize",
        "run-repeat-finalize",
        "ev-repeat-finalize",
    )

    first = close_terminal_arm(
        repo_root=root,
        trajectory=arm["traj_path"],
        terminal=arm["terminal_path"],
        trace=None,
    )
    receipt_path = arm["exp_dir"] / "rml_finalized/finalization.json"
    original = receipt_path.read_bytes()
    second = close_terminal_arm(
        repo_root=root,
        trajectory=arm["traj_path"],
        terminal=arm["terminal_path"],
        trace=None,
    )

    assert first["finalization_status"] == "FINALIZED"
    assert second["pipeline_status"] == "COMPLETE"
    assert second["finalization_status"] == "ALREADY_FINALIZED"
    assert receipt_path.read_bytes() == original
