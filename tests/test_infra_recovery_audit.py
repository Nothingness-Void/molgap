"""Focused regression probes for recovery and state acceptance invariants.

Every test uses an existing tiny fixture or a pure tensor model. Negative
cases reject corrupt metadata; positive cases preserve supported restoration.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import torch

pytest_plugins = ["test_experiment_workflow_resume", "test_experiment_family_workflow"]

from molgap import experiment_workflow_resume as resume_workflow
from molgap import kaggle_workflow
from molgap.experiment_family_workflow import inspect_output
from molgap.experiment_resume import (
    create_resume_bundle,
    validate_resume_bundle,
)
from molgap.experiment_workflow_resume import prepare_resumed_workflow
from molgap.family_training_session import validate_session_resume
from molgap.family_training_state import FamilyTrainingState
from molgap.gptrans_execution_hotpath import BoundForeachEMA
from molgap.research_memory.trace import file_digest, json_bytes
from test_experiment_family_workflow import _write_output
from test_experiment_resume import _context as resume_context
from test_experiment_resume import _source as resume_source
from test_experiment_resume import _trajectory as resume_trajectory
from test_family_training_session import events as session_events
from test_family_training_session import make_session
from test_family_training_state import calibrate, factory, make_case
from test_experiment_workflow_resume import _recovery_plan


def test_recovery_report_write_failure_preserves_staging_failure(
    tmp_path, resume_case, monkeypatch,
):
    case = resume_case

    def fail_stage(*args, **kwargs):
        raise OSError("staging failure")

    def fail_report(*args, **kwargs):
        raise OSError("report publication failure")

    monkeypatch.setattr(kaggle_workflow, "stage_inputs", fail_stage)
    monkeypatch.setattr(resume_workflow, "_atomic", fail_report)

    with pytest.raises(OSError, match="staging failure"):
        prepare_resumed_workflow(
            case["spec"], case["repo"], case["prepared"], case["package"],
            case["package_identity"], case["receipt"], tmp_path / "resumed",
            plan=_recovery_plan(case),
        )


def test_completed_resume_cursor_rejects_nonzero_next_batch(tmp_path, launch_contexts):
    """A completed output cannot carry a pending sampler batch."""
    context = launch_contexts[4]["gptrans_t"]
    root, expected, _, _ = _write_output(tmp_path, context, "gptrans-v1")
    resume_path = root / "resume.pt"
    state = torch.load(resume_path, map_location="cpu", weights_only=True)
    state["cursor"]["next_batch"] = 99
    torch.save(state, resume_path)
    manifest_path = root / "output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["resume"]["sha256"] = file_digest(resume_path)
    manifest_path.write_bytes(json_bytes(manifest))

    result = inspect_output(root, context=context, expected=expected)

    assert result["status"] == "BLOCKED"
    assert result["blockers"]


def test_completed_resume_cursor_zero_next_batch_remains_accepted(tmp_path, launch_contexts):
    """A normal completed cursor remains mechanically verifiable."""
    context = launch_contexts[4]["gptrans_t"]
    root, expected, _, _ = _write_output(tmp_path, context, "gptrans-v1")

    result = inspect_output(root, context=context, expected=expected)

    assert result["status"] == "MECHANICALLY_VERIFIED", result
    assert result["blockers"] == []


def test_resume_bundle_rejects_primary_restore_path_alias(tmp_path):
    """Two primary artifacts cannot restore to the same destination."""
    context = resume_context()
    source = resume_source(tmp_path, context)
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps(resume_trajectory(context)), encoding="utf-8")
    bundle = tmp_path / "bundle"
    create_resume_bundle(source, context, trajectory_path, bundle)

    manifest_path = bundle / "resume_bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["trace"]["restore_path"] = (
        manifest["artifacts"]["checkpoint"]["restore_path"]
    )
    manifest_path.write_bytes(json_bytes(manifest))

    with pytest.raises(ValueError, match="alias|restore"):
        validate_resume_bundle(bundle, context=context, trajectory=trajectory_path)


def test_resume_bundle_rejects_unrecognized_context_sidecar(tmp_path):
    """A context sidecar must carry the declared context identity."""
    context = resume_context()
    source = resume_source(tmp_path, context)
    (source / "bad_context.json").write_bytes(json_bytes({"unrelated": True}))
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps(resume_trajectory(context)), encoding="utf-8")
    bundle = tmp_path / "bundle"
    create_resume_bundle(source, context, trajectory_path, bundle)

    with pytest.raises(ValueError, match="context"):
        validate_resume_bundle(bundle, context=context, trajectory=trajectory_path)


def test_clean_ema_restore_requires_the_declared_calibration_policy():
    """A ready Clean-BN state cannot load without its policy identity."""
    owner, _, batches, policy = make_case()
    calibrate(owner, batches, policy)
    saved = owner.state_dict()
    fresh_live = factory()
    fresh = FamilyTrainingState(
        fresh_live,
        BoundForeachEMA(fresh_live),
        clean_factory=factory,
    )

    with pytest.raises(ValueError, match="calibrat|policy"):
        fresh.load_state_dict(saved)


def test_ssma_resume_rejects_an_unacknowledged_optimizer_moment():
    """SSMA may skip only its registered conditional parameters."""
    log = session_events()
    session = make_session(
        "neural_atom_k1",
        development=False,
        schedule_kind="cosine_step",
        events=log,
    )
    try:
        session.run()
        state = copy.deepcopy(session.state_dict())
    finally:
        session.close()

    recipe = replace(session.recipe, mode="ssma")
    state["cursor"]["recipe_identity"] = recipe.identity
    removed = min(state["optimizer"]["state"])
    del state["optimizer"]["state"][removed]
    state["cursor"]["optimizer_state_ids"].remove(removed)

    with pytest.raises(ValueError, match="optimizer|moment"):
        validate_session_resume(recipe, state)


def test_ssma_resume_allows_only_registered_addon_tail_without_moment():
    """SSMA may omit a moment for an appended addon parameter tail."""
    log = session_events()
    session = make_session(
        "neural_atom_k1",
        development=False,
        schedule_kind="cosine_step",
        events=log,
    )
    try:
        session.run()
        state = copy.deepcopy(session.state_dict())
        recipe = replace(session.recipe, mode="ssma")
    finally:
        session.close()

    state["cursor"]["recipe_identity"] = recipe.identity
    addon_weight = "k1_joint_aggregation.return_projection.weight"
    addon_affine_weight = "k1_joint_aggregation.affine_weight"
    addon_affine_bias = "k1_joint_aggregation.affine_bias"
    state["model"][addon_weight] = torch.zeros(1, 1)
    state["model"][addon_affine_weight] = torch.zeros(1, 1)
    state["model"][addon_affine_bias] = torch.zeros(1)
    state["ema"][addon_weight] = torch.zeros(1, 1)
    state["ema"][addon_affine_weight] = torch.zeros(1, 1)
    state["ema"][addon_affine_bias] = torch.zeros(1)
    state["optimizer"]["param_groups"][0]["params"].append(4)

    progress = validate_session_resume(recipe, state)

    assert progress.status == state["cursor"]["status"]
