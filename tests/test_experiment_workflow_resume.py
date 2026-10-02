"""Local recovery tests over the registered workflow owners.

The fixture uses synthetic CPU artifacts, but it drives the real preparation,
package, receipt, resume transport, staging, and release gates.  The owner
hook is intentionally stateful: a bundle is accepted only when its checkpoint
and trace agree with the bound arm and source archive.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

pytest_plugins = ["test_experiment_workflow"]

from molgap import experiment_launch as launch
from molgap import experiment_workflow as workflow
from molgap import experiment_workflow_resume as resume_workflow
from molgap import kaggle_workflow
from molgap import kaggle_pair_runtime
from molgap.experiment_family_workflow import RunContext, incomplete_observations_from_execution
from molgap.experiment_package import verify_experiment_source_package
from molgap.experiment_resume import RESUME_BUNDLE_FILE
from molgap.experiment_workflow_resume import (
    RESUME_PLAN_FORMAT,
    build_workflow_resume,
    prepare_resumed_workflow,
)
from molgap.experiment_retention import validate_execution_retention
from molgap.research_memory.trace import json_bytes

from test_experiment_lifecycle_integration import (
    _install_synthetic_owner,
    _launch_receipt,
    _prepare_reference_inputs,
    _rewrite_case,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _install_resume_owner(module_path: Path) -> None:
    """Add a real synthetic owner validator to the temporary source package."""
    module_path.write_text(
        module_path.read_text(encoding="utf-8")
        + """

def validate_screen_resume(spec, arm_id, manifest, bundle_root, artifacts,
                           sidecars, trajectory, context=None):
    import hashlib
    import json
    import torch

    checkpoint = torch.load(artifacts["checkpoint"], map_location="cpu",
                            weights_only=True)
    if checkpoint.get("format") != "synthetic-resume-v1":
        raise ValueError("synthetic resume checkpoint format mismatch")
    if checkpoint.get("arm_id") != arm_id:
        raise ValueError("synthetic resume checkpoint arm mismatch")
    if context is not None:
        if checkpoint.get("source_archive_sha256") != context["source_archive_sha256"]:
            raise ValueError("synthetic resume checkpoint source mismatch")
    trace_raw = artifacts["trace"].read_bytes()
    trace = json.loads(trace_raw.decode("utf-8"))
    if trace.get("arm_id") != arm_id:
        raise ValueError("synthetic resume trace arm mismatch")
    if checkpoint.get("trace_sha256") != hashlib.sha256(trace_raw).hexdigest():
        raise ValueError("synthetic resume checkpoint trace mismatch")
    if manifest.get("cursor") != checkpoint.get("cursor"):
        raise ValueError("synthetic resume cursor mismatch")
    if trajectory.get("decision", {}).get("outcome") not in (None, "ACTIVE"):
        raise ValueError("synthetic resume trajectory is terminal")
    return {"owner": "synthetic", "cursor": dict(checkpoint["cursor"]),
            "trace_rows": len(trace.get("rows", []))}
""",
        encoding="utf-8",
    )
    # The source package is built from committed bytes.  Reload the already
    # registered module so all subsequent owner calls use the same hook.
    import sys

    module = sys.modules["molgap.synthetic_trainer"]
    exec(compile(module_path.read_bytes(), str(module_path), "exec"), module.__dict__)

    import subprocess

    subprocess.run(["git", "add", module_path.relative_to(module_path.parents[2]).as_posix()],
                   cwd=module_path.parents[2], check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic resume owner"],
        cwd=module_path.parents[2], check=True, capture_output=True, text=True,
    )


def _write_incomplete_output(repo: Path, context: RunContext, arm_id: str) -> Path:
    """Write one owner-native incomplete prefix without running training."""
    output = repo / "retained" / "incomplete" / arm_id
    output.mkdir(parents=True)
    trace = {
        "format": "synthetic-resume-trace-v1",
        "arm_id": arm_id,
        "rows": [{"epoch": 0, "optimizer_steps": 2, "sample_presentations": 4}],
    }
    trace_raw = json_bytes(trace)
    (output / "trace.json").write_bytes(trace_raw)
    torch.save(
        {
            "format": "synthetic-resume-v1",
            "arm_id": arm_id,
            "source_archive_sha256": context.source_archive_sha256,
            "trace_sha256": _sha(trace_raw),
            "cursor": {
                "epoch": 1,
                "next_batch": 0,
                "optimizer_step": 2,
                "sample_presentations": 4,
            },
        },
        output / "last_checkpoint.pt",
    )
    return output


def _prior_allocation_ledger(spec, contexts: dict[str, RunContext]) -> dict:
    """Build one retained pair-allocation segment for every resume bundle."""
    arm_ids = sorted(contexts)
    hardware = ["Tesla T4", "Tesla T4"]
    return {
        "format": "molgap-allocation-ledger-v1",
        "spec_identity": spec.identity,
        "started_at": "2026-10-02T00:00:00+00:00",
        "observed_at": "2026-10-02T00:00:07+00:00",
        "status": "complete",
        "allocation_device_count": 2,
        "wall_seconds": 7.0,
        "allocated_device_seconds": 14.0,
        "unassigned_device_seconds": 0.0,
        "devices": [
            {"device": index, "hardware": name, "arm_id": arm_ids[index],
             "allocated_seconds": 7.0}
            for index, name in enumerate(hardware)
        ],
        "scope": "Python bootstrap/runtime observation window; includes idle allocated devices",
        "provisioning_before_python_seconds": {"value": None, "status": "measurement_missing"},
        "queue_seconds": {"value": None, "status": "measurement_missing"},
        "prior_segments": [],
        "prior_unobserved_intervals": False,
    }


@pytest.fixture
def resume_case(tmp_path, workflow_case, monkeypatch):
    """Prepare the original workflow and export one verified bundle per arm."""
    repo = workflow_case["repo"]
    _prepare_reference_inputs(repo)
    module_path = _install_synthetic_owner(repo, monkeypatch)
    _install_resume_owner(module_path)
    spec, plan, expected, recipes = _rewrite_case(workflow_case, repo, module_path)

    from molgap.experiment_workflow import prepare_workflow

    original = prepare_workflow(spec, repo, plan, tmp_path / "prepared")
    assert original["status"] == "PREPARED_FOR_PLATFORM", original
    prepared = tmp_path / "prepared"
    package_dir = prepared / "package"
    receipt = _launch_receipt(spec, package_dir, original["package_identity"], repo)
    contexts = {
        arm["arm_id"]: RunContext.from_launch(
            spec,
            receipt,
            package_dir,
            expected_package_identity=original["package_identity"],
            arm_id=arm["arm_id"],
        )
        for arm in spec.to_dict()["arms"]
    }
    sources = {
        arm_id: _write_incomplete_output(repo, context, arm_id)
        for arm_id, context in contexts.items()
    }
    prior_ledger = _prior_allocation_ledger(spec, contexts)
    for source in sources.values():
        (source / "allocation_ledger.json").write_bytes(json_bytes(prior_ledger))
    bundles = {}
    for arm_id, source in sources.items():
        destination = tmp_path / "bundles" / arm_id
        bundles[arm_id] = build_workflow_resume(
            spec,
            repo,
            prepared,
            package_dir,
            original["package_identity"],
            receipt,
            destination,
            arm_id=arm_id,
            source_output=source,
        )
        assert bundles[arm_id]["status"] == "RESUME_VERIFIED"
        assert (destination / RESUME_BUNDLE_FILE).is_file()
    return {
        "repo": repo,
        "spec": spec,
        "plan": plan,
        "prepared": prepared,
        "package": package_dir,
        "receipt": receipt,
        "package_identity": original["package_identity"],
        "contexts": contexts,
        "sources": sources,
        "bundles": bundles,
        "prior_ledger": prior_ledger,
    }


def _recovery_plan(case: dict, *, arms: dict | None = None) -> dict:
    return {
        "format": RESUME_PLAN_FORMAT,
        "spec_identity": case["spec"].identity,
        "arms": arms if arms is not None else {
            arm_id: item["manifest"] for arm_id, item in case["bundles"].items()
        },
    }


def _replace_active_trajectory(case: dict, arm_id: str, marker: str) -> Path:
    """Replace one canonical trajectory with another still-active valid record."""
    binding = next(item for item in case["spec"].to_dict()["prospective"]["arms"]
                   if item["arm_id"] == arm_id)
    trajectory_path = case["repo"] / binding["output"] / "trajectory.json"
    changed = json.loads(trajectory_path.read_bytes())
    changed["question"] = marker
    assert changed["decision"]["outcome"] == "ACTIVE"
    trajectory_path.write_bytes(json_bytes(changed))
    return trajectory_path


def test_build_then_prepare_resume_reuses_original_release_without_planning(
    tmp_path, resume_case, monkeypatch,
):
    case = resume_case
    original_trajectories = {
        arm["arm_id"]: (
            case["prepared"] / "source_dataset" / "prospective" / arm["arm_id"] / "trajectory.json"
        ).read_bytes()
        for arm in case["spec"].to_dict()["arms"]
    }

    def planning_is_forbidden(*args, **kwargs):
        raise AssertionError("resume preparation must not plan a new prospective action")

    monkeypatch.setattr(workflow, "plan_prospective", planning_is_forbidden)
    resumed = prepare_resumed_workflow(
        case["spec"],
        case["repo"],
        case["prepared"],
        case["package"],
        case["package_identity"],
        case["receipt"],
        tmp_path / "resumed",
        plan=_recovery_plan(case),
    )

    assert resumed["status"] == "PREPARED_FOR_PLATFORM", resumed
    assert resumed["prospective_published"] is False
    assert resumed["submitted"] is False
    for arm_id, expected in original_trajectories.items():
        staged = tmp_path / "resumed" / "source_dataset" / "prospective" / arm_id / "trajectory.json"
        assert staged.read_bytes() == expected
        assert (tmp_path / "resumed" / "source_dataset" / "resume" / arm_id / RESUME_BUNDLE_FILE).is_file()
    launch_config = json.loads((tmp_path / "resumed" / "source_dataset" / "experiment_launch.json").read_bytes())
    assert set(launch_config["resume"]) == set(original_trajectories)


def test_build_resume_requires_a_reconciled_receipt(tmp_path, resume_case):
    case = resume_case
    pending = launch.build_launch_receipt(
        case["spec"], case["package"], expected_package_identity=case["package_identity"]
    )
    pending_dir = case["repo"] / "pending-receipt"
    pending_dir.mkdir()
    pending_path = launch.write_launch_receipt(
        launch.canonical_json(pending),
        pending_dir,
        case["spec"],
        case["package"],
        expected_package_identity=case["package_identity"],
    )
    arm_id = next(iter(case["bundles"]))
    with pytest.raises(ValueError, match="reconciled observed launch"):
        build_workflow_resume(
            case["spec"], case["repo"], case["prepared"], case["package"],
            case["package_identity"], pending_path, tmp_path / "pending-bundle",
            arm_id=arm_id, source_output=case["sources"][arm_id],
        )


def test_changed_canonical_trajectory_rejects_resume_export(resume_case):
    case = resume_case
    arm_id = next(iter(case["bundles"]))
    binding = next(item for item in case["spec"].to_dict()["prospective"]["arms"]
                   if item["arm_id"] == arm_id)
    trajectory_path = case["repo"] / binding["output"] / "trajectory.json"
    changed = json.loads(trajectory_path.read_bytes())
    changed["question"] = "changed after original release"
    trajectory_path.write_bytes(json_bytes(changed))

    with pytest.raises(ValueError, match="exact original ACTIVE canonical prospective trajectory"):
        build_workflow_resume(
            case["spec"], case["repo"], case["prepared"], case["package"],
            case["package_identity"], case["receipt"], case["repo"] / "changed-bundle",
            arm_id=arm_id, source_output=case["sources"][arm_id],
        )


def test_build_resume_rejects_active_trajectory_replaced_before_bundle_export(
    resume_case, tmp_path, monkeypatch,
):
    case = resume_case
    arm_id = next(iter(case["bundles"]))
    trajectory_path = case["repo"] / next(
        item for item in case["spec"].to_dict()["prospective"]["arms"]
        if item["arm_id"] == arm_id
    )["output"] / "trajectory.json"
    original_create = resume_workflow.create_resume_bundle

    def replace_then_export(*args, **kwargs):
        _replace_active_trajectory(case, arm_id, "changed before bundle export")
        return original_create(*args, **kwargs)

    monkeypatch.setattr(resume_workflow, "create_resume_bundle", replace_then_export)
    with pytest.raises(ValueError, match="(?i)(trajectory|prospective)"):
        build_workflow_resume(
            case["spec"], case["repo"], case["prepared"], case["package"],
            case["package_identity"], case["receipt"], tmp_path / "raced-bundle",
            arm_id=arm_id, source_output=case["sources"][arm_id],
        )
    assert trajectory_path.read_bytes() != case["prepared"].joinpath(
        "source_dataset", "prospective", arm_id, "trajectory.json"
    ).read_bytes()


def test_completed_canonical_arm_rejects_resume_export(resume_case):
    case = resume_case
    arm_id = next(iter(case["bundles"]))
    binding = next(item for item in case["spec"].to_dict()["prospective"]["arms"]
                   if item["arm_id"] == arm_id)
    trajectory_path = case["repo"] / binding["output"] / "trajectory.json"
    completed = json.loads(trajectory_path.read_bytes())
    completed["decision"]["outcome"] = "COMPLETE"
    trajectory_path.write_bytes(json_bytes(completed))

    with pytest.raises(ValueError, match="exact original ACTIVE canonical prospective trajectory"):
        build_workflow_resume(
            case["spec"], case["repo"], case["prepared"], case["package"],
            case["package_identity"], case["receipt"], case["repo"] / "completed-bundle",
            arm_id=arm_id, source_output=case["sources"][arm_id],
        )


def test_prepare_resume_requires_all_spec_arms(resume_case, tmp_path):
    case = resume_case
    arm_id = next(iter(case["bundles"]))
    with pytest.raises(ValueError, match="bind every Spec arm exactly once"):
        prepare_resumed_workflow(
            case["spec"], case["repo"], case["prepared"], case["package"],
            case["package_identity"], case["receipt"], tmp_path / "missing-arm",
            plan=_recovery_plan(case, arms={arm_id: case["bundles"][arm_id]["manifest"]}),
        )


def test_prepare_resume_rejects_active_trajectory_replaced_before_freeze(
    resume_case, tmp_path, monkeypatch,
):
    case = resume_case
    arm_id = next(iter(case["bundles"]))
    original_freeze = kaggle_workflow.freeze_inputs

    def replace_then_freeze(stage, trajectories):
        _replace_active_trajectory(case, arm_id, "changed before resume freeze")
        return original_freeze(stage, trajectories)

    monkeypatch.setattr(kaggle_workflow, "freeze_inputs", replace_then_freeze)
    with pytest.raises(ValueError, match="(?i)(trajectory|prospective)"):
        prepare_resumed_workflow(
            case["spec"], case["repo"], case["prepared"], case["package"],
            case["package_identity"], case["receipt"], tmp_path / "raced-resume",
            plan=_recovery_plan(case),
        )


def test_resume_checkpoint_modified_after_stage_is_rejected_before_post(resume_case, tmp_path, monkeypatch):
    case = resume_case
    original_freeze = kaggle_workflow.freeze_inputs
    arm_id = next(iter(case["bundles"]))

    def freeze_then_tamper(stage, trajectories):
        original_freeze(stage, trajectories)
        checkpoint = stage["input_root"] / "resume" / arm_id / "files" / "last_checkpoint.pt"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered-after-stage")

    monkeypatch.setattr(kaggle_workflow, "freeze_inputs", freeze_then_tamper)
    output = tmp_path / "tampered-resume"
    resumed = prepare_resumed_workflow(
        case["spec"], case["repo"], case["prepared"], case["package"],
        case["package_identity"], case["receipt"], output,
        plan=_recovery_plan(case),
    )

    assert resumed["status"] == "BLOCKED", resumed
    assert resumed["submitted"] is False
    assert resumed["errors"]
    assert any("Resume artifact hash mismatch" in error["message"] for error in resumed["errors"])
    assert (output / "release_report.json").is_file()
    assert json.loads((output / "release_report.json").read_bytes())["errors"]


class _ImmediateWorker:
    """Minimal Popen replacement: no trainer/model process is started."""

    def __init__(self, returncode: int):
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        return None

    def kill(self):
        return None


def _patch_pair_runtime(monkeypatch, commands: list[list[str]], *, failed_phase=None):
    monkeypatch.setattr(
        kaggle_pair_runtime.subprocess,
        "check_output",
        lambda *args, **kwargs: "Tesla T4\nTesla T4\n",
    )

    def fake_popen(command, **kwargs):
        commands.append(list(command))
        phase = command[command.index("--phase") + 1]
        return _ImmediateWorker(1 if phase == failed_phase else 0)

    monkeypatch.setattr(kaggle_pair_runtime.subprocess, "Popen", fake_popen)


def _command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_resumed_pair_runtime_restores_bundles_and_binds_resume_paths_and_retention(
    tmp_path, resume_case, monkeypatch,
):
    case = resume_case
    resumed_root = tmp_path / "resumed-runtime"
    resumed = prepare_resumed_workflow(
        case["spec"], case["repo"], case["prepared"], case["package"],
        case["package_identity"], case["receipt"], resumed_root,
        plan=_recovery_plan(case),
    )
    assert resumed["status"] == "PREPARED_FOR_PLATFORM", resumed

    runtime_root = tmp_path / "execution-root"
    commands: list[list[str]] = []
    _patch_pair_runtime(monkeypatch, commands)
    manifest = verify_experiment_source_package(case["package"])
    report = kaggle_pair_runtime.run_two_phase_pair(
        source_root=case["repo"],
        package_dir=case["package"],
        input_root=resumed_root / "source_dataset",
        launch_path=resumed_root / "source_dataset" / "experiment_launch.json",
        output=runtime_root,
        manifest=manifest,
    )

    assert report["status"] == "complete"
    assert len(commands) == len(case["spec"].to_dict()["arms"]) * 2
    for arm_id in case["bundles"]:
        restored = runtime_root / arm_id
        assert (restored / "last_checkpoint.pt").is_file()
        assert (restored / "trace.json").is_file()
        assert (restored / "allocation_ledger.json").is_file()

    by_phase = {(cmd[cmd.index("--phase") + 1], _command_value(cmd, "--arm")): cmd
                for cmd in commands}
    for arm_id in case["bundles"]:
        preflight = by_phase[("preflight", arm_id)]
        assert Path(_command_value(preflight, "--output")) == (
            runtime_root / "_resume_preflight" / arm_id
        )
        assert Path(_command_value(preflight, "--resume-output")) == runtime_root / arm_id
        assert "--preflight-output" not in preflight

        train = by_phase[("train", arm_id)]
        assert Path(_command_value(train, "--output")) == runtime_root / arm_id
        assert Path(_command_value(train, "--preflight-output")) == (
            runtime_root / "_resume_preflight" / arm_id
        )
        assert "--resume-output" not in train

    retention = validate_execution_retention(
        runtime_root,
        case["spec"],
        package_identity=case["package_identity"],
        source_commit=manifest["source_commit"],
        source_archive_sha256=manifest["archive_sha256"],
    )
    root_ledger = retention["ledgers"]["root"]
    assert retention["manifest"]["package_identity"] == case["package_identity"]
    assert retention["manifest"]["source_commit"] == manifest["source_commit"]
    assert retention["manifest"]["source_archive_sha256"] == manifest["archive_sha256"]
    assert root_ledger["prior_segments"] == [case["prior_ledger"]]
    assert root_ledger["allocated_device_seconds"] == pytest.approx(
        2 * root_ledger["wall_seconds"]
    )
    assert all(ledger == root_ledger for ledger in retention["ledgers"]["arms"].values())

    with pytest.raises(ValueError, match="fresh output root"):
        kaggle_pair_runtime.run_two_phase_pair(
            source_root=case["repo"], package_dir=case["package"],
            input_root=resumed_root / "source_dataset",
            launch_path=resumed_root / "source_dataset" / "experiment_launch.json",
            output=runtime_root, manifest=manifest,
        )


def test_failed_resumed_preflight_retains_training_started_and_unknown_progress(
    tmp_path, resume_case, monkeypatch,
):
    case = resume_case
    resumed_root = tmp_path / "resumed-failure"
    resumed = prepare_resumed_workflow(
        case["spec"], case["repo"], case["prepared"], case["package"],
        case["package_identity"], case["receipt"], resumed_root,
        plan=_recovery_plan(case),
    )
    assert resumed["status"] == "PREPARED_FOR_PLATFORM", resumed

    commands: list[list[str]] = []
    _patch_pair_runtime(monkeypatch, commands, failed_phase="preflight")
    manifest = verify_experiment_source_package(case["package"])
    runtime_root = tmp_path / "execution-failure"
    with pytest.raises(RuntimeError, match="Arm failed during preflight"):
        kaggle_pair_runtime.run_two_phase_pair(
            source_root=case["repo"],
            package_dir=case["package"],
            input_root=resumed_root / "source_dataset",
            launch_path=resumed_root / "source_dataset" / "experiment_launch.json",
            output=runtime_root,
            manifest=manifest,
        )

    state = json.loads((runtime_root / "pair_state.json").read_bytes())
    assert state["status"] == "failed"
    assert all(arm["training_started"] is True for arm in state["arms"].values())
    assert all(arm["terminal_status"] == "failed" for arm in state["arms"].values())
    observations = incomplete_observations_from_execution(case["spec"], state)
    assert all(
        all(value is None for value in observation["progress"].values())
        for observation in observations.values()
    )
    assert all(0 not in observation["progress"].values() for observation in observations.values())
