"""Focused offline audit regressions for launch and pair-runtime durability."""

import json

import pytest

import test_experiment_workflow as fixtures
from molgap import kaggle_pair_runtime as runtime
from molgap import kaggle_submission as submission
from test_kaggle_submission import release as submission_release


def test_pair_phase_publication_failure_preserves_spawn_failure(tmp_path, monkeypatch):
    """A terminal receipt failure must not hide the worker/spawn exception."""
    from molgap import experiment_package

    spec = fixtures._candidate_pair_spec()
    jobs = fixtures._jobs(spec)
    source, package, inputs, output = [tmp_path / name for name in
                                       ("source", "package", "inputs", "output")]
    for path in (source, package, inputs):
        path.mkdir()
    (package / "experiment_spec.json").write_text(spec.to_json(), encoding="utf-8")
    launch = inputs / "experiment_launch.json"
    launch.write_text(json.dumps({
        "expected_package_identity": fixtures.HEX_A,
        "expected_source_archive_sha256": fixtures.HEX_B,
        "jobs": jobs,
    }), encoding="utf-8")
    monkeypatch.setattr(runtime.subprocess, "check_output",
                        lambda *args, **kwargs: "Tesla T4\nTesla T4\n")
    monkeypatch.setattr(
        experiment_package,
        "verify_experiment_source_package",
        lambda _package: {
            "package_identity": fixtures.HEX_A,
            "spec_identity": spec.identity,
            "archive_sha256": fixtures.HEX_B,
            "source_commit": "c" * 40,
        },
    )

    class Process:
        def __init__(self, phase):
            self.returncode = 0 if phase == "preflight" else None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    original = OSError("synthetic train spawn failure")
    publication = OSError("synthetic pair-state publication failure")
    spawn_failed = False

    def spawn(command, **kwargs):
        nonlocal spawn_failed
        arm = command[command.index("--arm") + 1]
        phase = command[command.index("--phase") + 1]
        if phase == "train" and arm == jobs[1]["arm_id"]:
            spawn_failed = True
            raise original
        return Process(phase)

    real_atomic_json = runtime.atomic_json

    def atomic_json(path, value):
        if spawn_failed and path.name == "pair_state.json":
            raise publication
        return real_atomic_json(path, value)

    monkeypatch.setattr(runtime.subprocess, "Popen", spawn)
    monkeypatch.setattr(runtime, "atomic_json", atomic_json)

    with pytest.raises(OSError) as caught:
        runtime.run_two_phase_pair(source_root=source, package_dir=package,
                                   input_root=inputs, launch_path=launch,
                                   output=output)

    assert caught.value is original


def test_submission_journal_failure_preserves_push_transport_fault(
        submission_release, monkeypatch):
    """A failed SUBMIT_UNKNOWN journal write must not hide the transport fault."""
    primary = RuntimeError("synthetic push transport fault")
    secondary = OSError("synthetic submission journal fault")

    def push(**kwargs):
        raise primary

    real_atomic_json = submission.atomic_json

    def atomic_json(path, value):
        if (path.name == "submission_state.json"
                and value.get("phase") == "SUBMIT_UNKNOWN"):
            raise secondary
        return real_atomic_json(path, value)

    monkeypatch.setattr(submission, "push_kernel_with_accelerator", push)
    monkeypatch.setattr(submission, "atomic_json", atomic_json)

    with pytest.raises(RuntimeError) as caught:
        submission_release.run()

    assert caught.value is primary


def test_rejects_resume_without_continuation_adapter(submission_release):
    """The generic first-run submitter must reject local-only recovery staging."""
    launch_path = submission_release.staged / "experiment_launch.json"
    launch = json.loads(launch_path.read_bytes())
    resume = {}
    for arm_id in ("original", "optimized"):
        relative = f"resume/{arm_id}/resume_bundle.json"
        destination = submission_release.staged / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"arm_id": arm_id}), encoding="utf-8")
        resume[arm_id] = {"manifest": relative, "sha256": "a" * 64}
    launch["resume"] = resume
    launch_path.write_text(json.dumps(launch), encoding="utf-8")

    with pytest.raises(ValueError, match="local-only|continuation adapter"):
        submission_release.run()

    assert submission_release.api.created == submission_release.api.pushed == 0


def test_success_path_report_publication_failure_is_raised(tmp_path):
    """A report IO error on a successful path remains actionable."""
    from molgap import experiment_workflow as workflow

    failure = OSError("synthetic successful report publication fault")

    def writer(_path, _report):
        raise failure

    with pytest.raises(OSError) as caught:
        workflow._write_preparation_report(
            tmp_path / "workflow_report.json", {"status": "PREPARED_FOR_PLATFORM"}, writer=writer
        )

    assert caught.value is failure
