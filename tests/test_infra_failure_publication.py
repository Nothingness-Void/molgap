"""Failure injection must not replace a worker/staging fault with receipt IO."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("publication", ["pair_state", "ledger", "worker_logs"])
def test_pair_publication_preserves_worker_spawn_fault(tmp_path, monkeypatch, publication):
    import test_experiment_workflow as fixtures
    from molgap import experiment_package, kaggle_pair_runtime as runtime
    spec = fixtures._candidate_pair_spec()
    jobs = fixtures._jobs(spec)
    source, package, inputs, output = [tmp_path / name for name in ("source", "package", "inputs", "output")]
    for path in (source, package, inputs):
        path.mkdir()
    (package / "experiment_spec.json").write_text(spec.to_json())
    launch = inputs / "experiment_launch.json"
    launch.write_text(json.dumps({"expected_package_identity": fixtures.HEX_A,
        "expected_source_archive_sha256": fixtures.HEX_B, "jobs": jobs}))
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **k: "Tesla T4\nTesla T4\n")
    monkeypatch.setattr(experiment_package, "verify_experiment_source_package", lambda _: {
        "package_identity": fixtures.HEX_A, "spec_identity": spec.identity,
        "archive_sha256": fixtures.HEX_B, "source_commit": "c" * 40})
    primary = RuntimeError("primary worker spawn fault")
    processes = []
    spawn_failed = False
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
    def spawn(command, **kwargs):
        nonlocal spawn_failed
        phase = command[command.index("--phase") + 1]
        if phase == "train" and command[command.index("--arm") + 1] == jobs[1]["arm_id"]:
            spawn_failed = True
            raise primary
        process = Process(phase)
        processes.append(process)
        return process
    monkeypatch.setattr(runtime.subprocess, "Popen", spawn)
    atomic, ledger_write = runtime.atomic_json, runtime.AllocationLedger.write
    def write_state(path, value):
        if publication == "pair_state" and path.name == "pair_state.json" and value.get("status") == "failed":
            raise OSError("secondary state publication fault")
        return atomic(path, value)
    def write_ledger(ledger, root, status, **kwargs):
        if publication == "ledger" and status == "failed":
            raise OSError("secondary ledger publication fault")
        return ledger_write(ledger, root, status, **kwargs)
    monkeypatch.setattr(runtime, "atomic_json", write_state)
    monkeypatch.setattr(runtime.AllocationLedger, "write", write_ledger)
    forward = runtime.forward_training_logs
    def relay(*args, **kwargs):
        if publication == "worker_logs" and spawn_failed:
            raise OSError("secondary worker log relay fault")
        return forward(*args, **kwargs)
    monkeypatch.setattr(runtime, "forward_training_logs", relay)
    with pytest.raises(RuntimeError) as caught:
        runtime.run_two_phase_pair(source_root=source, package_dir=package, input_root=inputs,
            launch_path=launch, output=output)
    assert caught.value is primary
    assert all(process.returncode is not None for process in processes)


def test_preparation_report_preserves_staging_failure(tmp_path, monkeypatch):
    from molgap import experiment_workflow as workflow
    primary = RuntimeError("primary staging fault")
    def stage(**kwargs):
        raise primary
    monkeypatch.setattr(workflow, "_platform_adapter", lambda _: SimpleNamespace(stage_inputs=stage))
    monkeypatch.setattr(workflow, "_workflow_inputs", lambda *a: {key: {} for key in (
        "metadata", "bindings", "jobs", "initial_states", "recipes", "validated_jobs", "source_files")})
    monkeypatch.setattr(workflow, "_workflow_checks", lambda *a: ({"family_recipes": {}, "prospective": {}}, []))
    monkeypatch.setattr(workflow, "build_experiment_source_package", lambda *a, **k: {})
    atomic = workflow._atomic
    def write_report(path, report):
        if report["status"].startswith("ERROR"):
            raise OSError("secondary preparation report fault")
        return atomic(path, report)
    monkeypatch.setattr(workflow, "_atomic", write_report)
    spec = SimpleNamespace(identity="a" * 64, to_dict=lambda: {"platform": {"name": "kaggle"}})
    with pytest.raises(RuntimeError) as caught:
        workflow.prepare_workflow(spec, tmp_path, {"kaggle": {}}, tmp_path / "prepared")
    assert caught.value is primary
