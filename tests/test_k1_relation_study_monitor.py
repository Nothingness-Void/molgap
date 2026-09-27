"""No network/model: idempotent Luna polling with a fake Kaggle client."""
import json
from pathlib import Path
import sys
import types

import pytest

from molgap.k1_relation_study_monitor import tick
from molgap.server_control import BoundRun, LocalServerControlStore


def test_two_runs_stay_silent_then_emit_one_terminal_event(tmp_path, monkeypatch):
    statuses = {"dual": "RUNNING", "rrwp": "QUEUED"}
    calls = []

    class FakeAPI:
        def authenticate(self):
            return None

        def kernels_status(self, kernel):
            calls.append(kernel)
            slot = kernel.split("/")[1]
            return types.SimpleNamespace(status="KernelWorkerStatus." + statuses[slot])

    fake = types.ModuleType("kaggle.api.kaggle_api_extended")
    fake.KaggleApi = FakeAPI
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", fake)
    monkeypatch.setattr("molgap.k1_relation_study_monitor._latest_identity",
        lambda api, kernel: {"kernel": kernel, "kernel_id": 123, "version": 1})
    monkeypatch.setenv("KAGGLE_USERNAME", "test-owner")
    monkeypatch.setenv("KAGGLE_KEY", "test-placeholder")
    credential = tmp_path / "account.json"
    credential.write_text(json.dumps({"username": "kaseichou", "key": "not-a-real-credential"}))
    jobs = []
    for slot in statuses:
        path = tmp_path / f"{slot}.json"
        run = BoundRun(campaign_id="study", chain_id=slot, run_id=f"kaseichou/{slot}:v1",
            attempt_id="v1", a_thread_id="A", b_thread_id="B", monitor_generation=1,
            remote_platform="kaggle2", remote_job_identity={"kernel": slot}, release_identity="frozen")
        LocalServerControlStore(path).bind_run(run)
        jobs.append({"slot": slot, "kernel": f"kaseichou/{slot}", "kernel_id": 123, "version": 1, "control_state": str(path)})
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps({"owner": "server", "credential_file": str(credential), "jobs": jobs}))
    assert not tick(binding)["events"]
    statuses["dual"] = "COMPLETE"
    first = tick(binding)["events"]
    assert len(first) == 1 and first[0]["event_type"] == "COMPLETE"
    assert tick(binding)["events"] == first
    LocalServerControlStore(Path(jobs[0]["control_state"])).deliver_to_a(first[0]["event_id"])
    assert not tick(binding)["events"]
    assert calls[-1] == "kaseichou/rrwp/1"

    # An unrelated newer version must never inherit this job's terminal status.
    monkeypatch.setattr("molgap.k1_relation_study_monitor._latest_identity",
        lambda api, kernel: {"kernel": kernel, "kernel_id": 123, "version": 2})
    before = len(calls)
    observed = tick(binding)
    assert not observed["events"]
    assert observed["observations"][0]["status"] == "UNKNOWN"
    assert len(calls) == before


def test_closed_binding_does_not_read_credentials(tmp_path):
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps({"owner": "server", "closed": True}))
    assert tick(binding) == {"events": [], "closed": True}


@pytest.mark.parametrize("raw, expected", [("CANCEL_ACKNOWLEDGED", "CANCELLED"),
                                          ("CANCELED", "CANCELLED"),
                                          ("CANCEL_REQUESTED", "UNKNOWN")])
def test_cancellation_is_terminal_only_after_acknowledgment(tmp_path, monkeypatch, raw, expected):
    class FakeAPI:
        def authenticate(self):
            pass

        def kernels_status(self, kernel):
            return types.SimpleNamespace(status=raw)

    fake = types.ModuleType("kaggle.api.kaggle_api_extended")
    fake.KaggleApi = FakeAPI
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", fake)
    monkeypatch.setattr("molgap.k1_relation_study_monitor._latest_identity",
        lambda api, kernel: {"kernel": kernel, "kernel_id": 123, "version": 2})
    monkeypatch.setenv("KAGGLE_USERNAME", "test-owner")
    monkeypatch.setenv("KAGGLE_KEY", "test-placeholder")
    credential = tmp_path / "account.json"
    credential.write_text(json.dumps({"username": "kaseichou", "key": "not-a-real-credential"}))
    control = tmp_path / "control.json"
    LocalServerControlStore(control).bind_run(BoundRun(
        campaign_id="study", chain_id="dual", run_id="kaseichou/dual:v2",
        attempt_id="v2", a_thread_id="A", b_thread_id="B", monitor_generation=2,
        remote_platform="kaggle2", remote_job_identity={"kernel": "dual"}, release_identity="frozen"))
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps({"owner": "server", "credential_file": str(credential),
        "jobs": [{"slot": "dual", "kernel": "kaseichou/dual", "kernel_id": 123,
                  "version": 2, "control_state": str(control)}]}))
    result = tick(binding)
    assert result["observations"][0]["status"] == expected
    assert LocalServerControlStore(control).load()["last_observation"]["detail"]["api_status"] == raw
    if expected == "CANCELLED":
        assert len(result["events"]) == 1
        assert tick(binding)["events"] == result["events"]
    else:
        assert not result["events"]
