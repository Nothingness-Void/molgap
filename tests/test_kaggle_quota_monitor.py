"""Unit tests for the read-only Kaggle multi-account GPU quota monitor."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "platforms" / "kaggle" / "check_quota.py"
    spec = importlib.util.spec_from_file_location("check_quota", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_quota_payload_and_table_formatting():
    mod = _load_module()
    now = datetime(2026, 10, 16, 2, 15, 0, tzinfo=timezone.utc)
    payload = {
        "quotaRefreshTime": "2026-10-17T00:00:00Z",
        "gpuQuota": {
            "timeUsed": {"seconds": 36000, "nanos": 0},
            "timeReserved": {"seconds": 1800, "nanos": 0},
            "totalTimeAllowed": {"seconds": 108000, "nanos": 0},
        },
    }
    parsed = mod.parse_quota_payload("kaggle2", "kaseichou", payload, now=now)
    assert parsed["account"] == "kaggle2"
    assert parsed["username"] == "kaseichou"
    assert parsed["total_hours"] == 30.0
    assert parsed["used_hours"] == 10.0
    assert parsed["reserved_hours"] == 0.5
    assert parsed["remaining_hours"] == 19.5
    assert parsed["remaining_hm"] == "19h 30m"
    assert parsed["refresh_time_utc"] == "2026-10-17 00:00 UTC"
    assert parsed["reset_in"] == "21h 45m"
    assert parsed["reset_in_hours"] == 21.75

    table = mod.format_table([parsed, {"account": "kaggle3", "error": "FileNotFoundError"}])
    assert "kaggle2" in table
    assert "kaseichou" in table
    assert "19.50 h (19h 30m)" in table
    assert "ERROR: FileNotFoundError" in table

    # Missing observations must not be displayed as measured zero usage.
    with pytest.raises(ValueError, match="Missing Kaggle GPU quota"):
        mod.parse_quota_payload("kaggle2", "kaseichou", {}, now=now)
    del payload["gpuQuota"]["timeUsed"]
    with pytest.raises(ValueError, match="Malformed Kaggle quota"):
        mod.parse_quota_payload("kaggle2", "kaseichou", payload, now=now)
    payload["gpuQuota"]["timeUsed"] = {"seconds": 0}
    assert mod.parse_quota_payload("kaggle2", "kaseichou", payload, now=now)["used_hours"] == 0


def test_dashboard_snapshot_and_http(tmp_path, monkeypatch):
    import json
    import threading
    import urllib.request
    directory = Path(__file__).resolve().parents[1] / "platforms/kaggle"
    monkeypatch.syspath_prepend(str(directory))
    import dashboard
    import check_quota
    import check_logs
    quota = dict(username="hidden", remaining_hours=12, total_hours=30, used_hours=17,
                 reserved_hours=1, refresh_time_utc="2026-11-26 00:00 UTC")
    monkeypatch.setattr(check_quota, "fetch_account_quota", lambda *args: quota)
    monkeypatch.setattr(check_logs, "_query_scheduler", lambda *args: ("RUNNING", "secret"))
    path = tmp_path / "status.json"
    accounts = [("Account 1", tmp_path / "unused")]
    dashboard.collect(path, accounts, ["hidden/example"])
    state = dashboard.Reader(path)
    with dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.make_handler(state)) as server:
        thread = threading.Thread(target=server.serve_forever); thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            assert b"Kaggle" in urllib.request.urlopen(url).read()
            body = urllib.request.urlopen(url + "/api/status?refresh=1").read()
            assert b"hidden" not in body and b"secret" not in body
            assert json.loads(body)["kernels"][0]["status"] == "RUNNING"
        finally:
            server.shutdown(); thread.join()
    def failed(*args):
        raise TimeoutError()
    monkeypatch.setattr(check_quota, "fetch_account_quota", failed)
    dashboard.collect(path, accounts, [])
    assert state.get()["stale"] and state.get()["accounts"][0]["remaining_hours"] == 12

    kernel = {"run": "experiment-name", "kernel": "owner/experiment", "arms": [{"completed_passes": 8}]}
    live = {"fetched_at_epoch": 1760000000, "summary": {"accounts_ok": 1}, "kernels": [kernel]}
    monkeypatch.setattr(dashboard, "fetch_live_snapshot", lambda **kwargs: live)
    dashboard.collect_full(path)
    reader = dashboard.SnapshotState(path, full=True)
    assert reader.get(force=True)["kernels"] == [kernel]

    def full_failed(**kwargs):
        raise TimeoutError()

    monkeypatch.setattr(dashboard, "fetch_live_snapshot", full_failed)
    dashboard.collect_full(path)
    assert reader.get()["error"] == "collector" and reader.get()["kernels"] == [kernel]


def _write_bound_run(runs_dir, run_id, kernel_ref, *, kernel_id=147349053, launch_jobs=None, receipt_arms=None):
    import json
    attempt = runs_dir / run_id / "v1"
    ops = attempt / "operations"
    ops.mkdir(parents=True, exist_ok=True)
    (ops / "submission_state.json").write_text(json.dumps({
        "phase": "BOUND",
        "binding": {"kernel": kernel_ref},
        "kernel_response": {"kernel": kernel_ref, "kernel_id": kernel_id, "accelerator": "NvidiaTeslaT4"},
    }), encoding="utf-8")
    if launch_jobs is not None:
        sd = attempt / "prepared" / "source_dataset"
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "experiment_launch.json").write_text(json.dumps({
            "accelerator": "NvidiaTeslaT4",
            "device_count": len(launch_jobs),
            "jobs": launch_jobs,
        }), encoding="utf-8")
    if receipt_arms is not None:
        (ops / "launch_receipt.json").write_text(json.dumps({
            "binding": {"arms": [{"arm_id": a} for a in receipt_arms]},
            "physical_runs": {"value": [{"arm_ids": receipt_arms}]},
        }), encoding="utf-8")


def test_dual_arm_metadata_and_live_snapshot(tmp_path, monkeypatch):
    import json
    directory = Path(__file__).resolve().parents[1] / "platforms/kaggle"
    monkeypatch.syspath_prepend(str(directory))
    import dashboard
    import check_quota
    import check_logs

    runs_dir = tmp_path / "runs"
    _write_bound_run(
        runs_dir, "run-launch", "acct1/k1", kernel_id=147349053,
        launch_jobs=[{"arm_id": "wd005", "device": 0}, {"arm_id": "wd010", "device": 1}],
    )
    meta_launch = dashboard._load_bound_job_meta(runs_dir, "run-launch", "acct1/k1")
    assert meta_launch["declared_arms"] == [{"arm_id": "wd005", "device": 0}, {"arm_id": "wd010", "device": 1}]
    assert meta_launch["device_count"] == 2
    assert meta_launch["accelerator"] == "NvidiaTeslaT4"
    assert meta_launch["kernel_id"] == 147349053

    _write_bound_run(
        runs_dir, "run-receipt", "acct2/k2", kernel_id=147349002,
        receipt_arms=["ring_specific", "ring_mean"],
    )
    meta_receipt = dashboard._load_bound_job_meta(runs_dir, "run-receipt", "acct2/k2")
    assert meta_receipt["declared_arms"] == [{"arm_id": "ring_specific", "device": None}, {"arm_id": "ring_mean", "device": None}]
    assert meta_receipt["device_count"] is None

    phase, arms_pre, _ = dashboard._parse_arm_curves(
        ["INPUT_RESOLVED wd005 /kaggle/input/x", "PAIR_PHASE_START train"],
        declared_arms=meta_launch["declared_arms"],
    )
    assert phase == "PAIR_PHASE_START train"
    assert [(a["arm_id"], a["device"], a["log_seen"], a["input_resolved"]) for a in arms_pre] == [
        ("wd005", 0, False, True),
        ("wd010", 1, False, False),
    ]

    log_a = json.dumps({"arm_id": "wd005", "completed_passes": 2, "planned_passes": 60, "development_mae_eV": 0.350, "best_development_mae_eV": 0.350, "train_live_mae_eV": 0.280, "status": "running"})
    log_b = json.dumps({"arm_id": "wd010", "completed_passes": 2, "planned_passes": 60, "development_mae_eV": 0.355, "best_development_mae_eV": 0.355, "train_live_mae_eV": 0.285, "status": "running"})
    _, arms_post, _ = dashboard._parse_arm_curves(
        [f"TRAIN_LOG {log_a}", f"TRAIN_LOG {log_b}"],
        declared_arms=meta_launch["declared_arms"],
    )
    assert [(a["arm_id"], a["device"], a["log_seen"], a["same_pass_delta_meV"]) for a in arms_post] == [
        ("wd005", 0, True, None),
        ("wd010", 1, True, 5.0),
    ]

    cred = tmp_path / "kaggle.json"
    cred.write_text('{"username":"acct1","key":"secret"}', encoding="utf-8")
    monkeypatch.setattr(check_logs, "ARTIFACTS_RUNS_DIR", runs_dir)
    monkeypatch.setattr(check_quota, "fetch_account_quota", lambda lbl, p: {
        "account": lbl, "username": "acct1", "remaining_hours": 20.0, "remaining_hm": "20h 00m",
        "used_hours": 10.0, "reserved_hours": 0.0, "total_hours": 30.0,
        "refresh_time_utc": "2026-11-30 00:00 UTC", "refresh_time_local": "2026-11-30 09:00 UTC+09:00",
        "reset_in": "10h 00m", "reset_in_hours": 10.0,
    })
    monkeypatch.setattr(check_logs, "_query_scheduler", lambda ref, accts: ("RUNNING", ""))
    monkeypatch.setattr(check_logs, "inspect_kernel", lambda ref, label, accts, **kw: {
        "run": label, "kernel": ref, "account": ref.split("/")[0], "scheduler": "RUNNING",
        "failure_message": "", "stream": "live_tail_timeout", "line_count": 2,
        "lines": ["INPUT_RESOLVED wd005 /x", "PAIR_PHASE_START train"],
    })
    snap = dashboard.fetch_live_snapshot(
        key_dir=tmp_path, cred_overrides=[f"kaggle1={cred}"],
        explicit_targets=["run-launch"],
    )
    assert snap["summary"]["running_kernels"] == 1
    assert snap["summary"]["running_arms"] == 2
    assert snap["kernels"][0]["arm_count"] == 2
    assert len(snap["kernels"][0]["arms"]) == 2
    assert snap["discovered"][0]["declared_arms"] == [{"arm_id": "wd005", "device": 0}, {"arm_id": "wd010", "device": 1}]
