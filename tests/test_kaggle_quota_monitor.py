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
