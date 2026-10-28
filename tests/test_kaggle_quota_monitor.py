"""Unit tests for the read-only Kaggle multi-account GPU quota monitor."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path


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
