"""Scheduler-only cancellation evidence; no network, tensors or models."""
import json
from pathlib import Path

import pytest

from molgap import k1_joint_failure_records as records


def _setup(root):
    base = root / records.REL
    release = base / "attempts/v2"
    cancel = base / "cancellation_v2"
    run = "kaseichou/molgap-k1-joint-atom-s42:v2"

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    write(cancel / "platform_observation.json", {"kernel": run.split(":")[0],
        "kernel_id": 123, "version": 2, "raw_status": "CANCEL_ACKNOWLEDGED",
        "files": [], "log": "", "next_page_token": None, "observed_at": "2026-09-27T15:36:08Z"})
    write(base / "submission_receipt_v2.json", {"source_commit": "a" * 40,
        "job": {"kernel": run.split(":")[0], "kernel_id": 123, "version": 2, "run_id": run}})
    write(release / "source_config.json", {"run_id": run, "source_commit": "a" * 40})
    for name in ("protocol.md", "training_contract.json", "cancellation_v2/decision.md"):
        write(base / name, {})
    for recipe in (*records.RECIPES, "audit"):
        sub = "audit" if recipe == "audit" else f"arms/{recipe}"
        write(release / sub / "rml_plan/trajectory.json",
              {"trajectory_id": "TC-" + recipe.replace("_", "-"), "actions": [{"run_ids": [run]}]})
    return cancel


def test_cancelled_run_cannot_fabricate_worker_evidence(tmp_path, monkeypatch):
    cancel = _setup(tmp_path)
    calls = []
    monkeypatch.setattr(records, "finalize", lambda *args: calls.append(args) or {"status": "FINALIZED"})
    result = records.close_cancelled_attempt(2, repo_root=tmp_path)
    assert len(calls) == 3 and result["scientific_claim"] is None
    terminal = json.loads((cancel / "terminal_k1_corrupt_gap.json").read_text())
    assert terminal["roles"] == terminal["costs"] == []
    assert terminal["role_use"]["train"] == "unknown"
    assert terminal["evidence"]["outcome"]["scientific_status"] == "not_evaluated"
    assert "trace" not in terminal


@pytest.mark.parametrize("field,value", [("version", 3), ("raw_status", "RUNNING"),
                                        ("files", ["checkpoint.pt"]), ("log", "ep01")])
def test_nonempty_or_mismatched_cancellation_is_rejected(tmp_path, monkeypatch, field, value):
    cancel = _setup(tmp_path)
    path = cancel / "platform_observation.json"
    payload = json.loads(path.read_text()); payload[field] = value
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(records, "finalize", lambda *args: pytest.fail("must not finalize"))
    with pytest.raises(ValueError, match="cancellation evidence"):
        records.close_cancelled_attempt(2, repo_root=tmp_path)
    assert not list(cancel.glob("terminal_*.json"))
