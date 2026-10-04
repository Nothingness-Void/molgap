"""Closure boundaries only; no model, inference, training or remote access."""
import json
from pathlib import Path

import pytest

from molgap.gptrans_portability_records import close_completed_audit


def test_completed_closure_confines_inputs(tmp_path):
    with pytest.raises(ValueError):
        close_completed_audit(tmp_path, tmp_path.parent / "outside", tmp_path / "inputs",
                             finalized_at="2026-10-04T00:00:00+00:00")


def test_completed_closure_rejects_different_physical_attempt(tmp_path, monkeypatch):
    import molgap.gptrans_portability as audit
    monkeypatch.setattr(audit, "analyze", lambda *args: {"accepted": True})
    base = tmp_path / "experiments/pcqm_gptrans_ema_portability/attempt_v4"
    (base / "rml_plan").mkdir(parents=True)
    for name, value in (
            ("rml_plan/trajectory.json", {"trajectory_id": "frozen"}),
            ("submission_receipt.json", {"version_number": 3,
                "kernel": "kaseichou/molgap-gptrans-ema-portability-audit"}),
            ("remote_kernel_verification.json", {})):
        (base / name).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Physical receipt"):
        close_completed_audit(tmp_path, tmp_path / "output", tmp_path / "inputs",
                             finalized_at="2026-10-04T00:00:00+00:00")
    assert not (base / "results/terminal.json").exists()


def test_completed_closure_never_bypasses_saved_artifact_acceptance(tmp_path, monkeypatch):
    import molgap.gptrans_portability as audit
    def reject(*args):
        raise ValueError("hash-bound payload rejected")
    monkeypatch.setattr(audit, "analyze", reject)
    with pytest.raises(ValueError, match="hash-bound"):
        close_completed_audit(tmp_path, tmp_path / "output", tmp_path / "inputs",
                             finalized_at="2026-10-04T00:00:00+00:00")
    assert not list(tmp_path.rglob("terminal.json"))
