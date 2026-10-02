"""Native physical allocation accounting including idle devices."""
import json

import pytest

from molgap.experiment_allocation import AllocationLedger


def test_one_arm_counts_both_allocated_devices_and_idle_peer(tmp_path):
    clock = [0.0]
    ledger = AllocationLedger(spec_identity="a" * 64, hardware=["Tesla T4", "Tesla T4"],
                              assignments={"candidate": 0}, clock=lambda: clock[0])
    clock[0] = 12.5
    result = ledger.write(tmp_path, "complete", arm_roots=[tmp_path / "candidate"])
    assert result["allocated_device_seconds"] == 25
    assert result["unassigned_device_seconds"] == 12.5
    assert result["devices"][1]["arm_id"] is None
    assert result["queue_seconds"]["value"] is None
    assert json.loads((tmp_path / "candidate/allocation_ledger.json").read_text()) == result


def test_failed_segment_preserves_uncertain_prior_observation():
    previous = {"format": "molgap-allocation-ledger-v1", "spec_identity": "a" * 64, "status": "running"}
    ledger = AllocationLedger(spec_identity="a" * 64, hardware=["Tesla T4"], assignments={"arm": 0},
                              prior_segments=[previous])
    result = ledger.snapshot("failed")
    assert result["prior_segments"] == [previous]
    assert result["prior_unobserved_intervals"] is True


def test_cost_segments_cannot_cross_spec_identity():
    with pytest.raises(ValueError, match="identity"):
        AllocationLedger(spec_identity="a" * 64, hardware=["Tesla T4"], assignments={"arm": 0},
            prior_segments=[{"format": "molgap-allocation-ledger-v1", "spec_identity": "b" * 64}])


def test_repeated_recovery_flattens_and_deduplicates_prior_invocations():
    clock = [0.0]
    first = AllocationLedger(spec_identity="a" * 64, hardware=["Tesla T4"],
                             assignments={"arm": 0}, clock=lambda: clock[0])
    clock[0] = 1.0
    previous = first.snapshot("running")
    second = AllocationLedger(spec_identity="a" * 64, hardware=["Tesla T4"],
        assignments={"arm": 0}, prior_segments=[previous], clock=lambda: clock[0])
    clock[0] = 2.0
    recent = second.snapshot("complete")
    third = AllocationLedger(spec_identity="a" * 64, hardware=["Tesla T4"],
        assignments={"arm": 0}, prior_segments=[recent, previous], clock=lambda: clock[0])
    result = third.snapshot("running")
    assert len(result["prior_segments"]) == 2
    assert all(segment["prior_segments"] == [] for segment in result["prior_segments"])
    assert result["prior_unobserved_intervals"] is True
