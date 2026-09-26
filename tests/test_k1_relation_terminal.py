"""Observed cost bookkeeping, with no model construction or inference."""
import pytest
from molgap.k1_relation_terminal import allocated_arm_cost


def test_single_worker_accounts_for_idle_second_device():
    summary = {"complete": True, "total_allocated_device_seconds": 240,
        "total_job_wall_seconds": 120,
        "workers": [{"mode": "one", "complete": True, "allocated_device_seconds": 100, "worker_wall_seconds": 100}]}
    cost = allocated_arm_cost(summary, "one")
    assert cost["device_seconds"] == 240
    assert cost["wall_seconds"] == 120
    assert cost["allocated_overhead_seconds"] == 140


def test_two_worker_allocations_sum_to_observed_notebook_cost():
    summary = {"complete": True, "total_allocated_device_seconds": 240,
        "total_job_wall_seconds": 120,
        "workers": [{"mode": name, "complete": True, "allocated_device_seconds": duration,
                     "worker_wall_seconds": duration} for name, duration in (("one", 100), ("two", 110))]}
    costs = [allocated_arm_cost(summary, mode) for mode in ("one", "two")]
    assert sum(row["device_seconds"] for row in costs) == 240
    assert costs[0]["allocated_overhead_seconds"] == 30
    assert costs[1]["allocated_overhead_seconds"] == 0
    summary["complete"] = False
    with pytest.raises(ValueError):
        allocated_arm_cost(summary, "one")
