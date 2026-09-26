"""Metadata/synthetic-array checks; no model or remote execution."""
import ast
from pathlib import Path

import pytest

from molgap.k1_relation_audit_records import measured_cost, paired_summary


def test_allocated_cost_includes_unused_device():
    result = measured_cost({"allocated_devices": ["Tesla T4", "Tesla T4"],
        "wall_seconds": 100, "allocated_device_seconds": 200,
        "spec": {"max_allocated_device_seconds": 5400}})
    assert result["device_hours"]["value"] == 200 / 3600
    assert result["queue_hours"]["status"] == "measurement_missing"


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), 99, 6000])
def test_inconsistent_or_unqualified_cost_rejected(seconds):
    with pytest.raises(ValueError):
        measured_cost({"allocated_devices": ["Tesla T4", "Tesla T4"],
            "wall_seconds": 100, "allocated_device_seconds": seconds,
            "spec": {"max_allocated_device_seconds": 5400}})


def test_paired_direction_and_nullable_margins():
    result = paired_summary([.2, .2, .2], [.1, .1, .1])
    assert result["candidate_minus_reference_eV"] == pytest.approx(.1)
    assert result["paired_row_bootstrap_95_eV"][0] > 0
    assert result["candidate_win_fraction"] == 0
    assert result["winning_margin_mean_eV"] is None


@pytest.mark.parametrize("candidate,reference", [([], []), ([.1, .2], [.1]), ([float("nan")], [.1])])
def test_invalid_pair_rejected(candidate, reference):
    with pytest.raises(ValueError):
        paired_summary(candidate, reference)


def test_adapter_has_no_training_or_model_execution():
    module = Path(__file__).resolve().parents[1] / "src/molgap/k1_relation_audit_records.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Attribute, ast.Name))}
    assert not calls.intersection({"_model", "_infer", "make_encoder", "backward", "step", "kernels_push"})
