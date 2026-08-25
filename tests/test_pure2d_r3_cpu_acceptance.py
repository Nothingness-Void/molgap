from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "experiments"
    / "top20_architecture_qm9"
    / "kaggle_pair_gps_2d_r3"
    / "cpu_acceptance"
    / "run_acceptance.py"
)


def load_runner_without_torch():
    module_name = "pure2d_r3_cpu_acceptance_for_test"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("torch")
    sys.modules["torch"] = types.ModuleType("torch")
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = previous
    return module


def test_qm9_target_identity_is_measured_but_not_gated() -> None:
    runner = load_runner_without_torch()

    report = runner.identity_report(
        "edge_state_structural_gps",
        "train",
        target_max_eV=0.0027,
        prediction_max_eV=0.2,
        prediction_constrained=False,
    )

    assert report["target_policy"] == "measured_not_gated"
    assert report["target_max_eV"] == 0.0027
    assert report["prediction_constraint_checked"] is False
    assert report["prediction_tolerance_eV"] is None


def test_constrained_prediction_identity_remains_a_hard_gate() -> None:
    runner = load_runner_without_torch()

    accepted = runner.identity_report(
        "edge_state_structural_orbital",
        "validation",
        target_max_eV=0.0027,
        prediction_max_eV=4.8e-7,
        prediction_constrained=True,
    )
    assert accepted["prediction_constraint_checked"] is True
    assert accepted["prediction_tolerance_eV"] == 1e-5

    with pytest.raises(RuntimeError, match="prediction identity failed"):
        runner.identity_report(
            "edge_state_structural_orbital",
            "validation",
            target_max_eV=0.0027,
            prediction_max_eV=1.1e-5,
            prediction_constrained=True,
        )


def test_cross_candidate_payload_requires_exact_rows_and_targets() -> None:
    runner = load_runner_without_torch()

    runner.validate_cross_candidate_payload(
        "pair_gps_2d_r3_triplet",
        "validation",
        source_indices_equal=True,
        targets_equal=True,
    )
    with pytest.raises(RuntimeError, match="source-index order differs"):
        runner.validate_cross_candidate_payload(
            "pair_gps_2d_r3_triplet",
            "validation",
            source_indices_equal=False,
            targets_equal=True,
        )
    with pytest.raises(RuntimeError, match="targets differ"):
        runner.validate_cross_candidate_payload(
            "pair_gps_2d_r3_triplet",
            "validation",
            source_indices_equal=True,
            targets_equal=False,
        )
