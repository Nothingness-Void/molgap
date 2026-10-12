"""Synthetic saved-curve checks; no model, remote operation or data access."""
import copy

import pytest

from molgap.research_curve_review import compare_curves, summarize_curve


def entry(role):
    semantics = {m: {"unit": "normalized" if m == "live_train_metric" else "eV",
        "target": "gap", "weights": "ema" if m.startswith("ema") else "live",
        "direction": "minimize", "role_identity": "train" if m == "live_train_metric" else "dev"}
        for m in ("live_train_metric", "live_dev_metric", "ema_dev_metric")}
    return {"trajectory_id": role, "run_id": role, "comparability_key": "world",
        "comparison_role": role, "comparability_identity": {}, "capability": "complete",
        "terminal_outcome": "INCONCLUSIVE", "terminal_label": None, "native_cost": {},
        "trace_sha256": "a" * 64, "metric_semantics": semantics,
        "prefix_observations": [{"optimizer_step": i*2, "sample_presentations": i*256,
            "epoch_or_pass": i, "learning_rate": .001,
            "live_train_metric": .1/i, "live_dev_metric": .2/i, "ema_dev_metric": .21/i}
            for i in range(1, 4)]}


def test_exact_alignment_and_positive_gain():
    a, b = entry("candidate"), entry("reference")
    for r in a["prefix_observations"]:
        r["ema_dev_metric"] -= .001
    x = compare_curves(a, b)
    assert x["aligned_points"] == 3
    assert x["paired_metric_summaries"]["ema_dev_metric"]["final_gain"] == pytest.approx(.001)
    assert x["terminal_label"] is None


def test_different_world_rejected():
    a, b = entry("candidate"), entry("reference")
    b["comparability_key"] = "other"
    with pytest.raises(ValueError, match="worlds"):
        compare_curves(a, b)


def test_presentations_mismatch_rejected():
    a, b = entry("candidate"), entry("reference")
    b["prefix_observations"][0]["sample_presentations"] += 1
    with pytest.raises(ValueError, match="Presentation"):
        compare_curves(a, b)


def test_train_units_never_mixed():
    a, b = entry("candidate"), entry("reference")
    b["metric_semantics"]["live_train_metric"]["unit"] = "eV"
    assert compare_curves(a, b)["paired_metric_summaries"]["live_train_metric"] is None


def test_no_interpolation_or_input_mutation():
    a, b = entry("candidate"), entry("reference")
    b["prefix_observations"].pop(1)
    frozen = copy.deepcopy(a)
    assert compare_curves(a, b)["aligned_points"] == 2
    assert a == frozen


def test_nonfinite_rejected():
    a = entry("candidate")
    a["prefix_observations"][0]["ema_dev_metric"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        summarize_curve(a["prefix_observations"], a["metric_semantics"])


def test_producer_epoch_coordinate_is_not_relabelled_as_full_pass():
    a = entry("candidate")
    a["prefix_observations"][-1]["epoch_or_pass"] = .6
    result = summarize_curve(a["prefix_observations"], a["metric_semantics"])
    assert result["ema_dev_metric"]["best_observed_epoch_or_pass"] == .6
    assert "best_pass" not in result["ema_dev_metric"]
