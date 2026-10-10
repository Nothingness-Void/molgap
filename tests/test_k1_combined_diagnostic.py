"""Synthetic joint report checks; no model, role or accelerator execution."""
from copy import deepcopy

import pytest

from molgap.k1_precision_diagnostic import qualify_combined_report


def report():
    cases = {}
    for name in ("baseline", "non_blocking", "no_redundant_sync", "transfer_and_sync", "cpu_layout",
                 "fused_layout", "fused_tf32_layout"):
        variant = name.startswith("fused_")
        seconds = .6 if name == "fused_tf32_layout" else .8 if variant else 1.
        cases[name] = dict(status="numerical_mismatch" if variant else "qualified",
            equivalence=dict(exact=not variant),optimizer_contract_change=variant,
            execution_only_admissible=not variant,
            timings=[dict(seconds_per_step=seconds) for _ in range(4)])
        if variant:
            cases[name]["declared_optimizer_flags"] = dict(foreach=False,fused=True)
    return dict(status="complete",equivalence_updates=3,contract_promotion=False,cases=cases)


def test_matched_increment_arithmetic_not_additive_or_quality_admission():
    value = report()
    before = deepcopy(value)
    result = qualify_combined_report(value)
    assert result["contrasts"]["fused_increment"]["median_time_reduction"] == pytest.approx(.2)
    assert result["contrasts"]["tf32_on_fused_increment"]["median_time_reduction"] == pytest.approx(.25)
    assert result["contrasts"]["combined_total_not_single_mechanism"]["median_time_reduction"] == pytest.approx(.4)
    assert not result["strict_fp32_admission"] and not result["generalization_accuracy_qualified"]
    assert not result["full_training_saving_qualified"] and value == before


@pytest.mark.parametrize("fault", ["missing_case", "missing_window", "nan", "zero", "failed",
                                 "inexact_control", "unclassified", "admissible", "wrong_flags", "wrong_exposure", "promotion"])
def test_incomplete_or_misclassified_joint_report_rejected(fault):
    value = report()
    row = value["cases"]["fused_tf32_layout"]
    if fault == "missing_case": del value["cases"]["baseline"]
    elif fault == "missing_window": row["timings"].pop()
    elif fault == "nan": row["timings"][0]["seconds_per_step"] = float("nan")
    elif fault == "zero": row["timings"][0]["seconds_per_step"] = 0
    elif fault == "failed": row["status"] = "failed_or_unsupported"
    elif fault == "inexact_control": value["cases"]["cpu_layout"]["equivalence"]["exact"] = False
    elif fault == "unclassified": row["optimizer_contract_change"] = False
    elif fault == "admissible": row["execution_only_admissible"] = True
    elif fault == "wrong_flags": row["declared_optimizer_flags"]["foreach"] = True
    elif fault == "wrong_exposure": value["equivalence_updates"] = 2
    elif fault == "promotion": value["contract_promotion"] = True
    with pytest.raises(ValueError): qualify_combined_report(value)
