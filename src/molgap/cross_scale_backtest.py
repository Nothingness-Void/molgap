"""Read-only historical backtests; no policy activation or training callbacks."""
from __future__ import annotations

from math import sqrt

from .cross_scale_analysis import read_bound_json
from .cross_scale_plan import number


def historical_pair_class(left, right):
    """Historical scalars stay context; only explicit same-role identities match."""
    required = ("scientific_contract", "dataset_identity", "row_split_identity", "optimizer_identity",
                "lr_schedule_identity", "precision_identity", "ema_semantics", "target_transform_identity",
                "evaluation_role_identity", "selection_role_identity", "x_axis_semantics")
    reasons = [key for key in required if left.get(key) is None or right.get(key) is None or left[key] != right[key]]
    return {"comparison_class": "CONTEXT_ONLY" if reasons else "MATCHED_PREFIX",
            "reasons": reasons, "paired_cross_scale": False,
            "automatic_training_action": False}


def reassessment_regression(root, binding):
    """Accepted September source-only analysis is a regression, not a new result."""
    report = read_bound_json(root, binding)
    if report.get("schema") != "molgap-pcqm-scale-transfer-reassessment-v1":
        raise ValueError("Unexpected scale reassessment schema")
    joint = report["matched_500k"]["arms"]["joint"]
    snapshots = joint["snapshots"]
    steps = [row["optimizer_step"] for row in snapshots]
    if steps != sorted(set(steps)) or any(row["sample_presentations"] != row["optimizer_step"] * 128 for row in snapshots):
        raise ValueError("Historical observed-step axis corrupted")
    early, late = joint["first_10_mean_gain_eV"], joint["last_10_mean_gain_eV"]
    for value in (early, late, joint["terminal_gain_eV"]):
        number(abs(value), "historical gain")
    context = report["cross_scale_context"]
    return {"pattern": "EARLY_ADVANTAGE_CONTRACTS" if early > late else "NO_CONTRACTION",
            "first_10_gain_eV": early, "last_10_gain_eV": late,
            "terminal_gain_eV": joint["terminal_gain_eV"], "observed_snapshots": snapshots,
            "100k_vs_500k_class": "CONTEXT_ONLY" if
                context["100k_and_500k_development_cohorts_identical"] is False or
                context["100k_weight_semantics"] != context["500k_weight_semantics"] else "INSUFFICIENT_EVIDENCE",
            "projection_accepted": False, "policy_activated": False, "rml_written": False}


def _wilson(errors, total):
    if not total:
        return None
    z = 1.959963984540054
    p, den = errors / total, 1 + z*z/total
    center = (p + z*z/(2*total)) / den
    radius = z * sqrt(p*(1-p)/total + z*z/(4*total*total)) / den
    return [max(0., center-radius), min(1., center+radius)]


def backtest_decisions(cases):
    """Describe held-out, precomputed policy predictions; never fit on held-out data.

    A case supplies training group IDs used by its already-frozen policy. Fold
    leakage fails; missing terminal/native cost prevents a policy-readiness claim.
    No executable early-stop rule, ranker, fitted STQS or automatic action is returned.
    """
    seen, scored, unknown = set(), [], []
    for case in cases:
        from .cross_scale_plan import exact
        exact(case, {"case_id", "group_id", "policy_training_groups", "prediction", "terminal_positive",
                     "terminal_accepted", "comparable", "cost_saved_native", "hardware"}, "backtest case")
        if case["case_id"] in seen or case["group_id"] in case["policy_training_groups"]:
            raise ValueError("Duplicate case or leave-group-out leakage")
        seen.add(case["case_id"])
        if case["prediction"] not in {"REJECT", "PROMOTE", "ABSTAIN"}:
            raise ValueError("Unknown backtest prediction")
        if case["terminal_positive"] is not None and type(case["terminal_positive"]) is not bool:
            raise ValueError("Terminal label must be bool or unknown")
        if case["cost_saved_native"] is not None:
            number(case["cost_saved_native"], "native cost saved")
        if case["terminal_accepted"] is not True or case["comparable"] is not True or case["terminal_positive"] is None:
            unknown.append(case["case_id"])
        else:
            scored.append(case)
    positives = [c for c in scored if c["terminal_positive"]]
    negatives = [c for c in scored if not c["terminal_positive"]]
    false_reject = sum(c["prediction"] == "REJECT" for c in positives)
    false_promote = sum(c["prediction"] == "PROMOTE" for c in negatives)
    native, missing = {}, []
    for case in scored:
        value = case["cost_saved_native"]
        if value is None:
            missing.append(case["case_id"])
        else:
            native[case["hardware"]] = native.get(case["hardware"], 0.) + value
    return {"status": "INSUFFICIENT_EVIDENCE" if unknown or missing or not positives or not negatives else "DESCRIPTIVE_BACKTEST_ONLY",
            "false_reject": {"count": false_reject, "denominator": len(positives), "wilson95": _wilson(false_reject, len(positives))},
            "false_promote": {"count": false_promote, "denominator": len(negatives), "wilson95": _wilson(false_promote, len(negatives))},
            "uncertainty_scope": "case-level descriptive interval; correlated groups are not independent trials",
            "unscored_cases": unknown, "missing_native_cost_cases": missing,
            "observed_cost_saved_by_hardware": native or None,
            "policy_activated": False, "automatic_training_action": False, "cross_hardware_total": None}
