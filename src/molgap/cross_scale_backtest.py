"""Read-only historical backtests; no policy activation or training callbacks."""
from __future__ import annotations

from math import sqrt
from pathlib import Path

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


def backtest_repository(root):
    """Read and replay the real RML owners; never write derived files or activate.

    The legacy scale heuristic is retained as a descriptive input only. Even
    its three-pair flag cannot become training authority in this projection.
    """
    from .research_memory.validate import validate_repository_records
    from .research_memory.backtest import build_screening_backtest, build_policy_backtests
    from .research_memory.replay import build_replay_pool
    from .research_memory.policy import load_policy_registry
    from .screen_backtest import ScaleObservation, backtest_low_cost_screening
    records = validate_repository_records(root)["records"]
    pool = build_replay_pool(Path(root), records)
    policies = load_policy_registry(Path(root))
    observations = []
    for group in sorted({e["comparability_key"] for e in pool["entries"]}):
        entries = [e for e in pool["entries"] if e["comparability_key"] == group]
        refs = [e for e in entries if e["comparison_role"] == "reference"]
        if len(refs) != 1:
            continue
        ref = refs[0]
        for candidate in (e for e in entries if e["comparison_role"] == "candidate"):
            metric = "ema_dev_metric" if candidate["metric_semantics"].get("ema_dev_metric") else "live_dev_metric"
            a = {r[candidate["axis"]]: r.get(metric) for r in candidate["prefix_observations"]}
            b = {r[ref["axis"]]: r.get(metric) for r in ref["prefix_observations"]}
            points = [p for p in sorted(a.keys() & b.keys()) if a[p] is not None and b[p] is not None]
            complete = candidate["capability"] == ref["capability"] == "complete"
            observations.append(ScaleObservation(candidate["trajectory_id"], "observed-first-prefix",
                "accepted-terminal", b[points[0]] - a[points[0]] if points else None,
                b[points[-1]] - a[points[-1]] if complete and points else None,
                True, complete and len(points) > 1, "Within-contract replay, not 100K-to-500K extrapolation"))
    return {"format": "molgap-cross-scale-rml-backtest-v1", "source_digest": pool["source_digest"],
        "screening": build_screening_backtest([m for _, m in records["traces"]]),
        "replay": build_policy_backtests(policies, pool),
        "legacy_scale_description": backtest_low_cost_screening(observations),
        "role_history": {"events": len(records["roles"]), "new_role_access": False},
        "early_stop_status": "NOT_RELEASED", "policy_activated": False,
        "automatic_training_action": False, "veto_500k": False, "rml_written": False}
