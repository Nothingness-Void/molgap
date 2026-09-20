"""Strict trace-comparability reporting without policy activation."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


BASE_COMPARABILITY_FIELDS = (
    "scientific_contract",
    "dataset_identity",
    "row_split_identity",
    "optimizer_identity",
    "lr_schedule_identity",
    "target_transform_identity",
    "precision_identity",
    "ema_semantics",
    "evaluation_role_identity",
    "selection_role_identity",
    "x_axis_semantics",
    "terminal_endpoint_identity",
)


def _comparison_key(trace: dict[str, Any]) -> str:
    identity = trace["comparability_identity"]
    fields = list(BASE_COMPARABILITY_FIELDS)
    if identity["matched_architecture_required"]:
        fields.append("architecture_identity")
    key = {
        "reference_id": trace["reference_id"],
        "identity": {field: identity[field] for field in fields},
    }
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def build_screening_backtest(traces: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = []
    candidates = []
    for trace in traces:
        reasons = list(trace["backtest_eligibility"]["exclusion_reasons"])
        if not trace["backtest_eligibility"]["eligible"]:
            reasons.append("canonical_trace_marked_ineligible")
        if trace["x_axis"] == "epochs":
            reasons.append("epoch_axis_not_cross_scale_comparable")
        exposure = trace["exposure"]
        if exposure.get("optimizer_steps") is None and exposure.get("sample_presentations") is None:
            reasons.append("missing_optimizer_step_and_presentation_exposure")
        if reasons:
            excluded.append(
                {
                    "trajectory_id": trace["trajectory_id"],
                    "run_id": trace["run_id"],
                    "reasons": sorted(set(reasons)),
                }
            )
        else:
            candidates.append(trace)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in candidates:
        groups[_comparison_key(trace)].append(trace)
    included_groups = []
    for key, rows in sorted(groups.items()):
        roles = {row["comparison_role"] for row in rows}
        if roles != {"candidate", "reference"}:
            for trace in rows:
                excluded.append(
                    {
                        "trajectory_id": trace["trajectory_id"],
                        "run_id": trace["run_id"],
                        "reasons": ["no_compatible_candidate_reference_trace_pair"],
                    }
                )
            continue
        included_groups.append(
            {
                "comparability_key": key,
                "trajectory_ids": sorted({row["trajectory_id"] for row in rows}),
                "run_ids": sorted(row["run_id"] for row in rows),
            }
        )

    status = "available" if included_groups else "insufficient_evidence"
    return {
        "format": "molgap-rml-screening-backtest-v1",
        "status": status,
        "trace_count": len(traces),
        "included_comparable_groups": included_groups,
        "included_trajectory_ids": sorted(
            {trajectory_id for group in included_groups for trajectory_id in group["trajectory_ids"]}
        ),
        "excluded_traces": sorted(
            excluded, key=lambda item: (item["trajectory_id"], item["run_id"])
        ),
        "candidate_prefix_rule": None,
        "false_stop_count": None,
        "false_stop_rate": None,
        "false_promote_count": None,
        "false_promote_rate": None,
        "slow_starter_count": None,
        "native_cost_saved": None,
        "native_cost_added": None,
        "family_coverage": [],
        "confounding_warnings": (
            [] if status == "available" else ["no_strictly_comparable_trace_pair"]
        ),
        "recommended_early_stop": None,
        "policy_activated": False,
    }


def replay_policy(policy: dict[str, Any], pool: dict[str, Any]) -> dict[str, Any]:
    """Score decisions only after a pure rule sees its bounded observable view."""
    import copy
    import hashlib
    from .policy import decide, validate_policy
    from .trace import json_bytes

    validate_policy(policy)
    kind = policy["policy_type"]
    decisions, exclusions = [], []
    source = pool["entries"] if kind in {"early_stop", "screening"} else pool.get("action_entries", [])
    for entry in source:
        if entry["comparison_role"] != "candidate":
            continue
        identity = {"trajectory_id": entry["trajectory_id"], "run_id": entry["run_id"]}
        selected = {**entry["comparability_identity"], "reference_id": entry["reference_id"]}
        if any(selected.get(k) != v for k, v in policy["comparability_selector"].items()):
            exclusions.append({**identity, "reason": "policy_contract_mismatch"})
            continue
        prefix_row = None
        if kind in {"early_stop", "screening"}:
            point = policy["observation_point"]
            if entry["axis"] != point["axis"]:
                exclusions.append({**identity, "reason": "policy_axis_mismatch"})
                continue
            if any(entry["metric_semantics"].get(k) != v for k, v in policy.get("metric_semantics", {}).items()):
                exclusions.append({**identity, "reason": "policy_metric_semantics_mismatch"})
                continue
            # Exact first-observed cutoff: neither interpolation nor a later
            # checkpoint/metric at the same step can leak into this decision.
            prefix = []
            for row in entry["prefix_observations"]:
                if row[point["axis"]] > point["value"]:
                    break
                prefix.append(row)
                if row[point["axis"]] == point["value"]:
                    prefix_row = row
                    break
            if prefix_row is None:
                decision = {"action": None, "reason": "exact_observation_point_unavailable"}
            else:
                observable = {field: copy.deepcopy(prefix_row.get(field)) for field in policy["required_observable_fields"]}
                decision = decide(policy, observable)
            revealed = len(prefix)
        else:
            replay = entry.get("action_replay")
            if replay is None or replay["inputs"].get("policy_type") != kind:
                exclusions.append({**identity, "reason": "missing_frozen_action_input"})
                continue
            source_view = entry["decision_state"] if kind == "research_action" else replay["inputs"]["observables"]
            observable = {field: copy.deepcopy(source_view.get(field)) for field in policy["required_observable_fields"]}
            decision = decide(policy, observable)
            if kind == "research_action" and decision["action"] not in entry["decision_state"]["available_actions"]:
                decision = {"action": None, "reason": "action_not_available_in_frozen_state"}
            revealed = 0
        # Terminal truth is accessed only after decide has returned. Ambiguous
        # science is excluded from confusion metrics, never mapped to failure.
        label = entry.get("terminal_label") if kind in {"early_stop", "screening"} else entry["action_replay"]["truth"].get("label")
        correct_actions = None
        if kind == "research_action":
            correct_actions = entry["action_replay"]["truth"].get("correct_actions")
            if not isinstance(correct_actions, list) or not correct_actions or any(not isinstance(a, str) for a in correct_actions):
                exclusions.append({**identity, "reason": "research_action_truth_unavailable"})
                continue
            label = {"winner": False, "promotion_passed": False}
        if not isinstance(label, dict) or any(not isinstance(label.get(k), bool) for k in ("winner", "promotion_passed")):
            exclusions.append({**identity, "reason": "terminal_truth_not_applicable"})
            continue
        action = decision["action"]
        false_stop = action == "STOP" and label["winner"]
        false_promote = action in {"PROMOTE", "SCALE"} and not label["promotion_passed"]
        saved, added = _replay_cost(kind, entry, prefix_row, action)
        decisions.append({**identity, **decision, "family_id": entry["family_id"],
                          "comparability_key": entry["comparability_key"],
                          "revealed_observation_count": revealed, "terminal_label": label,
                          "action_correct": action in correct_actions if correct_actions is not None and action is not None else None,
                          "false_stop": false_stop, "false_promote": false_promote,
                          "native_device_hours_saved": saved, "native_device_hours_added": added,
                          "hardware": entry["native_cost"]["hardware"]})
    evaluated = [d for d in decisions if d["action"] not in {None, "DEFER"}]
    stops = [d for d in evaluated if d["action"] == "STOP"]
    promotions = [d for d in evaluated if d["action"] in {"PROMOTE", "SCALE"}]
    winners = [d for d in decisions if d["terminal_label"]["winner"]]
    retained = [d for d in winners if d["action"] in {"CONTINUE", "PROMOTE", "SCALE"}]
    unknown = sum(d["native_device_hours_saved"] is None or d["native_device_hours_added"] is None for d in evaluated)
    buckets = {}
    for d in evaluated:
        hardware = d["hardware"][0] if len(d["hardware"]) == 1 else "unknown"
        bucket = buckets.setdefault(hardware, {"saved": 0.0, "added": 0.0, "unknown_count": 0})
        if d["native_device_hours_saved"] is None or d["native_device_hours_added"] is None:
            bucket["unknown_count"] += 1
        else:
            bucket["saved"] += d["native_device_hours_saved"]
            bucket["added"] += d["native_device_hours_added"]
    for bucket in buckets.values():
        if bucket["unknown_count"]:
            bucket["saved"] = bucket["added"] = None
        bucket["net"] = bucket["saved"] - bucket["added"] if bucket["saved"] is not None else None
    total = next(iter(buckets.values())) if len(buckets) == 1 and not unknown else None
    false_stops = sum(d["false_stop"] for d in evaluated)
    false_promotes = sum(d["false_promote"] for d in evaluated)
    reasons = {}
    if not stops:
        reasons["false_stop_rate"] = "no_labeled_stop_decisions"
    if not promotions:
        reasons["false_promote_rate"] = "no_labeled_promotion_decisions"
    if not winners:
        reasons["winner_retention_rate"] = "no_labeled_winners"
    if total is None:
        reasons["native_device_hours"] = "missing_measurements_or_multiple_hardware_units_or_no_decisions"
    report = {
        "policy_id": policy["policy_id"], "policy_version": policy["version"],
        "policy_type": kind, "policy_status": policy["status"],
        "policy_sha256": hashlib.sha256(json_bytes(policy)).hexdigest(),
        "source_digest": pool["source_digest"], "policy_activated": False,
        "candidate_prefix_rule": policy.get("observation_point"),
        "trajectory_count": len({d["trajectory_id"] for d in decisions}),
        "comparable_group_count": len({d["comparability_key"] for d in decisions}),
        "evaluated_decisions": len(evaluated),
        "could_not_decide_count": sum(d["action"] is None for d in decisions),
        "deferred_decision_count": sum(d["action"] == "DEFER" for d in decisions),
        "false_stop_count": false_stops, "false_stop_rate": false_stops / len(stops) if stops else None,
        "false_promote_count": false_promotes,
        "false_promote_rate": false_promotes / len(promotions) if promotions else None,
        "slow_starter_count": false_stops if kind in {"early_stop", "screening"} else None,
        "winner_retained_count": len(retained),
        "winner_retention_rate": len(retained) / len(winners) if winners else None,
        "native_device_hours_saved": total["saved"] if total else None,
        "native_device_hours_added": total["added"] if total else None,
        "net_native_device_hours": total["net"] if total else None,
        "native_cost_by_hardware": dict(sorted(buckets.items())), "unknown_cost_count": unknown,
        "family_coverage": sorted({d["family_id"] for d in decisions}),
        "decisions": decisions, "exclusions": exclusions, "unavailable_reasons": reasons,
        "recommended_early_stop": None,
    }
    if kind == "research_action":
        for field in ("false_stop_count", "false_stop_rate", "false_promote_count", "false_promote_rate",
                      "slow_starter_count", "winner_retained_count", "winner_retention_rate"):
            report[field] = None
            report["unavailable_reasons"][field] = "not_a_screening_or_scale_label"
        report["action_match_count"] = sum(d["action_correct"] is True for d in evaluated)
        report["action_match_rate"] = report["action_match_count"] / len(evaluated) if evaluated else None
    elif kind == "scale_up":
        report["unavailable_reasons"]["slow_starter_count"] = "not_a_training_prefix_policy"
    return report


def _replay_cost(kind: str, entry: dict[str, Any], row: dict[str, Any] | None,
                 action: str | None) -> tuple[float | None, float | None]:
    if action in {None, "DEFER"}:
        return None, None
    if kind not in {"early_stop", "screening"}:
        # Scale-up costs are not extrapolated from a small-scale run. Only
        # explicitly measured action costs in frozen truth can quantify them.
        costs = entry["action_replay"]["truth"].get("action_costs", {}).get(action)
        if not isinstance(costs, dict) or costs.get("status") != "measured":
            return None, None
        import math
        values = costs.get("saved_device_hours"), costs.get("added_device_hours")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("invalid measured action cost")
        if entry["native_cost"]["hardware"] != [costs.get("hardware")]:
            return None, None
        return values
    total = entry["native_cost"]["device_hours"]
    if total is None:
        return None, None
    if action != "STOP":
        return 0.0, 0.0
    rows = entry["prefix_observations"]
    start = row.get("cumulative_device_time_seconds") if row else None
    end = rows[-1].get("cumulative_device_time_seconds") if rows else None
    if start is None or end is None or entry["capability"] != "complete" or end < start or end / 3600 > total + 1e-9:
        return None, None
    return (end - start) / 3600, 0.0


def build_policy_backtests(policies: list[dict[str, Any]], pool: dict[str, Any]) -> dict[str, Any]:
    return {"format": "molgap-rml-policy-backtest-v1", "source_digest": pool["source_digest"],
            "policy_activated": False,
            "reports": [replay_policy(policy, pool) for policy in policies if policy["status"] != "retired"]}
