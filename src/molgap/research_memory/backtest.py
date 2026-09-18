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
