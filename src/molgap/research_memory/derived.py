"""Structural validation for every generated RML JSON document."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .schemas import validate_id


EXPECTED_FORMATS = {
    "trajectory_index.json": "molgap-rml-trajectory-index-v1",
    "trajectory_graph.json": "molgap-rml-trajectory-graph-v1",
    "cost_ledger.json": "molgap-rml-cost-ledger-v1",
    "role_reuse_index.json": "molgap-rml-role-reuse-index-v1",
    "reference_reuse_index.json": "molgap-rml-reference-reuse-index-v1",
    "ready_for_desktop_index.json": "molgap-rml-ready-for-desktop-index-v1",
    "screening_backtest.json": "molgap-rml-screening-backtest-v1",
    "replay_pool.json": "molgap-rml-replay-pool-v1",
    "policy_backtest.json": "molgap-rml-policy-backtest-v1",
    "completeness_report.json": "molgap-rml-completeness-report-v1",
    "research_summary.json": "molgap-rml-research-summary-v1",
}


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"derived {label} must be an object")
    return value


def _array(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"derived {label} must be an array")
    return value


def _keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"derived {label} missing fields: {missing}")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"derived {label} fields must be exactly {sorted(expected)}; "
            f"received {sorted(value)}"
        )


def _validate_cost_ledger(value: Mapping[str, Any]) -> None:
    _keys(
        value,
        {
            "event_count",
            "totals_by_hardware",
            "totals_by_category",
            "totals_by_trajectory",
            "measurement_missing_event_count",
            "not_applicable_event_count",
            "estimated_event_count",
            "cross_hardware_total",
        },
        "cost ledger",
    )
    if value["cross_hardware_total"] is not None:
        raise ValueError("derived cost ledger must not aggregate across hardware")
    for view in ("totals_by_hardware", "totals_by_category", "totals_by_trajectory"):
        for group, units in _object(value[view], f"cost {view}").items():
            if not isinstance(group, str) or not group:
                raise ValueError(f"derived cost {view} identity is invalid")
            for unit, bucket in _object(units, f"cost {view} {group}").items():
                bucket = _object(bucket, f"cost {group}:{unit}")
                expected_bucket_keys = {
                    "measured",
                    "estimated",
                    "unknown_records",
                    "not_applicable_records",
                }
                if set(bucket) != expected_bucket_keys:
                    raise ValueError(
                        f"derived cost {group}:{unit} fields must be exactly "
                        f"{sorted(expected_bucket_keys)}"
                    )
                for count_field in ("unknown_records", "not_applicable_records"):
                    count = bucket[count_field]
                    if (
                        not isinstance(count, int)
                        or isinstance(count, bool)
                        or count < 0
                    ):
                        raise ValueError(
                            f"derived cost {group}:{unit}:{count_field} is invalid"
                        )
                for basis in ("measured", "estimated"):
                    totals = _object(bucket[basis], f"cost {group}:{unit}:{basis}")
                    if set(totals) != {"known_total", "known_records"}:
                        raise ValueError(
                            f"derived cost {group}:{unit}:{basis} fields are invalid"
                        )
                    records = totals["known_records"]
                    total = totals["known_total"]
                    if (
                        not isinstance(records, int)
                        or isinstance(records, bool)
                        or records < 0
                    ):
                        raise ValueError(
                            f"derived cost {group}:{unit}:{basis} record count is invalid"
                        )
                    if records == 0 and total is not None:
                        raise ValueError(
                            f"derived cost {group}:{unit}:{basis} empty total must be null"
                        )
                    if records > 0 and (
                        not isinstance(total, (int, float))
                        or isinstance(total, bool)
                        or total < 0
                    ):
                        raise ValueError(
                            f"derived cost {group}:{unit}:{basis} known total is invalid"
                        )


def validate_derived_outputs(outputs: Mapping[str, Any]) -> None:
    if set(outputs) != set(EXPECTED_FORMATS):
        raise ValueError(
            f"derived JSON set mismatch: expected {sorted(EXPECTED_FORMATS)}, "
            f"received {sorted(outputs)}"
        )
    for name, expected_format in EXPECTED_FORMATS.items():
        value = _object(outputs[name], name)
        if value.get("format") != expected_format:
            raise ValueError(f"derived {name} has wrong format")
    pool = outputs["replay_pool.json"]
    for key in ("entries", "action_entries", "exclusions"):
        _array(pool.get(key), f"replay pool {key}")
    backtests = outputs["policy_backtest.json"]
    if backtests.get("policy_activated") is not False:
        raise ValueError("backtest must not activate a policy")
    for report in _array(backtests.get("reports"), "policy reports"):
        if report.get("policy_activated") is not False:
            raise ValueError("replay must not activate a policy")
        _keys(report, {"policy_id", "policy_version", "decisions", "exclusions", "unknown_cost_count"}, "policy report")

    trajectory_index = outputs["trajectory_index.json"]
    _exact_keys(
        trajectory_index,
        {"format", "source_digest", "trajectories"},
        "trajectory index",
    )
    for row in _array(trajectory_index["trajectories"], "trajectory index rows"):
        row = _object(row, "trajectory index row")
        _keys(
            row,
            {
                "trajectory_id",
                "track",
                "owner",
                "family_id",
                "record_mode",
                "outcome",
                "decision_ref",
                "record_path",
            },
            "trajectory index row",
        )
        validate_id(row["trajectory_id"], "derived trajectory_id")

    graph = outputs["trajectory_graph.json"]
    _exact_keys(graph, {"format", "nodes", "edges"}, "trajectory graph")
    node_ids = set()
    for node in _array(graph["nodes"], "trajectory graph nodes"):
        node = _object(node, "trajectory graph node")
        _keys(node, {"id", "type"}, "trajectory graph node")
        validate_id(node["id"], "derived graph node ID")
        if node["id"] in node_ids:
            raise ValueError(f"duplicate derived graph node ID: {node['id']}")
        node_ids.add(node["id"])
    _array(graph["edges"], "trajectory graph edges")

    cost_ledger = outputs["cost_ledger.json"]
    _exact_keys(
        cost_ledger,
        {
            "format",
            "event_count",
            "totals_by_hardware",
            "totals_by_category",
            "totals_by_trajectory",
            "infrastructure_event_ids",
            "retry_event_ids",
            "estimated_event_count",
            "measurement_missing_event_count",
            "not_applicable_event_count",
            "cross_hardware_total",
        },
        "cost ledger",
    )
    _validate_cost_ledger(cost_ledger)
    role_index = outputs["role_reuse_index.json"]
    _exact_keys(
        role_index,
        {
            "format",
            "explicit_role_event_count",
            "identities",
            "reused_role_identities",
            "repeatedly_selected_role_identities",
            "coarse_historical_role_records",
            "ambiguous_historical_role_record_count",
            "protected_roles_reported_untouched",
            "full_training_membership_conflicts",
        },
        "role reuse index",
    )
    _keys(
        outputs["role_reuse_index.json"],
        {
            "explicit_role_event_count",
            "identities",
            "coarse_historical_role_records",
            "protected_roles_reported_untouched",
        },
        "role reuse index",
    )
    reference_index = outputs["reference_reuse_index.json"]
    _exact_keys(
        reference_index,
        {
            "format",
            "references",
            "missing_reference_ids",
            "missing_reference_policy",
            "human_reference_index",
        },
        "reference reuse index",
    )
    _keys(
        reference_index,
        {"references", "missing_reference_ids", "missing_reference_policy"},
        "reference reuse index",
    )
    ready = outputs["ready_for_desktop_index.json"]
    _exact_keys(
        ready,
        {"format", "valid", "invalid", "superseded", "consumed", "blocked"},
        "READY index",
    )
    backtest = outputs["screening_backtest.json"]
    _exact_keys(
        backtest,
        {
            "format",
            "status",
            "trace_count",
            "included_comparable_groups",
            "included_trajectory_ids",
            "excluded_traces",
            "candidate_prefix_rule",
            "false_stop_count",
            "false_stop_rate",
            "false_promote_count",
            "false_promote_rate",
            "slow_starter_count",
            "native_cost_saved",
            "native_cost_added",
            "family_coverage",
            "confounding_warnings",
            "recommended_early_stop",
            "policy_activated",
        },
        "screening backtest",
    )
    _keys(
        backtest,
        {
            "status",
            "included_comparable_groups",
            "excluded_traces",
            "false_stop_count",
            "false_promote_count",
            "native_cost_saved",
            "recommended_early_stop",
            "policy_activated",
        },
        "screening backtest",
    )
    if backtest["status"] == "insufficient_evidence":
        for field in (
            "false_stop_count",
            "false_stop_rate",
            "false_promote_count",
            "false_promote_rate",
            "native_cost_saved",
            "native_cost_added",
            "recommended_early_stop",
        ):
            if backtest[field] is not None:
                raise ValueError(
                    f"insufficient screening evidence requires {field}=null"
                )
    if backtest["policy_activated"] is not False:
        raise ValueError("derived screening backtest cannot activate policy")
    completeness = outputs["completeness_report.json"]
    _exact_keys(
        completeness,
        {
            "format",
            "issues",
            "counts_by_severity",
            "broken_required_pointers",
            "ready_prerequisites_missing",
            "cost_completeness",
        },
        "completeness report",
    )
    _keys(
        completeness,
        {"issues", "counts_by_severity", "broken_required_pointers", "cost_completeness"},
        "completeness report",
    )
    cost_completeness = _object(
        completeness["cost_completeness"], "cost completeness"
    )
    _exact_keys(
        cost_completeness,
        {
            "trajectories_total",
            "with_cost_event",
            "without_cost_event",
            "with_complete_native_measurement",
            "with_incomplete_native_measurement",
        },
        "cost completeness",
    )
    for field, count in cost_completeness.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"derived cost completeness {field} is invalid")
    if (
        cost_completeness["with_cost_event"]
        + cost_completeness["without_cost_event"]
        != cost_completeness["trajectories_total"]
    ):
        raise ValueError("derived cost-event completeness counts are inconsistent")
    if (
        cost_completeness["with_complete_native_measurement"]
        + cost_completeness["with_incomplete_native_measurement"]
        != cost_completeness["with_cost_event"]
    ):
        raise ValueError("derived native-cost completeness counts are inconsistent")
    summary = outputs["research_summary.json"]
    _exact_keys(
        summary,
        {
            "format",
            "source_digest",
            "evidence",
            "trajectories",
            "transfer",
            "roles",
            "cost",
            "backtest",
            "memory_gaps",
        },
        "research summary",
    )
    _keys(
        summary,
        {"source_digest", "evidence", "trajectories", "transfer", "roles", "cost", "backtest", "memory_gaps"},
        "research summary",
    )
    summary_cost = _object(summary["cost"], "research summary cost")
    _exact_keys(
        summary_cost,
        {"totals_by_hardware", "unknown_events", "completeness"},
        "research summary cost",
    )
    if summary_cost["completeness"] != cost_completeness:
        raise ValueError("derived cost completeness summary is inconsistent")
