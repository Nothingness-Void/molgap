"""Canonical RML record validation without optional runtime dependencies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any


TRAJECTORY_SCHEMA = "molgap-trajectory-v1"
COST_SCHEMA = "molgap-cost-event-v1"
ROLE_SCHEMA = "molgap-role-event-v1"
TRACE_SCHEMA = "molgap-trace-manifest-v1"
READY_SCHEMA = "molgap-ready-for-desktop-v1"
COMPARISON_CLASSES = frozenset(
    {"STRICT_CAUSAL", "PAIRED_ENDPOINT", "MATCHED_PREFIX", "CONTEXT_ONLY", "NO_COMPARISON"}
)

TRAJECTORY_OUTCOMES = frozenset(
    {
        "ACTIVE",
        "NO_TRAIN",
        "POSITIVE_UNDER_CONTRACT",
        "POSITIVE_BELOW_GATE",
        "NEGATIVE_UNDER_CONTRACT",
        "INCONCLUSIVE",
        "STOP_FOR_COST",
        "DUPLICATE_EVIDENCE",
        "INFRASTRUCTURE_ONLY",
        "READY_FOR_DESKTOP",
        "CLOSED",
    }
)
COST_CATEGORIES = frozenset(
    {
        "preflight",
        "cache_build",
        "training",
        "inference",
        "acceptance",
        "audit",
        "infrastructure_failure",
        "retry",
        "other",
    }
)
ROLE_ACCESS_KINDS = frozenset(
    {
        "training_membership",
        "prediction_input",
        "labels_read",
        "metric_computed",
        "selection_used",
        "external_submission",
    }
)
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
MEASUREMENT_STATUSES = frozenset(
    {"measured", "estimated", "measurement_missing", "not_applicable"}
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return value


def _text(record: Mapping[str, Any], field: str, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{field} must be non-empty text")
    return value


def validate_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must match {ID_PATTERN.pattern!r}; received {value!r}"
        )
    return value


def _id(record: Mapping[str, Any], field: str, label: str) -> str:
    return validate_id(record.get(field), f"{label}.{field}")


def _texts(record: Mapping[str, Any], field: str, label: str) -> list[str]:
    values = _sequence(record.get(field), f"{label}.{field}")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{label}.{field} must contain non-empty text")
    return list(values)


def _measurement(value: Any, label: str) -> dict[str, Any]:
    value = _mapping(value, label)
    if set(value) != {"value", "status"}:
        raise ValueError(f"{label} must contain exactly value and status")
    status = value.get("status")
    number = value.get("value")
    if status not in MEASUREMENT_STATUSES:
        raise ValueError(f"{label}.status is invalid")
    if status in {"measured", "estimated"}:
        if (
            not isinstance(number, (int, float))
            or isinstance(number, bool)
            or number < 0
        ):
            raise ValueError(f"{label}.value must be non-negative for {status}")
    elif number is not None:
        raise ValueError(f"{label}.value must be null for {status}")
    return dict(value)


def validate_trajectory(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, "trajectory")
    if record.get("schema") != TRAJECTORY_SCHEMA:
        raise ValueError("unsupported trajectory schema")
    _id(record, "trajectory_id", "trajectory")
    for field in ("track", "owner", "family_id", "question"):
        _text(record, field, "trajectory")
    if record.get("record_mode") not in {"prospective", "retrospective_partial"}:
        raise ValueError("invalid trajectory.record_mode")
    if record.get("owner") not in {"server", "desktop", "shared", "historical"}:
        raise ValueError("invalid trajectory.owner")

    hypothesis = _mapping(record.get("hypothesis"), "trajectory.hypothesis")
    _id(hypothesis, "hypothesis_id", "trajectory.hypothesis")
    for field in (
        "supporting_evidence_ids",
        "alternative_explanations",
        "related_closed_family_ids",
    ):
        _texts(hypothesis, field, "trajectory.hypothesis")

    state = _mapping(record.get("state_at_start"), "trajectory.state_at_start")
    _text(state, "source_commit", "trajectory.state_at_start")
    for field in (
        "contract_refs",
        "reference_ids",
        "parent_trajectory_ids",
        "prior_evidence_ids",
        "role_snapshot_refs",
    ):
        _texts(state, field, "trajectory.state_at_start")
    prior_trajectory_ids = (
        _texts(state, "prior_trajectory_ids", "trajectory.state_at_start")
        if "prior_trajectory_ids" in state
        else []
    )

    action_ids: set[str] = set()
    for action in _sequence(record.get("actions"), "trajectory.actions"):
        action = _mapping(action, "trajectory.action")
        action_id = _id(action, "action_id", "trajectory.action")
        if action_id in action_ids:
            raise ValueError(f"duplicate action_id within trajectory: {action_id}")
        action_ids.add(action_id)
        _text(action, "type", "trajectory.action")
        _text(action, "source_commit", "trajectory.action")
        for field in ("run_ids", "attempt_ids", "evidence_refs", "cost_event_ids"):
            _texts(action, field, "trajectory.action")

    result = _mapping(record.get("result"), "trajectory.result")
    _texts(result, "evidence_ids", "trajectory.result")
    _texts(result, "evidence_refs", "trajectory.result")
    decision = _mapping(record.get("decision"), "trajectory.decision")
    _text(decision, "decision_ref", "trajectory.decision")
    outcome = _text(decision, "outcome", "trajectory.decision")
    if outcome not in TRAJECTORY_OUTCOMES:
        raise ValueError(f"unsupported trajectory outcome: {outcome}")
    _texts(decision, "next_allowed_actions", "trajectory.decision")
    _texts(decision, "reopen_conditions", "trajectory.decision")

    for field in ("supporting_evidence_ids",):
        for value in hypothesis[field]:
            validate_id(value, f"trajectory.hypothesis.{field}")
    for field in ("reference_ids", "parent_trajectory_ids", "prior_evidence_ids"):
        for value in state[field]:
            validate_id(value, f"trajectory.state_at_start.{field}")
    for value in prior_trajectory_ids:
        validate_id(value, "trajectory.state_at_start.prior_trajectory_ids")
    for action in record["actions"]:
        for value in action["cost_event_ids"]:
            validate_id(value, "trajectory.action.cost_event_ids")
    for value in result["evidence_ids"]:
        validate_id(value, "trajectory.result.evidence_ids")
    readiness = record.get("readiness")
    if readiness is not None:
        readiness = _mapping(readiness, "trajectory.readiness")
        for field in ("candidate_evidence_id", "compatible_reference_id"):
            _id(readiness, field, "trajectory.readiness")
        for field in (
            "qualification_100k_evidence_ids",
            "qualification_500k_evidence_ids",
            "native_cost_event_ids",
        ):
            for value in _texts(readiness, field, "trajectory.readiness"):
                validate_id(value, f"trajectory.readiness.{field}")

    comparison_fields = (
        "comparison_class",
        "comparison_readiness_ref",
        "comparison_blockers",
        "reference_bundle_id",
    )
    present_comparison_fields = [field for field in comparison_fields if field in record]
    if present_comparison_fields and len(present_comparison_fields) != len(comparison_fields):
        raise ValueError("trajectory comparison metadata must be complete when present")
    if present_comparison_fields:
        if record["comparison_class"] not in COMPARISON_CLASSES:
            raise ValueError("invalid trajectory comparison_class")
        _text(record, "comparison_readiness_ref", "trajectory")
        _texts(record, "comparison_blockers", "trajectory")
        reference_bundle_id = record["reference_bundle_id"]
        if reference_bundle_id is not None:
            validate_id(reference_bundle_id, "trajectory.reference_bundle_id")

    if record["record_mode"] == "prospective":
        prospective_text = (
            "observed_deficiency",
            "changed_mechanism",
            "cheapest_falsifier",
            "expected_native_cost_ref",
            "decision_changed_if_positive",
            "decision_changed_if_negative",
        )
        for field in prospective_text:
            _text(hypothesis, field, "prospective trajectory.hypothesis")
        if not hypothesis["supporting_evidence_ids"]:
            raise ValueError("prospective trajectory requires supporting evidence")
        if not hypothesis["alternative_explanations"]:
            raise ValueError("prospective trajectory requires an alternative explanation")
        if not hypothesis["related_closed_family_ids"]:
            raise ValueError("prospective trajectory requires related closed-route state")
        _text(state, "source_config_identity", "prospective trajectory.state_at_start")
        if state["source_commit"].startswith("unknown"):
            raise ValueError("prospective trajectory requires a known source commit")
        if not state["contract_refs"]:
            raise ValueError("prospective trajectory requires contract identity")
        if not state["prior_evidence_ids"]:
            raise ValueError("prospective trajectory requires prior evidence state")
        if not state["role_snapshot_refs"]:
            raise ValueError("prospective trajectory requires role state")
        _text(state, "budget_snapshot_ref", "prospective trajectory.state_at_start")
        if not record["actions"] and outcome != "NO_TRAIN":
            raise ValueError("prospective trajectory requires an action or NO_TRAIN")
    return dict(record)


def validate_cost_event(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, "cost event")
    if record.get("schema") != COST_SCHEMA:
        raise ValueError("unsupported cost-event schema")
    for field in ("cost_event_id", "trajectory_id", "action_id"):
        _id(record, field, "cost event")
    for field in (
        "run_id",
        "attempt_id",
        "platform",
        "hardware",
        "evidence_ref",
    ):
        _text(record, field, "cost event")
    if record.get("category") not in COST_CATEGORIES:
        raise ValueError("unsupported cost-event category")
    measurement = _mapping(record.get("measurement"), "cost event.measurement")
    if set(measurement) != {"device_hours", "cpu_hours", "wall_hours", "queue_hours"}:
        raise ValueError("cost measurement fields are incomplete or unknown")
    for field, value in measurement.items():
        _measurement(value, f"cost event.measurement.{field}")
    return dict(record)


def validate_role_event(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, "role event")
    if record.get("schema") != ROLE_SCHEMA:
        raise ValueError("unsupported role-event schema")
    for field in ("role_event_id", "trajectory_id", "action_id"):
        _id(record, field, "role event")
    for field in (
        "run_id",
        "dataset_identity",
        "row_manifest_hash",
        "role_name",
        "evidence_ref",
    ):
        _text(record, field, "role event")
    if record.get("access_kind") not in ROLE_ACCESS_KINDS:
        raise ValueError("unsupported role access_kind")
    if not isinstance(record.get("selection_used"), bool):
        raise ValueError("role event.selection_used must be boolean")
    return dict(record)


def validate_trace_manifest(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, "trace manifest")
    if record.get("schema") != TRACE_SCHEMA:
        raise ValueError("unsupported trace-manifest schema")
    _id(record, "trajectory_id", "trace manifest")
    _id(record, "reference_id", "trace manifest")
    if record.get("comparison_role") not in {"candidate", "reference"}:
        raise ValueError("trace manifest.comparison_role must be candidate or reference")
    for field in (
        "run_id",
        "contract_ref",
        "model_identity",
        "x_axis",
        "presentation_semantics_ref",
        "weight_semantics",
        "metric_semantics",
        "evaluation_role_identity",
        "selection_semantics",
        "trace_artifact_ref",
        "terminal_evidence_ref",
    ):
        _text(record, field, "trace manifest")
    eligibility = _mapping(record.get("backtest_eligibility"), "trace manifest.backtest_eligibility")
    if not isinstance(eligibility.get("eligible"), bool):
        raise ValueError("trace backtest eligibility must be boolean")
    _texts(eligibility, "exclusion_reasons", "trace manifest.backtest_eligibility")
    identity = _mapping(record.get("comparability_identity"), "trace manifest.comparability_identity")
    for field in (
        "scientific_contract",
        "dataset_identity",
        "row_split_identity",
        "architecture_identity",
        "optimizer_identity",
        "lr_schedule_identity",
        "target_transform_identity",
        "precision_identity",
        "ema_semantics",
        "evaluation_role_identity",
        "selection_role_identity",
        "x_axis_semantics",
        "terminal_endpoint_identity",
    ):
        _text(identity, field, "trace manifest.comparability_identity")
    if not isinstance(identity.get("matched_architecture_required"), bool):
        raise ValueError(
            "trace manifest.comparability_identity.matched_architecture_required "
            "must be boolean"
        )
    exposure = _mapping(record.get("exposure"), "trace manifest.exposure")
    for field in ("optimizer_steps", "sample_presentations"):
        value = exposure.get(field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError(f"trace manifest.exposure.{field} must be null or positive integer")
    if record["x_axis"] != identity["x_axis_semantics"]:
        raise ValueError("trace manifest x-axis identity is inconsistent")
    if record["x_axis"] == "optimizer_steps" and exposure["optimizer_steps"] is None:
        raise ValueError("optimizer-step trace requires optimizer_steps exposure")
    if record["x_axis"] == "presentations" and exposure["sample_presentations"] is None:
        raise ValueError("presentation trace requires sample_presentations exposure")
    if "checkpoint_identity" in record:
        _text(record, "checkpoint_identity", "trace manifest")
    if "trace_fields" in record:
        fields = _mapping(record["trace_fields"], "trace manifest.trace_fields")
        expected = {
            "optimizer_step",
            "sample_presentations",
            "epoch_or_pass",
            "learning_rate",
            "live_train_metric",
            "live_dev_metric",
            "ema_dev_metric",
            "checkpoint_identity",
        }
        if set(fields) != expected or any(not isinstance(value, bool) for value in fields.values()):
            raise ValueError(
                "trace manifest.trace_fields must explicitly mark every future trace field"
            )
    return dict(record)


def validate_ready_package(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, "READY package")
    if record.get("schema") != READY_SCHEMA:
        raise ValueError("unsupported READY package schema")
    for field in ("package_id", "trajectory_id", "candidate_evidence_id"):
        _id(record, field, "READY package")
    for field in (
        "source_commit",
        "cost_summary_ref",
        "transfer_status",
        "desktop_question",
        "candidate_contract_identity",
        "reference_contract_identity",
        "source_config_identity",
    ):
        _text(record, field, "READY package")
    for field in (
        "contract_refs",
        "reference_ids",
        "comparison_refs",
        "role_history_refs",
        "artifact_refs",
        "known_limitations",
        "qualification_100k_evidence_ids",
        "qualification_500k_evidence_ids",
        "native_cost_event_ids",
    ):
        _texts(record, field, "READY package")
    if record.get("transfer_status") != "qualified":
        raise ValueError("READY package transfer_status must be qualified")
    for field in (
        "reference_ids",
        "qualification_100k_evidence_ids",
        "qualification_500k_evidence_ids",
        "native_cost_event_ids",
    ):
        for value in record[field]:
            validate_id(value, f"READY package.{field}")
    for field in (
        "contract_refs",
        "reference_ids",
        "comparison_refs",
        "role_history_refs",
        "artifact_refs",
        "qualification_100k_evidence_ids",
        "native_cost_event_ids",
    ):
        if not record[field]:
            raise ValueError(f"READY package.{field} must not be empty")
    requires_500k = record.get("funnel_requires_500k")
    if not isinstance(requires_500k, bool):
        raise ValueError("READY package.funnel_requires_500k must be boolean")
    if requires_500k and not record["qualification_500k_evidence_ids"]:
        raise ValueError("READY package requires 500K qualification evidence")
    artifact_hashes = _mapping(record.get("artifact_hashes"), "READY package.artifact_hashes")
    if set(artifact_hashes) != set(record["artifact_refs"]):
        raise ValueError("READY package artifact hashes must exactly cover artifact refs")
    for locator, digest in artifact_hashes.items():
        if not isinstance(locator, str) or not locator.strip():
            raise ValueError("READY package artifact hash locator is invalid")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"READY package artifact SHA256 is invalid: {locator}")
    return dict(record)
