"""Fail-closed V5 comparison readiness and reusable-reference semantics.

The helpers in this module are machine-neutral.  They classify existing or
planned evidence; they never launch training, read a protected role, or infer a
missing historical fact.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


COMPARISON_READINESS_FORMAT = "molgap-comparison-readiness-v1"
COMPARISON_PRELAUNCH_FORMAT = "molgap-comparison-prelaunch-v1"
REFERENCE_BUNDLE_FORMAT = "molgap-reference-bundle-v1"
TARGET_TRANSFORM_FORMAT = "molgap-target-transform-asset-v1"
PRELAUNCH_STRICT_STATUS = "PRELAUNCH_STRICT_PLANNED"

COMPARISON_CLASSES = frozenset(
    {
        "STRICT_CAUSAL",
        "PAIRED_ENDPOINT",
        "MATCHED_PREFIX",
        "CONTEXT_ONLY",
        "NO_COMPARISON",
    }
)
HISTORICAL_COMPARISON_CLASSES = frozenset(
    {
        "STRICT_CAUSAL_UNDER_ORIGINAL_CONTRACT",
        "PAIRED_ENDPOINT_UNDER_ORIGINAL_CONTRACT",
        "MATCHED_PREFIX_UNDER_ORIGINAL_CONTRACT",
        "CONTEXT_ONLY_UNDER_ORIGINAL_EVIDENCE",
        "NO_COMPARISON",
    }
)
PROSPECTIVE_REUSE_STATUSES = frozenset(
    {"READY", "INCOMPLETE_RECOVERABLE", "NOT_COMPATIBLE", "UNAVAILABLE"}
)

# Every field is explicit so a fingerprint cannot conceal a scientific
# difference.  ``architecture_config_identity`` may differ only when it is a
# predeclared intervention; all other differences remain confounders.
STRICT_IDENTITY_FIELDS = (
    "benchmark_identity",
    "dataset_identity",
    "data_role_identity",
    "row_membership_identity",
    "row_order_identity",
    "feature_identity",
    "target_identity",
    "seed",
    "precision",
    "tf32_enabled",
    "deterministic_algorithms",
    "physical_batch_per_device",
    "gradient_accumulation_steps",
    "tail_batch_policy",
    "optimizer_identity",
    "optimizer_mode",
    "optimizer_fused",
    "schedule_identity",
    "loss_identity",
    "target_transform_identity",
    "target_transform_asset_sha256",
    "sample_presentations",
    "optimizer_steps",
    "checkpoint_selection_identity",
    "evaluation_role_identity",
    "selection_role_identity",
    "architecture_config_identity",
    "runtime_certificate_scope",
    "weight_semantics",
    "ema_enabled",
    "ema_decay",
    "ema_update_frequency",
    "evaluation_weight_source",
)

REQUIRED_REFERENCE_ARTIFACTS = frozenset(
    {
        "checkpoint",
        "runtime_certificate",
        "prediction_manifest",
        "row_manifest",
        "target_manifest",
        "trace_manifest",
        "role_history",
        "target_transform_asset",
        "acceptance",
        "decision",
    }
)
REQUIRED_CANDIDATE_ONLY_ARTIFACTS = frozenset({"paired_analysis"})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPLETE = frozenset({"complete", "accepted", "available"})

ROLE_EVENT_KINDS = frozenset(
    {
        "training_membership",
        "prediction_input",
        "labels_read",
        "metric_computed",
        "selection_used",
        "external_submission",
    }
)
ROLE_APPLICABILITY_STATES = frozenset({"applicable", "not_applicable"})
TRACE_FIELD_DECLARATIONS = frozenset(
    {
        "optimizer_step",
        "sample_presentations",
        "epoch_or_pass",
        "learning_rate",
        "live_train_metric",
        "live_dev_metric",
        "ema_dev_metric",
        "checkpoint_identity",
    }
)

INTERVENTION_FIELDS_BY_PURPOSE = {
    "architecture_comparison": frozenset({"architecture_config_identity"}),
    "mechanism_comparison": frozenset({"architecture_config_identity"}),
    "optimizer_comparison": frozenset(
        {"optimizer_identity", "optimizer_mode", "optimizer_fused"}
    ),
    "ema_comparison": frozenset(
        {
            "ema_enabled",
            "ema_decay",
            "ema_update_frequency",
            "weight_semantics",
            "evaluation_weight_source",
        }
    ),
    "schedule_comparison": frozenset({"schedule_identity"}),
}
NONCAUSAL_PURPOSES = frozenset(
    {
        "diagnostic",
        "transfer_study",
        "delivery_experiment",
        "contextual_experiment",
        "NO_TRAIN",
    }
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _texts(value: Any, label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must contain non-empty text")
    return list(value)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_intervention_scope(
    *,
    experiment_purpose: str,
    intervention_group_id: str,
    mechanism_id: str | None,
    declared_intervention_fields: Sequence[str],
    require_nonempty: bool,
) -> list[str]:
    purpose = _text(experiment_purpose, "experiment purpose")
    _text(intervention_group_id, "intervention group ID")
    declared = _texts(declared_intervention_fields, "declared intervention fields")
    if len(declared) != len(set(declared)):
        raise ValueError("declared intervention fields must be unique")
    allowed = INTERVENTION_FIELDS_BY_PURPOSE.get(purpose)
    if allowed is None and purpose in NONCAUSAL_PURPOSES:
        allowed = frozenset()
    if allowed is None:
        raise ValueError(f"unsupported causal experiment purpose: {purpose}")
    disallowed = sorted(set(declared) - allowed)
    if disallowed:
        raise ValueError(
            f"{purpose} cannot declare intervention fields outside one logical group: "
            f"{disallowed}"
        )
    if require_nonempty and purpose in INTERVENTION_FIELDS_BY_PURPOSE and not declared:
        raise ValueError(f"{purpose} requires one declared intervention group")
    if purpose == "mechanism_comparison":
        _text(mechanism_id, "mechanism ID")
    elif mechanism_id is not None:
        _text(mechanism_id, "mechanism ID")
    return declared


def validate_role_applicability_plan(value: Any) -> dict[str, str]:
    plan = _mapping(value, "role applicability plan")
    if set(plan) != set(ROLE_EVENT_KINDS):
        raise ValueError("role applicability plan must explicitly declare every role kind")
    invalid = {
        kind: state
        for kind, state in plan.items()
        if state not in ROLE_APPLICABILITY_STATES
    }
    if invalid:
        raise ValueError(f"invalid role applicability states: {invalid}")
    return {kind: str(plan[kind]) for kind in sorted(plan)}


def validate_observed_role_events(
    role_applicability_plan: Mapping[str, Any], observed_event_kinds: Sequence[str]
) -> dict[str, Any]:
    """Require events only for applicable roles and forbid fabricated ones."""

    plan = validate_role_applicability_plan(role_applicability_plan)
    observed = _texts(observed_event_kinds, "observed role event kinds")
    if len(observed) != len(set(observed)):
        raise ValueError("observed role event kinds must be unique")
    unknown = sorted(set(observed) - set(ROLE_EVENT_KINDS))
    if unknown:
        raise ValueError(f"unknown observed role event kinds: {unknown}")
    unexpected = sorted(kind for kind in observed if plan[kind] == "not_applicable")
    if unexpected:
        raise ValueError(f"not-applicable role kinds cannot have events: {unexpected}")
    missing = sorted(
        kind
        for kind, applicability in plan.items()
        if applicability == "applicable" and kind not in observed
    )
    return {
        "status": "complete" if not missing else "incomplete",
        "missing_applicable_event_kinds": missing,
        "observed_event_kinds": sorted(observed),
    }


def validate_trace_plan(value: Any) -> dict[str, bool]:
    plan = _mapping(value, "trace plan")
    if set(plan) != set(TRACE_FIELD_DECLARATIONS):
        raise ValueError("trace plan must explicitly declare every trace field")
    if any(not isinstance(item, bool) for item in plan.values()):
        raise ValueError("trace plan declarations must be boolean")
    return {field: bool(plan[field]) for field in sorted(plan)}


def validate_runtime_qualification_plan(value: Any) -> dict[str, Any]:
    plan = _mapping(value, "runtime qualification plan")
    if plan.get("status") != "declared":
        raise ValueError("runtime qualification plan status must be declared")
    if plan.get("runtime_certificate_required") is not True:
        raise ValueError("runtime qualification plan must require a runtime certificate")
    _text(plan.get("qualification_scope"), "runtime qualification scope")
    return dict(plan)


def validate_historical_comparison_metadata(value: Any) -> dict[str, Any]:
    """Keep original-contract validity independent from future V5 reuse."""

    record = _mapping(value, "historical comparison metadata")
    if record.get("historical_comparison_class") not in HISTORICAL_COMPARISON_CLASSES:
        raise ValueError("invalid historical comparison class")
    if record.get("prospective_reuse_status") not in PROSPECTIVE_REUSE_STATUSES:
        raise ValueError("invalid prospective reuse status")
    if not isinstance(record.get("requires_new_prospective_experiment"), bool):
        raise ValueError("requires_new_prospective_experiment must be boolean")
    return dict(record)


def target_transform_asset_digest(record: Mapping[str, Any]) -> str:
    """Hash the canonical transform payload without its self-identifying digest."""

    payload = {key: value for key, value in record.items() if key != "asset_sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_target_transform_asset(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a portable target transform rather than a local reduction."""

    record = _mapping(record, "target transform asset")
    if record.get("format") != TARGET_TRANSFORM_FORMAT:
        raise ValueError("unsupported target-transform asset format")
    for field in ("asset_id", "target_identity", "variance_convention"):
        _text(record.get(field), f"target transform asset.{field}")
    for field in ("mean", "std"):
        value = record.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"target transform asset.{field} must be numeric")
    if record["std"] <= 0:
        raise ValueError("target transform asset.std must be positive")
    ddof = record.get("ddof")
    if not isinstance(ddof, int) or isinstance(ddof, bool) or ddof < 0:
        raise ValueError("target transform asset.ddof must be a non-negative integer")
    for field in ("source_row_manifest_sha256", "target_sha256", "asset_sha256"):
        _sha(record.get(field), f"target transform asset.{field}")
    expected = target_transform_asset_digest(record)
    if record["asset_sha256"] != expected:
        raise ValueError("target transform asset.asset_sha256 does not match canonical payload")
    return dict(record)


def validate_reference_bundle(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one complete, reusable comparator bundle.

    A scalar metric is deliberately insufficient: aligned prediction, row, and
    target identities are mandatory.
    """

    record = _mapping(record, "reference bundle")
    if record.get("format") != REFERENCE_BUNDLE_FORMAT:
        raise ValueError("unsupported reference-bundle format")
    for field in (
        "reference_bundle_id",
        "reference_id",
        "contract_ref",
        "architecture_config_identity",
        "source_commit_or_archive",
        "checkpoint_identity",
        "runtime_certificate_ref",
        "row_manifest_ref",
        "target_manifest_ref",
        "trace_manifest_ref",
        "role_history_ref",
        "target_transform_asset_ref",
        "cost_records_ref",
        "acceptance_ref",
        "decision_ref",
    ):
        _text(record.get(field), f"reference bundle.{field}")

    prediction = _mapping(record.get("prediction_manifest"), "reference bundle.prediction_manifest")
    for field in ("prediction_sha256", "source_idx_sha256", "target_sha256"):
        _sha(prediction.get(field), f"reference bundle.prediction_manifest.{field}")
    for field in ("ordering_semantics", "evaluation_role_identity"):
        _text(prediction.get(field), f"reference bundle.prediction_manifest.{field}")
    for field in ("row_count", "unique_source_idx"):
        value = prediction.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"reference bundle.prediction_manifest.{field} must be positive")
    if prediction["unique_source_idx"] != prediction["row_count"]:
        raise ValueError("reference prediction source_idx values are not unique")

    identity = _mapping(record.get("comparison_identity"), "reference bundle.comparison_identity")
    missing = [field for field in STRICT_IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(f"reference comparison identity is incomplete: {missing}")
    validate_stochasticity(record.get("stochasticity"))
    return dict(record)


def validate_stochasticity(value: Any) -> dict[str, Any]:
    """Keep row uncertainty and repeat-run stochasticity separate."""

    record = _mapping(value, "stochasticity")
    row = _mapping(record.get("row_bootstrap_uncertainty"), "row bootstrap uncertainty")
    training = _mapping(record.get("training_stochasticity"), "training stochasticity")
    if row.get("status") not in {"measured", "unavailable", "not_requested"}:
        raise ValueError("invalid row-bootstrap status")
    if training.get("status") not in {"measured", "unavailable", "historical_estimate"}:
        raise ValueError("invalid training-stochasticity status")
    method = str(training.get("estimation_method", "")).lower()
    if "bootstrap" in method or training.get("source") == "row_bootstrap_uncertainty":
        raise ValueError("row bootstrap cannot establish training stochasticity")
    repeats = training.get("same_contract_repeat_ids")
    if not isinstance(repeats, list) or any(not isinstance(x, str) for x in repeats):
        raise ValueError("training stochasticity repeat IDs must be an array")
    n_repeats = training.get("n_repeats")
    if not isinstance(n_repeats, int) or isinstance(n_repeats, bool) or n_repeats < 0:
        raise ValueError("training stochasticity n_repeats must be non-negative")
    if n_repeats != len(repeats):
        raise ValueError("training stochasticity repeat count is inconsistent")
    floor = training.get("stochasticity_floor_eV")
    if training["status"] == "measured":
        if n_repeats < 2 or not isinstance(floor, (int, float)) or floor < 0:
            raise ValueError("measured training stochasticity requires repeats and a floor")
    elif floor is not None and training["status"] == "unavailable":
        raise ValueError("unavailable training stochasticity cannot have a numeric floor")
    return {"row_bootstrap_uncertainty": dict(row), "training_stochasticity": dict(training)}


def _artifact_state(side: Mapping[str, Any]) -> tuple[set[str], list[str]]:
    artifacts = side.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return set(), ["artifacts"]
    complete = {name for name, status in artifacts.items() if str(status).lower() in _COMPLETE}
    missing = sorted(REQUIRED_REFERENCE_ARTIFACTS - complete)
    return complete, missing


def _observed_role_status(side: Mapping[str, Any]) -> str:
    plan = side.get("role_applicability_plan")
    observed = side.get("observed_role_event_kinds")
    if plan is None and observed is None:
        return str(side.get("role_status", "unknown"))
    if plan is None or observed is None:
        return "incomplete"
    return validate_observed_role_events(plan, observed)["status"]


def assess_comparison_readiness(
    *,
    candidate_id: str,
    candidate: Mapping[str, Any],
    reference_id: str | None,
    reference: Mapping[str, Any] | None,
    declared_intervention_fields: Sequence[str] = (),
    experiment_purpose: str = "architecture_comparison",
    intervention_group_id: str = "architecture-config",
    mechanism_id: str | None = None,
    prefix_aligned: bool = False,
    scalar_context_available: bool = False,
) -> dict[str, Any]:
    """Classify observed post-run evidence without inferring absent facts."""

    _text(candidate_id, "candidate_id")
    candidate = _mapping(candidate, "candidate")
    declared = _validate_intervention_scope(
        experiment_purpose=experiment_purpose,
        intervention_group_id=intervention_group_id,
        mechanism_id=mechanism_id,
        declared_intervention_fields=declared_intervention_fields,
        require_nonempty=experiment_purpose in INTERVENTION_FIELDS_BY_PURPOSE,
    )
    unknown_declared = sorted(set(declared) - set(STRICT_IDENTITY_FIELDS))
    if unknown_declared:
        raise ValueError(f"unknown declared intervention fields: {unknown_declared}")

    base = {
        "format": COMPARISON_READINESS_FORMAT,
        "candidate_id": candidate_id,
        "reference_id": reference_id,
        "strict_ready": False,
        "strict_status": "PENDING",
        "experiment_purpose": experiment_purpose,
        "intervention_group_id": intervention_group_id,
        "mechanism_id": mechanism_id,
        "declared_intervention_fields": declared,
        "matched_fields": {},
        "mismatched_fields": {},
        "missing_fields": [],
        "missing_artifacts": [],
        "reference_prediction_status": "unavailable",
        "row_alignment_status": str(candidate.get("row_alignment_status", "unknown")),
        "runtime_certificate_status": str(candidate.get("runtime_certificate_status", "unknown")),
        "role_status": str(candidate.get("role_status", "unknown")),
        "trace_status": str(candidate.get("trace_status", "unknown")),
        "stochasticity_status": str(candidate.get("stochasticity_status", "unknown")),
        "blocker_codes": [],
        "decision_scope": [],
        "forbidden_claims": [],
        "recoverable_without_training": [],
        "rerun_required_blockers": [],
    }
    if reference is None or reference_id is None:
        base.update(
            {
                "comparison_class": "NO_COMPARISON",
                "blocker_codes": ["REFERENCE_UNAVAILABLE"],
                "decision_scope": ["candidate-only description"],
                "forbidden_claims": ["comparative claim", "causal architecture claim"],
                "recoverable_without_training": ["locate or assemble an existing reference bundle"],
            }
        )
        return base

    reference = _mapping(reference, "reference")
    candidate_identity = candidate.get("comparison_identity", {})
    reference_identity = reference.get("comparison_identity", {})
    if not isinstance(candidate_identity, Mapping):
        candidate_identity = {}
    if not isinstance(reference_identity, Mapping):
        reference_identity = {}
    missing_fields: list[str] = []
    matched: dict[str, Any] = {}
    mismatched: dict[str, Any] = {}
    for field in STRICT_IDENTITY_FIELDS:
        candidate_has = field in candidate_identity
        reference_has = field in reference_identity
        if not candidate_has:
            missing_fields.append(f"candidate.{field}")
        if not reference_has:
            missing_fields.append(f"reference.{field}")
        if candidate_has and reference_has:
            if candidate_identity[field] == reference_identity[field]:
                matched[field] = candidate_identity[field]
            else:
                mismatched[field] = {
                    "candidate": candidate_identity[field],
                    "reference": reference_identity[field],
                }

    _, candidate_missing_artifacts = _artifact_state(candidate)
    _, reference_missing_artifacts = _artifact_state(reference)
    candidate_artifacts = candidate.get("artifacts", {})
    if not isinstance(candidate_artifacts, Mapping):
        candidate_artifacts = {}
    candidate_missing_artifacts.extend(
        sorted(
            artifact
            for artifact in REQUIRED_CANDIDATE_ONLY_ARTIFACTS
            if str(candidate_artifacts.get(artifact, "")).lower() not in _COMPLETE
        )
    )
    missing_artifacts = [f"candidate.{x}" for x in candidate_missing_artifacts]
    missing_artifacts.extend(f"reference.{x}" for x in reference_missing_artifacts)
    prediction_status = str(reference.get("prediction_status", "unavailable"))
    candidate_prediction_status = str(candidate.get("prediction_status", "unavailable"))
    row_alignment = str(candidate.get("row_alignment_status", "unknown"))
    terminal_complete = bool(candidate.get("terminal_complete")) and bool(
        reference.get("terminal_complete")
    )
    runtime_status = (
        "accepted"
        if candidate.get("runtime_certificate_status") == "accepted"
        and reference.get("runtime_certificate_status") == "accepted"
        else "incomplete"
    )
    role_status = (
        "complete"
        if _observed_role_status(candidate) == "complete"
        and _observed_role_status(reference) == "complete"
        else "incomplete"
    )
    trace_status = (
        "complete"
        if candidate.get("trace_status") == "complete" and reference.get("trace_status") == "complete"
        else "incomplete"
    )
    blockers: list[str] = []
    if missing_fields:
        blockers.append("MISSING_IDENTITY_FIELDS")
    non_intervention_mismatch = sorted(set(mismatched) - set(declared))
    if non_intervention_mismatch:
        blockers.append("SCIENTIFIC_CONTRACT_MISMATCH")
    if set(declared) - set(mismatched):
        blockers.append("DECLARED_INTERVENTION_NOT_OBSERVED")
    if missing_artifacts:
        blockers.append("MISSING_REQUIRED_ARTIFACTS")
    if prediction_status not in _COMPLETE or candidate_prediction_status not in _COMPLETE:
        blockers.append("MISSING_ALIGNED_PREDICTIONS")
    if row_alignment != "aligned":
        blockers.append("ROW_ALIGNMENT_NOT_CONFIRMED")
    if runtime_status != "accepted":
        blockers.append("RUNTIME_CERTIFICATE_INCOMPLETE")
    if role_status != "complete":
        blockers.append("ROLE_HISTORY_INCOMPLETE")
    if trace_status != "complete":
        blockers.append("TRACE_INCOMPLETE")
    if not terminal_complete:
        blockers.append("TERMINAL_ENDPOINT_UNAVAILABLE")

    strict_ready = not blockers
    if strict_ready:
        comparison_class = "STRICT_CAUSAL"
        strict_status = "READY"
        scope = ["causal claim limited to declared intervention fields", "paired endpoint analysis"]
        forbidden = ["claim about any undeclared mechanism"]
    elif not terminal_complete and prefix_aligned and not missing_fields and not non_intervention_mismatch:
        comparison_class = "MATCHED_PREFIX"
        strict_status = "PENDING_TERMINAL"
        scope = ["futility decision", "STOP_FOR_COST", "matched-prefix trajectory comparison"]
        forbidden = ["terminal architecture superiority", "terminal paired endpoint claim"]
    elif (
        terminal_complete
        and row_alignment == "aligned"
        and prediction_status in _COMPLETE
        and candidate_prediction_status in _COMPLETE
    ):
        comparison_class = "PAIRED_ENDPOINT"
        strict_status = "NOT_STRICT"
        scope = ["paired endpoint prediction comparison", "row bootstrap", "ensemble analysis"]
        forbidden = ["causal attribution to architecture or mechanism"]
    else:
        comparison_class = "CONTEXT_ONLY"
        strict_status = "PENDING" if reference is not None else "UNAVAILABLE"
        scope = ["historical context"] if scalar_context_available else ["evidence-gap accounting"]
        forbidden = ["strict causal claim", "paired statistical claim"]

    recoverable = []
    rerun = []
    recoverable_codes = {
        "MISSING_IDENTITY_FIELDS",
        "MISSING_REQUIRED_ARTIFACTS",
        "MISSING_ALIGNED_PREDICTIONS",
        "RUNTIME_CERTIFICATE_INCOMPLETE",
        "ROLE_HISTORY_INCOMPLETE",
        "TRACE_INCOMPLETE",
    }
    for blocker in blockers:
        if blocker in recoverable_codes:
            recoverable.append(blocker)
        elif blocker in {
            "SCIENTIFIC_CONTRACT_MISMATCH",
            "DECLARED_INTERVENTION_NOT_OBSERVED",
            "TERMINAL_ENDPOINT_UNAVAILABLE",
        }:
            rerun.append(blocker)
    if "ROW_ALIGNMENT_NOT_CONFIRMED" in blockers:
        recoverable.append("ROW_ALIGNMENT_NOT_CONFIRMED")

    base.update(
        {
            "comparison_class": comparison_class,
            "strict_ready": strict_ready,
            "strict_status": strict_status,
            "matched_fields": matched,
            "mismatched_fields": mismatched,
            "missing_fields": sorted(missing_fields),
            "missing_artifacts": sorted(missing_artifacts),
            "reference_prediction_status": prediction_status,
            "row_alignment_status": row_alignment,
            "runtime_certificate_status": runtime_status,
            "role_status": role_status,
            "trace_status": trace_status,
            "blocker_codes": sorted(set(blockers)),
            "decision_scope": scope,
            "forbidden_claims": forbidden,
            "recoverable_without_training": sorted(set(recoverable)),
            "rerun_required_blockers": sorted(set(rerun)),
        }
    )
    return base


def validate_comparison_readiness(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate observed post-run readiness and its fail-closed invariant."""

    record = _mapping(record, "comparison readiness")
    if record.get("format") != COMPARISON_READINESS_FORMAT:
        raise ValueError("unsupported comparison-readiness format")
    _text(record.get("candidate_id"), "comparison readiness.candidate_id")
    if record.get("reference_id") is not None:
        _text(record.get("reference_id"), "comparison readiness.reference_id")
    comparison_class = record.get("comparison_class")
    if comparison_class not in COMPARISON_CLASSES:
        raise ValueError("invalid comparison class")
    if not isinstance(record.get("strict_ready"), bool):
        raise ValueError("comparison readiness.strict_ready must be boolean")
    declared = _validate_intervention_scope(
        experiment_purpose=record.get("experiment_purpose"),
        intervention_group_id=record.get("intervention_group_id"),
        mechanism_id=record.get("mechanism_id"),
        declared_intervention_fields=record.get("declared_intervention_fields"),
        require_nonempty=bool(record["strict_ready"]),
    )
    for field in (
        "missing_fields",
        "missing_artifacts",
        "blocker_codes",
        "decision_scope",
        "forbidden_claims",
        "recoverable_without_training",
        "rerun_required_blockers",
    ):
        _texts(record.get(field), f"comparison readiness.{field}")
    for field in ("matched_fields", "mismatched_fields"):
        _mapping(record.get(field), f"comparison readiness.{field}")
    for field in (
        "strict_status",
        "reference_prediction_status",
        "row_alignment_status",
        "runtime_certificate_status",
        "role_status",
        "trace_status",
        "stochasticity_status",
    ):
        _text(record.get(field), f"comparison readiness.{field}")
    if record["strict_ready"]:
        if comparison_class != "STRICT_CAUSAL":
            raise ValueError("strict_ready requires STRICT_CAUSAL")
        if record["missing_fields"] or record["missing_artifacts"] or record["blocker_codes"]:
            raise ValueError("strict_ready cannot contain missing evidence or blockers")
        accounted = set(record["matched_fields"]) | set(record["mismatched_fields"])
        if accounted != set(STRICT_IDENTITY_FIELDS):
            raise ValueError("strict_ready must account for every strict identity field")
        undeclared_mismatches = set(record["mismatched_fields"]) - set(
            declared
        )
        if undeclared_mismatches:
            raise ValueError(
                f"strict_ready contains undeclared mismatches: {sorted(undeclared_mismatches)}"
            )
        missing_declared_mismatches = set(declared) - set(record["mismatched_fields"])
        if missing_declared_mismatches:
            raise ValueError(
                "strict_ready declared interventions were not observed as differences: "
                f"{sorted(missing_declared_mismatches)}"
            )
        required_statuses = {
            "strict_status": "READY",
            "reference_prediction_status": "complete",
            "row_alignment_status": "aligned",
            "runtime_certificate_status": "accepted",
            "role_status": "complete",
            "trace_status": "complete",
        }
        bad_statuses = {
            field: {"expected": expected, "actual": record.get(field)}
            for field, expected in required_statuses.items()
            if record.get(field) != expected
        }
        if bad_statuses:
            raise ValueError(f"strict_ready status is incomplete: {bad_statuses}")
        if record.get("reference_id") is None:
            raise ValueError("strict_ready requires a reference_id")
    elif comparison_class == "STRICT_CAUSAL":
        raise ValueError("STRICT_CAUSAL must be strict_ready")
    return dict(record)


def assess_comparison_prelaunch(
    *,
    candidate_id: str,
    candidate_plan: Mapping[str, Any],
    reference_id: str | None,
    reference_bundle: Mapping[str, Any] | None,
    experiment_purpose: str,
    intervention_group_id: str,
    declared_intervention_fields: Sequence[str],
    role_applicability_plan: Mapping[str, Any],
    trace_plan: Mapping[str, Any],
    runtime_qualification_plan: Mapping[str, Any],
    mechanism_id: str | None = None,
) -> dict[str, Any]:
    """Assess planned comparability without demanding future terminal artifacts."""

    _text(candidate_id, "candidate ID")
    candidate_plan = _mapping(candidate_plan, "candidate plan")
    declared = _validate_intervention_scope(
        experiment_purpose=experiment_purpose,
        intervention_group_id=intervention_group_id,
        mechanism_id=mechanism_id,
        declared_intervention_fields=declared_intervention_fields,
        require_nonempty=experiment_purpose in INTERVENTION_FIELDS_BY_PURPOSE,
    )
    roles = validate_role_applicability_plan(role_applicability_plan)
    trace = validate_trace_plan(trace_plan)
    runtime = validate_runtime_qualification_plan(runtime_qualification_plan)

    candidate_identity = candidate_plan.get("comparison_identity", {})
    if not isinstance(candidate_identity, Mapping):
        candidate_identity = {}
    source_status = str(candidate_plan.get("source_config_status", "unknown"))
    source_identity = candidate_plan.get("source_commit_or_archive")

    blockers: list[str] = []
    missing_fields = [
        f"candidate.{field}"
        for field in STRICT_IDENTITY_FIELDS
        if field not in candidate_identity
    ]
    if source_status != "frozen" or not isinstance(source_identity, str) or not source_identity:
        blockers.append("CANDIDATE_SOURCE_CONFIG_NOT_FROZEN")
    if missing_fields:
        blockers.append("PLANNED_IDENTITY_INCOMPLETE")

    reference_identity: Mapping[str, Any] = {}
    reference_bundle_id: str | None = None
    causal = experiment_purpose in INTERVENTION_FIELDS_BY_PURPOSE
    if reference_bundle is None or reference_id is None:
        if causal:
            blockers.append("REFERENCE_BUNDLE_UNAVAILABLE")
    else:
        validated_reference = validate_reference_bundle(reference_bundle)
        if validated_reference["reference_id"] != reference_id:
            raise ValueError("reference bundle identity does not match reference_id")
        reference_bundle_id = validated_reference["reference_bundle_id"]
        reference_identity = validated_reference["comparison_identity"]

    matched: dict[str, Any] = {}
    mismatched: dict[str, Any] = {}
    if reference_identity:
        for field in STRICT_IDENTITY_FIELDS:
            if field not in reference_identity:
                missing_fields.append(f"reference.{field}")
            elif field in candidate_identity:
                if candidate_identity[field] == reference_identity[field]:
                    matched[field] = candidate_identity[field]
                else:
                    mismatched[field] = {
                        "candidate": candidate_identity[field],
                        "reference": reference_identity[field],
                    }
    non_intervention_mismatch = sorted(set(mismatched) - set(declared))
    if non_intervention_mismatch:
        blockers.append("PLANNED_SCIENTIFIC_CONTRACT_MISMATCH")
    if causal and set(declared) - set(mismatched):
        blockers.append("PLANNED_INTERVENTION_NOT_REFLECTED_IN_IDENTITY")
    if any(field.startswith("reference.") for field in missing_fields):
        blockers.append("REFERENCE_IDENTITY_INCOMPLETE")

    prelaunch_ready = not blockers
    planned_status = (
        PRELAUNCH_STRICT_STATUS
        if prelaunch_ready and causal
        else "PRELAUNCH_NONCAUSAL_PLANNED"
        if prelaunch_ready
        else "PRELAUNCH_BLOCKED"
    )
    return {
        "format": COMPARISON_PRELAUNCH_FORMAT,
        "candidate_id": candidate_id,
        "reference_id": reference_id,
        "reference_bundle_id": reference_bundle_id,
        "experiment_purpose": experiment_purpose,
        "intervention_group_id": intervention_group_id,
        "mechanism_id": mechanism_id,
        "planned_status": planned_status,
        "prelaunch_ready": prelaunch_ready,
        "declared_intervention_fields": declared,
        "matched_fields": matched,
        "mismatched_fields": mismatched,
        "missing_fields": sorted(set(missing_fields)),
        "blocker_codes": sorted(set(blockers)),
        "candidate_source_config_status": source_status,
        "candidate_source_commit_or_archive": source_identity,
        "role_applicability_plan": roles,
        "trace_plan": trace,
        "runtime_qualification_plan": runtime,
        "terminal_candidate_artifacts_required": False,
    }


def validate_comparison_prelaunch(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate planned comparability while keeping it distinct from post-run evidence."""

    record = _mapping(record, "comparison prelaunch")
    if record.get("format") != COMPARISON_PRELAUNCH_FORMAT:
        raise ValueError("unsupported comparison-prelaunch format")
    _text(record.get("candidate_id"), "comparison prelaunch.candidate_id")
    if record.get("reference_id") is not None:
        _text(record.get("reference_id"), "comparison prelaunch.reference_id")
    if record.get("reference_bundle_id") is not None:
        _text(record.get("reference_bundle_id"), "comparison prelaunch.reference_bundle_id")
    if not isinstance(record.get("prelaunch_ready"), bool):
        raise ValueError("comparison prelaunch.prelaunch_ready must be boolean")
    if record.get("terminal_candidate_artifacts_required") is not False:
        raise ValueError("prelaunch must not require terminal candidate artifacts")
    declared = _validate_intervention_scope(
        experiment_purpose=record.get("experiment_purpose"),
        intervention_group_id=record.get("intervention_group_id"),
        mechanism_id=record.get("mechanism_id"),
        declared_intervention_fields=record.get("declared_intervention_fields"),
        require_nonempty=bool(record["prelaunch_ready"]),
    )
    for field in ("matched_fields", "mismatched_fields"):
        _mapping(record.get(field), f"comparison prelaunch.{field}")
    for field in ("missing_fields", "blocker_codes"):
        _texts(record.get(field), f"comparison prelaunch.{field}")
    validate_role_applicability_plan(record.get("role_applicability_plan"))
    validate_trace_plan(record.get("trace_plan"))
    validate_runtime_qualification_plan(record.get("runtime_qualification_plan"))
    _text(record.get("candidate_source_config_status"), "candidate source/config status")
    source_identity = record.get("candidate_source_commit_or_archive")
    if source_identity is not None:
        _text(source_identity, "candidate source commit or archive")
    _text(record.get("planned_status"), "planned comparison status")

    if record["prelaunch_ready"]:
        causal = record["experiment_purpose"] in INTERVENTION_FIELDS_BY_PURPOSE
        expected = PRELAUNCH_STRICT_STATUS if causal else "PRELAUNCH_NONCAUSAL_PLANNED"
        if record["planned_status"] != expected:
            raise ValueError("prelaunch ready status is inconsistent with experiment purpose")
        if record["missing_fields"] or record["blocker_codes"]:
            raise ValueError("prelaunch_ready cannot contain missing fields or blockers")
        if record["candidate_source_config_status"] != "frozen":
            raise ValueError("prelaunch_ready requires frozen candidate source/config")
        _text(source_identity, "candidate source commit or archive")
        if causal and (
            record.get("reference_id") is None
            or record.get("reference_bundle_id") is None
        ):
            raise ValueError("prelaunch_ready requires a reusable reference bundle")
        if causal:
            accounted = set(record["matched_fields"]) | set(record["mismatched_fields"])
            if accounted != set(STRICT_IDENTITY_FIELDS):
                raise ValueError("prelaunch_ready must account for every planned identity field")
        undeclared = set(record["mismatched_fields"]) - set(declared)
        if undeclared:
            raise ValueError(f"prelaunch contains undeclared mismatches: {sorted(undeclared)}")
        missing_declared = set(declared) - set(record["mismatched_fields"])
        if missing_declared:
            raise ValueError(
                "prelaunch declared interventions are not reflected in planned identity: "
                f"{sorted(missing_declared)}"
            )
    elif record["planned_status"] != "PRELAUNCH_BLOCKED":
        raise ValueError("an unready prelaunch record must be PRELAUNCH_BLOCKED")
    return dict(record)


def validate_server_comparison_prelaunch(
    prelaunch: Mapping[str, Any], *, experiment_purpose: str
) -> dict[str, Any]:
    """Hard gate server compute using planned, not terminal, comparability."""

    validated = validate_comparison_prelaunch(prelaunch)
    purpose = _text(experiment_purpose, "experiment purpose")
    if validated["experiment_purpose"] != purpose:
        raise ValueError("prelaunch experiment purpose does not match release request")
    if not validated["prelaunch_ready"]:
        raise ValueError("server training is blocked until planned comparability is complete")
    return {
        "gate": "PASS",
        "experiment_purpose": purpose,
        "planned_status": validated["planned_status"],
        "baseline_rerun_authorized": False,
    }
