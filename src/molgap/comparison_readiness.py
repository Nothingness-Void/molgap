"""Fail-closed V5 comparison readiness and reusable-reference semantics.

The helpers in this module are machine-neutral.  They classify existing or
planned evidence; they never launch training, read a protected role, or infer a
missing historical fact.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


COMPARISON_READINESS_FORMAT = "molgap-comparison-readiness-v1"
REFERENCE_BUNDLE_FORMAT = "molgap-reference-bundle-v1"
TARGET_TRANSFORM_FORMAT = "molgap-target-transform-asset-v1"

COMPARISON_CLASSES = frozenset(
    {
        "STRICT_CAUSAL",
        "PAIRED_ENDPOINT",
        "MATCHED_PREFIX",
        "CONTEXT_ONLY",
        "NO_COMPARISON",
    }
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

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPLETE = frozenset({"complete", "accepted", "available"})


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


def assess_comparison_readiness(
    *,
    candidate_id: str,
    candidate: Mapping[str, Any],
    reference_id: str | None,
    reference: Mapping[str, Any] | None,
    declared_intervention_fields: Sequence[str] = (),
    prefix_aligned: bool = False,
    scalar_context_available: bool = False,
) -> dict[str, Any]:
    """Classify one comparison without inferring absent historical facts."""

    _text(candidate_id, "candidate_id")
    candidate = _mapping(candidate, "candidate")
    declared = _texts(declared_intervention_fields, "declared intervention fields")
    unknown_declared = sorted(set(declared) - set(STRICT_IDENTITY_FIELDS))
    if unknown_declared:
        raise ValueError(f"unknown declared intervention fields: {unknown_declared}")

    base = {
        "format": COMPARISON_READINESS_FORMAT,
        "candidate_id": candidate_id,
        "reference_id": reference_id,
        "strict_ready": False,
        "strict_status": "PENDING",
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
        if candidate.get("role_status") == "complete" and reference.get("role_status") == "complete"
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
        elif blocker in {"SCIENTIFIC_CONTRACT_MISMATCH", "TERMINAL_ENDPOINT_UNAVAILABLE"}:
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
    """Validate a readiness record and re-check the fail-closed invariant."""

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
    for field in (
        "declared_intervention_fields",
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
            record["declared_intervention_fields"]
        )
        if undeclared_mismatches:
            raise ValueError(
                f"strict_ready contains undeclared mismatches: {sorted(undeclared_mismatches)}"
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


def validate_server_comparison_prelaunch(
    readiness: Mapping[str, Any], *, experiment_purpose: str
) -> dict[str, Any]:
    """Hard gate future server training before compute is released."""

    validated = validate_comparison_readiness(readiness)
    purpose = _text(experiment_purpose, "experiment purpose")
    causal = purpose in {"architecture_comparison", "mechanism_comparison"}
    allowed_noncausal = {
        "diagnostic",
        "transfer_study",
        "delivery_experiment",
        "contextual_experiment",
        "NO_TRAIN",
    }
    if causal and not validated["strict_ready"]:
        raise ValueError("causal server training is blocked until prelaunch STRICT_CAUSAL readiness")
    if not causal and purpose not in allowed_noncausal:
        raise ValueError("non-causal server training requires an explicit V5 purpose label")
    return {
        "gate": "PASS",
        "experiment_purpose": purpose,
        "comparison_class": validated["comparison_class"],
        "baseline_rerun_authorized": False,
    }
