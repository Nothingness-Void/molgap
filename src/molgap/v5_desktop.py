"""Pure local policy gates for the V5 desktop workflow.

The helpers in this module deliberately do not query schedulers, create
automations, or persist live control state.  They turn an already-observed
local binding and authoritative remote observation into a fail-closed plan.
Shared V5 evidence and comparability semantics live in ``v5_common`` and
``comparison_readiness``; Desktop-specific promotion remains governed by the
existing V4 screen policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .screen_policy import (
    REFERENCE_MATCH_FIELDS,
    REFERENCE_PROVENANCE_FIELDS,
    REFERENCE_SCREEN_POLICY,
    validate_reference_screen_contract,
)
from .v5_common import (
    OUTCOME_FIELDS,
    V5_CONTRACT_ID,
    V5_EVIDENCE_FORMAT,
    _non_empty_fields,
    validate_v5_evidence_envelope,
)


V5_DESKTOP_CONTRACT_ID = "MOLGAP-DESKTOP-V5-FINAL"

ACTIVE_REMOTE_STATES = frozenset({"queued", "running"})
TERMINAL_REMOTE_STATES = frozenset({"complete", "failed", "cancelled"})
PROTECTED_EVALUATION_ROLES = frozenset(
    {"official_validation", "test_dev", "test_challenge"}
)

_DURABILITY_FIELDS = (
    "remote_job_id",
    "source_config_identity",
    "accepted_input_identity",
    "remote_output_locator",
    "remote_checkpoint_locator",
    "resume_contract",
    "provenance",
)
_DURABILITY_FLAGS = (
    "source_config_frozen",
    "input_cache_accepted",
    "remote_artifacts_durable",
    "resume_supported",
)


def _outcome(**values: Any) -> dict[str, Any]:
    """Return explicit V5 outcome dimensions instead of one overloaded flag."""
    result = {field: "not_evaluated" for field in OUTCOME_FIELDS}
    result.update(values)
    return result


def validate_desktop_shutdown_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the durable local record required before desktop shutdown.

    A desktop binding is intentionally self-contained.  It never requires a
    server acknowledgement or a server-owned monitor, and it cannot be used
    to transfer ownership of the remote job.
    """
    if not isinstance(binding, Mapping):
        raise TypeError("desktop shutdown binding must be a mapping")
    if binding.get("owner") != "desktop":
        raise ValueError("desktop shutdown binding must remain desktop-owned")

    missing = _non_empty_fields(binding, _DURABILITY_FIELDS)
    missing += [field for field in _DURABILITY_FLAGS if binding.get(field) is not True]
    if missing:
        raise ValueError(f"desktop shutdown binding is not durable: {missing}")

    forbidden = (
        "server_handoff_required",
        "server_monitor_required",
        "server_ack_required",
    )
    if any(binding.get(field) is True for field in forbidden):
        raise ValueError("desktop shutdown cannot depend on server handoff")
    if binding.get("monitor_owner") == "server":
        raise ValueError("desktop work cannot be assigned to a server monitor")

    return {
        **_outcome(
            execution_status="bound",
            artifact_status="durable",
            comparison_status="not_applicable",
            scientific_status="not_evaluated",
            transfer_status="not_applicable",
            budget_decision="not_evaluated",
            full_handoff_status="not_applicable",
        ),
        "contract": V5_DESKTOP_CONTRACT_ID,
        "owner": "desktop",
        "remote_job_id": binding["remote_job_id"],
        "shutdown_ready": True,
        "server_handoff_required": False,
        "server_monitor_required": False,
    }


def reconcile_desktop_remote_state(
    binding: Mapping[str, Any], remote_status: Mapping[str, Any]
) -> dict[str, Any]:
    """Plan the next local action from an authoritative remote observation.

    Local conversation state is never treated as scheduler state.  Unknown or
    non-authoritative observations therefore request a fresh status query and
    never authorize a resubmission.
    """
    if not isinstance(binding, Mapping) or binding.get("owner") != "desktop":
        raise ValueError("only desktop-owned bindings can be reconciled here")
    if not isinstance(remote_status, Mapping):
        raise TypeError("remote status must be a mapping")

    observed_authoritatively = bool(
        remote_status.get("authoritative") is True
        or remote_status.get("source") == "authoritative_scheduler"
    )
    expected_job = binding.get("remote_job_id")
    observed_job = remote_status.get("remote_job_id")
    if (
        not observed_authoritatively
        or not expected_job
        or observed_job != expected_job
    ):
        return {
            **_outcome(
                execution_status="unknown",
                artifact_status="unknown",
                comparison_status="pending",
                scientific_status="not_evaluated",
                transfer_status="not_applicable",
                budget_decision="hold",
                full_handoff_status="not_applicable",
            ),
            "next_action": "query_authoritative_status",
            "automatic_resubmission": False,
            "stale_local_state_ignored": True,
        }

    state = str(remote_status.get("state", "unknown")).strip().lower()
    if state in ACTIVE_REMOTE_STATES:
        return {
            **_outcome(
                execution_status=state,
                artifact_status="pending",
                comparison_status="pending",
                scientific_status="not_evaluated",
                transfer_status="not_applicable",
                budget_decision="hold",
                full_handoff_status="not_applicable",
            ),
            "next_action": "wait_for_remote_completion",
            "automatic_resubmission": False,
            "stale_local_state_ignored": True,
        }
    if state == "complete":
        return {
            **_outcome(
                execution_status="complete",
                artifact_status="pending",
                comparison_status="pending",
                scientific_status="not_evaluated",
                transfer_status="pending",
                budget_decision="hold",
                full_handoff_status="not_applicable",
            ),
            "next_action": "retrieve_and_accept_artifacts",
            "automatic_resubmission": False,
            "stale_local_state_ignored": True,
        }
    if state in {"failed", "cancelled"}:
        return {
            **_outcome(
                execution_status=state,
                artifact_status="pending",
                comparison_status="pending",
                scientific_status="not_evaluated",
                transfer_status="not_applicable",
                budget_decision="hold",
                full_handoff_status="not_applicable",
            ),
            "next_action": "diagnose_before_resubmission",
            "automatic_resubmission": False,
            "stale_local_state_ignored": True,
        }
    return {
        **_outcome(
            execution_status="unknown",
            artifact_status="unknown",
            comparison_status="pending",
            scientific_status="not_evaluated",
            transfer_status="not_applicable",
            budget_decision="hold",
            full_handoff_status="not_applicable",
        ),
        "next_action": "query_authoritative_status",
        "automatic_resubmission": False,
        "stale_local_state_ignored": True,
    }


def authorize_desktop_500k_action(
    *,
    experiment_owner: str,
    desktop_online: bool,
    remote_state: str | None,
) -> dict[str, Any]:
    """Keep ownership stable and prevent duplicate desktop/server work."""
    if experiment_owner not in {"desktop", "server"}:
        raise ValueError("experiment_owner must be desktop or server")
    state = str(remote_state or "unknown").strip().lower()
    common = {
        "owner": experiment_owner,
        "server_fallback": False,
        "ownership_transfer": False,
    }
    if experiment_owner == "server":
        active = state in ACTIVE_REMOTE_STATES
        return {
            **common,
            "allowed": False,
            "reason": (
                "server_owned_active_experiment"
                if active
                else "server_owned_experiment"
            ),
            "next_action": "do_not_duplicate" if active else "leave_with_server_owner",
        }
    if experiment_owner == "desktop" and not desktop_online:
        return {
            **common,
            "allowed": False,
            "reason": "desktop_offline_silent_time",
            "next_action": "wait_for_desktop_reconciliation",
        }
    if state == "unknown":
        return {
            **common,
            "allowed": False,
            "reason": "remote_state_not_reconciled",
            "next_action": "query_authoritative_status",
        }
    return {
        **common,
        "allowed": True,
        "reason": "desktop_owned_and_online",
        "next_action": "continue_existing_authority",
    }


def classify_reference_comparison(
    *,
    reference: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
    runtime_certificates: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify reference readiness while delegating strict V4 checks.

    Missing evidence is a pending comparison, never a request to retrain a
    baseline.  A V3 artifact is explicitly incompatible with the V4
    comparator, while a complete V4 payload is checked by the existing shared
    screen-policy implementation.
    """
    base = {
        "baseline_rerun_allowed": False,
        "strict_v4_comparator": False,
    }
    if reference is None:
        return {
            **_outcome(
                execution_status="not_started",
                artifact_status="missing_reference",
                comparison_status="pending",
                scientific_status="not_evaluated",
                transfer_status="pending",
                budget_decision="hold",
                full_handoff_status="not_applicable",
            ),
            **base,
            "reason": "reference_payload_missing",
            "next_action": "locate_accepted_reference",
        }
    if not isinstance(candidate, Mapping):
        raise TypeError("candidate must be a mapping")

    markers = " ".join(
        str(record.get(key, "")).lower()
        for record in (reference, candidate)
        for key in ("policy", "format", "comparison_policy")
    )
    if "v3" in markers and "v4" not in markers:
        return {
            **_outcome(
                execution_status="ready",
                artifact_status="present",
                comparison_status="incompatible",
                scientific_status="not_evaluated",
                transfer_status="not_evaluated",
                budget_decision="hold",
                full_handoff_status="not_applicable",
            ),
            **base,
            "reason": "v3_artifact_is_not_a_strict_v4_comparator",
            "next_action": "use_matching_v4_reference",
        }

    required = (*REFERENCE_MATCH_FIELDS, *REFERENCE_PROVENANCE_FIELDS)
    missing_reference = [field for field in required if field not in reference]
    missing_candidate = [field for field in required if field not in candidate]
    if missing_reference or missing_candidate:
        return {
            **_outcome(
                execution_status="ready",
                artifact_status="incomplete",
                comparison_status="pending",
                scientific_status="not_evaluated",
                transfer_status="pending",
                budget_decision="hold",
                full_handoff_status="not_applicable",
            ),
            **base,
            "reason": "reference_or_candidate_payload_incomplete",
            "missing_reference_fields": missing_reference,
            "missing_candidate_fields": missing_candidate,
            "next_action": "retrieve_complete_evidence",
        }

    certificates = runtime_certificates or {}
    try:
        comparison = validate_reference_screen_contract(
            reference=reference,
            candidate=candidate,
            runtime_certificates=certificates,
        )
    except (KeyError, TypeError, ValueError) as exc:
        reason = "runtime_or_reference_evidence_pending"
        if certificates and "mismatch" in str(exc).lower():
            reason = "strict_v4_contract_mismatch"
        return {
            **_outcome(
                execution_status="ready",
                artifact_status="present",
                comparison_status=(
                    "pending" if reason.endswith("pending") else "incompatible"
                ),
                scientific_status="not_evaluated",
                transfer_status="pending",
                budget_decision="hold",
                full_handoff_status="not_applicable",
            ),
            **base,
            "reason": reason,
            "detail": str(exc),
            "next_action": "complete_or_reconcile_reference_evidence",
        }
    return {
        **_outcome(
            execution_status="ready",
            artifact_status="present",
            comparison_status="ready",
            scientific_status="not_evaluated",
            transfer_status="pending",
            budget_decision="not_evaluated",
            full_handoff_status="not_applicable",
        ),
        "baseline_rerun_allowed": False,
        "strict_v4_comparator": comparison.get("policy") == REFERENCE_SCREEN_POLICY,
        "comparison": comparison,
        "next_action": "perform_paired_comparison",
    }


def decide_role_use(
    role_history: Mapping[str, Any] | Sequence[Any],
    role: str,
    *,
    explicit_authorization: bool = False,
) -> dict[str, Any]:
    """Separate prior role use from permission to read a protected role."""
    if not isinstance(role, str) or not role.strip():
        raise ValueError("role must be non-empty")
    role = role.strip()
    used = False
    if isinstance(role_history, Mapping):
        used = bool(role_history.get(role, False))
    else:
        for item in role_history:
            if item == role:
                used = True
                break
            if isinstance(item, Mapping) and item.get("role") == role:
                used = bool(item.get("used", item.get("consumed", True)))
                if used:
                    break
    protected = role.lower().replace("-", "_") in PROTECTED_EVALUATION_ROLES
    authorization_required = protected and not explicit_authorization
    can_read = not used and not authorization_required
    if used:
        reason = "role_was_previously_consulted"
    elif authorization_required:
        reason = "explicit_authorization_required"
    else:
        reason = "no_prior_use_recorded"
    return {
        "role": role,
        "role_use_status": "consumed" if used else "untouched",
        "can_read": can_read,
        "authorization_status": (
            "required"
            if authorization_required
            else "authorized"
            if protected
            else "not_required"
        ),
        "reason": reason,
    }


def admit_full_scale(
    *,
    candidate_100k: Mapping[str, Any],
    candidate_500k: Mapping[str, Any],
    reference: Mapping[str, Any],
    paired_comparison: Mapping[str, Any],
    explicit_desktop_authorization: bool = False,
) -> dict[str, Any]:
    """Require qualified 500K evidence and explicit desktop authority."""
    checks = (
        (candidate_100k.get("qualified") is True, "100k_not_qualified"),
        (candidate_500k.get("complete") is True, "500k_incomplete"),
        (candidate_500k.get("qualified") is True, "500k_not_qualified"),
        (reference.get("complete") is True, "reference_incomplete"),
        (paired_comparison.get("complete") is True, "paired_comparison_incomplete"),
        (candidate_500k.get("identity_frozen") is True, "candidate_identity_not_frozen"),
        (reference.get("immutable") is True, "reference_not_immutable"),
        (
            paired_comparison.get("artifacts_aligned") is True,
            "comparison_artifacts_not_aligned",
        ),
        (
            paired_comparison.get("strict_comparison_passed") is True,
            "strict_comparison_not_passed",
        ),
        (
            paired_comparison.get("statistical_limitations_recorded") is True,
            "statistical_limitations_missing",
        ),
        (
            candidate_500k.get("target_hardware_cost_recorded") is True,
            "target_hardware_cost_missing",
        ),
        (
            candidate_500k.get("role_use_history_recorded") is True,
            "role_use_history_missing",
        ),
        (
            candidate_500k.get("recovery_schedule_recorded") is True,
            "recovery_schedule_missing",
        ),
        (
            candidate_500k.get("duplicate_full_evidence_checked") is True,
            "duplicate_full_evidence_not_checked",
        ),
    )
    for passed, reason in checks:
        if not passed:
            return {
                **_outcome(
                    execution_status="qualified_screen_only",
                    artifact_status="pending",
                    comparison_status="pending",
                    scientific_status="not_evaluated",
                    transfer_status="not_qualified",
                    budget_decision="hold",
                    full_handoff_status="blocked",
                ),
                "allowed": False,
                "automatic_full": False,
                "reason": reason,
            }
    if not explicit_desktop_authorization:
        return {
            **_outcome(
                execution_status="qualified_screen",
                artifact_status="complete",
                comparison_status="complete",
                scientific_status="qualified_for_review",
                transfer_status="qualified",
                budget_decision="pending_desktop_decision",
                full_handoff_status="ready_for_desktop_decision",
            ),
            "allowed": False,
            "automatic_full": False,
            "reason": "explicit_desktop_authorization_required",
        }
    return {
        **_outcome(
            execution_status="qualified_screen",
            artifact_status="complete",
            comparison_status="complete",
            scientific_status="qualified_for_review",
            transfer_status="qualified",
            budget_decision="authorized",
            full_handoff_status="authorized",
        ),
        "allowed": True,
        "automatic_full": False,
        "reason": "qualified_and_explicitly_authorized",
    }


def track_b_delivery_guard(
    *, positive: bool, explicit_production_gate: bool = False
) -> dict[str, Any]:
    """Keep Track B evidence from silently mutating the Track A registry."""
    return {
        "production_registry_change_allowed": bool(
            positive and explicit_production_gate
        ),
        "reason": (
            "track_b_requires_separate_track_a_gate"
            if positive
            else "no_positive_delivery_evidence"
        ),
        "automatic": False,
    }


def shared_helper_failure_decision(
    helper_name: str, error: str
) -> dict[str, Any]:
    """Preserve a shared-helper failure instead of bypassing it locally."""
    if not helper_name.strip():
        raise ValueError("helper_name must be non-empty")
    return {
        **_outcome(
            execution_status="blocked",
            artifact_status="not_evaluated",
            comparison_status="pending",
            scientific_status="not_evaluated",
            transfer_status="not_applicable",
            budget_decision="hold",
            full_handoff_status="not_applicable",
        ),
        "helper": helper_name,
        "error": error,
        "local_bypass_allowed": False,
        "next_action": "preserve_failure_and_report",
    }
