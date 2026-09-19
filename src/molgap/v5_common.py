"""Machine-neutral V5 evidence-envelope semantics.

This module intentionally has no desktop, server, scheduler, screen-policy, or
optional runtime dependency.  Both workflow adapters and the research-memory
compiler may import it without reversing their ownership boundary.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .comparison_readiness import (
    COMPARISON_CLASSES,
    COMPARISON_PRELAUNCH_FORMAT,
    COMPARISON_READINESS_FORMAT,
    CAUSAL_REQUIRED_ROLE_KINDS,
    CAUSAL_REQUIRED_TRACE_FIELDS,
    HISTORICAL_COMPARISON_CLASSES,
    INTERVENTION_FIELDS_BY_PURPOSE,
    PRELAUNCH_STRICT_STATUS,
    PROSPECTIVE_REUSE_STATUSES,
    REFERENCE_BUNDLE_FORMAT,
    ROLE_APPLICABILITY_STATES,
    ROLE_EVENT_KINDS,
    REQUIRED_CANDIDATE_OBSERVED_BINDINGS,
    REQUIRED_OBSERVED_BINDINGS,
    STRICT_IDENTITY_FIELDS,
    TARGET_TRANSFORM_FORMAT,
    TRACE_FIELD_DECLARATIONS,
    assess_comparison_prelaunch,
    assess_comparison_readiness,
    reference_bundle_digest,
    target_transform_asset_digest,
    validate_artifact_bindings,
    validate_comparison_prelaunch,
    validate_comparison_readiness,
    validate_historical_comparison_metadata,
    validate_observed_role_events,
    validate_reference_bundle,
    validate_role_applicability_plan,
    validate_runtime_qualification_plan,
    validate_server_comparison_prelaunch,
    validate_stochasticity,
    validate_target_transform_asset,
    validate_trace_plan,
)


V5_CONTRACT_ID = "MOLGAP-COMMON-V5-FINAL"
V5_EVIDENCE_FORMAT = "molgap-v5-evidence-envelope-v1"

OUTCOME_FIELDS = (
    "execution_status",
    "artifact_status",
    "comparison_status",
    "scientific_status",
    "transfer_status",
    "budget_decision",
    "full_handoff_status",
)

_ROLE_USE_STATES = frozenset({"untouched", "consumed", "not_applicable"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _non_empty_fields(record: Mapping[str, Any], fields: Sequence[str]) -> list[str]:
    return [
        field
        for field in fields
        if not isinstance(record.get(field), str) or not record[field].strip()
    ]


def validate_v5_evidence_envelope(
    evidence: Mapping[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    """Validate a pointer-only V5 wrapper around immutable evidence."""

    if not isinstance(evidence, Mapping):
        raise TypeError("V5 evidence envelope must be a mapping")
    if evidence.get("format") != V5_EVIDENCE_FORMAT:
        raise ValueError("unsupported V5 evidence envelope format")
    if evidence.get("contract") != V5_CONTRACT_ID:
        raise ValueError("V5 evidence envelope has the wrong contract")
    if "accepted" in evidence:
        raise ValueError("V5 evidence must use separate outcome dimensions")

    missing = _non_empty_fields(
        evidence, ("evidence_id", "track", "scope", "legacy_contract")
    )
    if missing:
        raise ValueError(f"V5 evidence identity is incomplete: {missing}")

    outcome = evidence.get("outcome")
    if not isinstance(outcome, Mapping):
        raise ValueError("V5 evidence outcome must be a mapping")
    if set(outcome) != set(OUTCOME_FIELDS):
        raise ValueError("V5 evidence outcome fields are incomplete or unknown")
    empty_outcomes = _non_empty_fields(outcome, OUTCOME_FIELDS)
    if empty_outcomes:
        raise ValueError(f"V5 evidence outcome values are empty: {empty_outcomes}")

    authority = evidence.get("authority")
    pointers = authority.get("pointers") if isinstance(authority, Mapping) else None
    if (
        not isinstance(pointers, Sequence)
        or isinstance(pointers, (str, bytes))
        or not pointers
        or any(not isinstance(pointer, str) or not pointer.strip() for pointer in pointers)
    ):
        raise ValueError("V5 evidence requires non-empty authority pointers")
    if repo_root is not None:
        root = Path(repo_root).resolve()
        for pointer in pointers:
            path = (root / pointer).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"authority pointer escapes repository: {pointer}") from exc
            if not path.is_file():
                raise ValueError(f"authority pointer is missing: {pointer}")

    role_use = evidence.get("role_use")
    if not isinstance(role_use, Mapping):
        raise ValueError("V5 evidence role_use must be a mapping")
    for role in ("official_validation", "test_dev", "test_challenge"):
        if role_use.get(role) not in _ROLE_USE_STATES:
            raise ValueError(f"invalid or missing role-use state: {role}")

    artifacts = evidence.get("artifacts")
    if (
        not isinstance(artifacts, Sequence)
        or isinstance(artifacts, (str, bytes))
        or not artifacts
    ):
        raise ValueError("V5 evidence artifacts must be a non-empty sequence")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("V5 evidence artifact entries must be mappings")
        artifact_missing = _non_empty_fields(
            artifact, ("name", "locator", "availability")
        )
        if artifact_missing:
            raise ValueError(f"V5 evidence artifact is incomplete: {artifact_missing}")
        digest = artifact.get("sha256")
        if digest is not None and (
            not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest)
        ):
            raise ValueError(f"invalid artifact SHA256: {artifact.get('name')}")

    migration = evidence.get("migration")
    if not isinstance(migration, Mapping):
        raise ValueError("V5 evidence migration record must be a mapping")
    migration_missing = _non_empty_fields(
        migration, ("migrated_at", "verification_scope")
    )
    if migration_missing:
        raise ValueError(f"V5 evidence migration record is incomplete: {migration_missing}")
    for field in ("training_executed", "inference_executed", "scientific_reinterpretation"):
        if migration.get(field) is not False:
            raise ValueError(f"historical migration must record {field}=false")

    return {
        "format": V5_EVIDENCE_FORMAT,
        "evidence_id": evidence["evidence_id"],
        "outcome": dict(outcome),
        "authority_pointer_count": len(pointers),
        "artifact_count": len(artifacts),
        "valid": True,
    }
