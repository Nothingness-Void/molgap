"""Fail-closed V5 acceptance helpers for server-owned 100K/500K evidence.

The existing scientific screen policy remains the authority for matched
contracts.  This module wraps it with the V5 evidence transaction and keeps
mechanical acceptance, scientific interpretation, cost, transfer, and desktop
handoff as separate fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from .comparison_readiness import validate_server_comparison_prelaunch
from .evidence_pointers import validate_release_reference_bundle_evidence

from .screen_policy import (
    REFERENCE_MATCH_FIELDS,
    REFERENCE_PROVENANCE_FIELDS,
    validate_reference_screen_contract,
)


OUTCOME_FIELDS = (
    "execution_status",
    "artifact_status",
    "comparison_status",
    "scientific_status",
    "transfer_status",
    "budget_decision",
    "full_handoff_status",
)
READY_FOR_DESKTOP = "READY_FOR_DESKTOP"
OLD_REFERENCE_POLICIES = frozenset({"molgap-screen-comparability-v3", "v3"})
def validate_server_scientific_prelaunch(
    *,
    comparison_prelaunch: Mapping[str, Any],
    experiment_purpose: str,
    reference_bundle: Mapping[str, Any] | None,
    repo_root: str | Path,
    reference_bundle_path: str | Path,
) -> dict[str, Any]:
    """Release compute from the planned gate, never from post-run readiness."""

    if reference_bundle is None:
        raise ValueError("server scientific prelaunch requires a reference bundle")
    validated_reference_bundle = validate_release_reference_bundle_evidence(
        reference_bundle,
        repo_root=repo_root,
        reference_bundle_path=reference_bundle_path,
    )

    gate = validate_server_comparison_prelaunch(
        comparison_prelaunch,
        experiment_purpose=experiment_purpose,
        reference_bundle=validated_reference_bundle,
    )
    return {
        **gate,
        "role_applicability_declared": True,
        "trace_fields_declared": True,
        "runtime_qualification_declared": True,
    }


def write_server_comparison_prelaunch(
    path: Path,
    *,
    comparison_prelaunch: Mapping[str, Any],
    experiment_purpose: str,
    reference_bundle: Mapping[str, Any] | None,
    repo_root: str | Path,
    reference_bundle_path: str | Path,
) -> str:
    """Atomically persist the mandatory prelaunch gate after validation."""

    path = Path(path)
    if path.name != "comparison_readiness_prelaunch.json":
        raise ValueError("server prelaunch record must be comparison_readiness_prelaunch.json")
    validate_server_scientific_prelaunch(
        comparison_prelaunch=comparison_prelaunch,
        experiment_purpose=experiment_purpose,
        reference_bundle=reference_bundle,
        repo_root=repo_root,
        reference_bundle_path=reference_bundle_path,
    )
    record = dict(comparison_prelaunch)
    payload = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        temporary = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_outcome(
    *,
    execution_status: str = "PENDING",
    artifact_status: str = "PENDING",
    comparison_status: str = "PENDING",
    scientific_status: str = "PENDING",
    transfer_status: str = "NOT_READY",
    budget_decision: str = "PENDING",
    full_handoff_status: str = "NONE",
) -> dict[str, str]:
    """Create the seven-field V5 outcome without an overloaded accepted flag."""

    values = {
        "execution_status": execution_status,
        "artifact_status": artifact_status,
        "comparison_status": comparison_status,
        "scientific_status": scientific_status,
        "transfer_status": transfer_status,
        "budget_decision": budget_decision,
        "full_handoff_status": full_handoff_status,
    }
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise ValueError(f"V5 outcome fields must be non-empty strings: {values}")
    return values


def _pending_outcome(*, reason: str, missing: Sequence[str]) -> dict[str, Any]:
    outcome = make_outcome()
    outcome.update({"reason": reason, "missing_evidence": sorted(set(missing))})
    return outcome


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 hex string") from exc
    return value


def _flat_finite(value: Any, label: str) -> list[float]:
    """Return a one-dimensional finite numeric list, rejecting shape drift."""

    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} has the wrong shape; expected a one-dimensional sequence")
    if any(isinstance(item, (Sequence, Mapping)) for item in value):
        raise ValueError(f"{label} has the wrong shape; expected a one-dimensional sequence")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{label} contains a non-numeric value")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains a non-numeric value") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label} contains NaN or Inf")
        result.append(number)
    if not result:
        raise ValueError(f"{label} is empty")
    return result


def _source_indices(value: Any, label: str) -> list[int]:
    values = _flat_finite(value, label)
    result = []
    for item in values:
        if item != int(item):
            raise ValueError(f"{label} contains a non-integral source index")
        result.append(int(item))
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate source_idx values")
    return result


def validate_prediction_bundle(bundle: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Validate finite predictions, targets, strict shape, and source identity."""

    missing = [key for key in ("prediction", "target", "source_idx") if key not in bundle]
    if missing:
        raise ValueError(f"{label} bundle is missing: {missing}")
    prediction = _flat_finite(bundle["prediction"], f"{label}.prediction")
    target = _flat_finite(bundle["target"], f"{label}.target")
    source_idx = _source_indices(bundle["source_idx"], f"{label}.source_idx")
    if not (len(prediction) == len(target) == len(source_idx)):
        raise ValueError(f"{label} prediction/target/source_idx shapes are misaligned")
    return {
        "rows": len(source_idx),
        "prediction": prediction,
        "target": target,
        "source_idx": source_idx,
    }


def validate_paired_predictions(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Require identical, unique source rows before any paired comparison."""

    reference_checked = validate_prediction_bundle(reference, "reference")
    candidate_checked = validate_prediction_bundle(candidate, "candidate")
    if reference_checked["source_idx"] != candidate_checked["source_idx"]:
        raise ValueError("reference and candidate source_idx are not aligned")
    if reference_checked["rows"] != candidate_checked["rows"]:
        raise ValueError("reference and candidate row counts differ")
    if reference_checked["target"] != candidate_checked["target"]:
        raise ValueError("reference and candidate targets are not aligned")
    return {"reference": reference_checked, "candidate": candidate_checked}


def _contract_missing(contract: Mapping[str, Any]) -> list[str]:
    required = (*REFERENCE_MATCH_FIELDS, *REFERENCE_PROVENANCE_FIELDS)
    return [field for field in required if field not in contract]


def _validate_artifacts(artifacts: Mapping[str, Any]) -> None:
    if not artifacts:
        raise ValueError("artifact hashes are empty")
    for name, digest in artifacts.items():
        _sha256(digest, f"artifact hash {name}")


def assess_strict_comparison(
    *,
    reference_contract: Optional[Mapping[str, Any]],
    candidate_contract: Mapping[str, Any],
    runtime_certificates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    reference_bundle: Optional[Mapping[str, Any]] = None,
    candidate_bundle: Optional[Mapping[str, Any]] = None,
    paired_analysis: Optional[Mapping[str, Any]] = None,
    bootstrap: Optional[Mapping[str, Any]] = None,
    artifact_hashes: Optional[Mapping[str, Any]] = None,
    resume_cursor: Optional[Mapping[str, Any]] = None,
    role_history: Optional[Mapping[str, Any]] = None,
    actual_native_cost: Optional[Mapping[str, Any]] = None,
    scientific_status: str = "PENDING",
    budget_decision: str = "PENDING",
) -> dict[str, Any]:
    """Run a fail-closed comparison transaction.

    Missing evidence returns ``comparison_status=PENDING``.  Malformed evidence
    raises because accepting it would hide a broken artifact.  No path in this
    function retrains a reference model or consumes a protected role.
    """

    if not isinstance(candidate_contract, Mapping):
        raise ValueError("candidate_contract must be a mapping")
    if reference_contract is None:
        return _pending_outcome(
            reason="missing_reference",
            missing=("reference_contract", "reference_bundle"),
        )
    if not isinstance(reference_contract, Mapping):
        raise ValueError("reference_contract must be a mapping")
    policy = reference_contract.get("policy") or reference_contract.get("screen_policy")
    if policy in OLD_REFERENCE_POLICIES:
        return {
            **make_outcome(comparison_status="REJECTED"),
            "reason": "old_reference_policy_incompatible",
            "reference_policy": policy,
        }

    missing: list[str] = []
    for label, contract in (("reference", reference_contract), ("candidate", candidate_contract)):
        missing.extend(f"{label}.{field}" for field in _contract_missing(contract))
    if runtime_certificates is None:
        missing.append("runtime_certificates")
    if reference_bundle is None:
        missing.append("reference_bundle")
    if candidate_bundle is None:
        missing.append("candidate_bundle")
    if paired_analysis is None:
        missing.append("paired_analysis")
    if bootstrap is None:
        missing.append("paired_bootstrap")
    if artifact_hashes is None:
        missing.append("artifact_hashes")
    if resume_cursor is None:
        missing.append("resume_cursor")
    if role_history is None:
        missing.append("role_history")
    if actual_native_cost is None:
        missing.append("actual_native_cost")
    if missing:
        return _pending_outcome(reason="incomplete_comparison_evidence", missing=missing)

    try:
        contract_check = validate_reference_screen_contract(
            reference=reference_contract,
            candidate=candidate_contract,
            runtime_certificates=runtime_certificates,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            **make_outcome(comparison_status="REJECTED"),
            "reason": "reference_contract_incompatible",
            "detail": str(exc),
        }
    paired = validate_paired_predictions(reference_bundle, candidate_bundle)
    _validate_artifacts(artifact_hashes)
    if not isinstance(paired_analysis, Mapping) or not paired_analysis:
        raise ValueError("paired_analysis must be a non-empty mapping")
    if not isinstance(bootstrap, Mapping) or not bootstrap:
        raise ValueError("paired_bootstrap must be a non-empty mapping")
    if not isinstance(resume_cursor, Mapping) or not resume_cursor:
        raise ValueError("resume_cursor must be a non-empty mapping")
    if not isinstance(role_history, Mapping) or not role_history:
        raise ValueError("role_history must be a non-empty mapping")
    if not isinstance(actual_native_cost, Mapping) or not actual_native_cost:
        raise ValueError("actual_native_cost must be a non-empty mapping")

    normalized_scientific = str(scientific_status).upper()
    normalized_budget = str(budget_decision).upper()
    scale_rows = int(candidate_contract.get("scale_rows", 0))
    qualified = normalized_scientific in {"QUALIFIED", "POSITIVE"}
    budget_allows_transfer = normalized_budget not in {"STOP_FOR_COST", "REJECTED"}
    ready = scale_rows >= 500_000 and qualified and budget_allows_transfer
    outcome = make_outcome(
        execution_status=str(candidate_contract.get("execution_status", "COMPLETE")).upper(),
        artifact_status="ACCEPTED",
        comparison_status="ACCEPTED",
        scientific_status=normalized_scientific,
        transfer_status=READY_FOR_DESKTOP if ready else "NOT_READY",
        budget_decision=normalized_budget,
        full_handoff_status=READY_FOR_DESKTOP if ready else "NONE",
    )
    outcome.update(
        {
            "strict_reference_mode": contract_check["comparison_mode"],
            "scientific_contract_sha256": contract_check["scientific_contract_sha256"],
            "paired_rows": paired["candidate"]["rows"],
            "reference_run_id": contract_check["reference_run_id"],
            "candidate_run_id": contract_check["candidate_run_id"],
            "native_cost": dict(actual_native_cost),
            "role_history": dict(role_history),
        }
    )
    return outcome


def write_ready_for_desktop_package(path: Path, package: Mapping[str, Any]) -> str:
    """Atomically write a durable handoff package without waking desktop."""

    required = (
        "format",
        "source_config_identity",
        "scale_decisions",
        "reference_candidate_identity",
        "paired_analysis",
        "role_history",
        "recovery_state",
        "native_cost",
        "limitations",
        "full_scale_question",
    )
    missing = [key for key in required if key not in package]
    if missing:
        raise ValueError(f"READY_FOR_DESKTOP package is missing: {missing}")
    if package.get("format") != "molgap-ready-for-desktop-v5":
        raise ValueError("Unsupported READY_FOR_DESKTOP package format")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(package), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary: Optional[Path] = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        temporary = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
