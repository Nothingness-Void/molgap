"""Fail-closed READY_FOR_DESKTOP package construction."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from .schemas import validate_ready_package, validate_trajectory


_DURABLE_AVAILABILITY = frozenset(
    {"durable_remote_verified", "durable_ims_verified", "committed_local_artifacts_hash_verified"}
)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"READY prerequisite missing: {label}")
    return value


def _nonempty_list(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"READY prerequisite missing: {label}")
    return list(value)


def _require_evidence_ids(
    ids: list[str], evidence_by_id: Mapping[str, Mapping[str, Any]], label: str
) -> None:
    missing = sorted(set(ids) - set(evidence_by_id))
    if missing:
        raise ValueError(f"READY {label} contains dangling evidence IDs: {missing}")


def _has_measured_native_cost(event: Mapping[str, Any]) -> bool:
    measurement = event.get("measurement")
    if not isinstance(measurement, Mapping):
        return False
    return any(
        isinstance(measurement.get(unit), Mapping)
        and measurement[unit].get("status") == "measured"
        and isinstance(measurement[unit].get("value"), (int, float))
        for unit in ("device_hours", "cpu_hours")
    )


def build_ready_package(
    trajectory: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    cost_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    validate_trajectory(trajectory)
    if trajectory.get("record_mode") != "prospective":
        raise ValueError("READY rejects retrospective_partial trajectories")
    if trajectory.get("owner") != "server":
        raise ValueError("READY requires a server-owned trajectory")
    decision = trajectory.get("decision")
    if not isinstance(decision, Mapping) or decision.get("outcome") != "READY_FOR_DESKTOP":
        raise ValueError("READY requires a final READY_FOR_DESKTOP decision")
    if decision.get("final") is not True:
        raise ValueError("READY requires decision.final=true")
    _nonempty_text(decision.get("decision_ref"), "final decision pointer")

    readiness = trajectory.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("READY requires an explicit readiness block")
    candidate_evidence_id = _nonempty_text(
        readiness.get("candidate_evidence_id"), "candidate evidence ID"
    )
    qualification_100k = _nonempty_list(
        readiness.get("qualification_100k_evidence_ids"), "100K qualification evidence"
    )
    funnel_requires_500k = readiness.get("funnel_requires_500k")
    if not isinstance(funnel_requires_500k, bool):
        raise ValueError("READY requires explicit funnel_requires_500k")
    qualification_500k = readiness.get("qualification_500k_evidence_ids", [])
    if funnel_requires_500k:
        qualification_500k = _nonempty_list(
            qualification_500k, "500K qualification evidence"
        )
    elif not isinstance(qualification_500k, list):
        raise ValueError("READY qualification_500k_evidence_ids must be a list")
    reference_id = _nonempty_text(
        readiness.get("compatible_reference_id"), "compatible reference ID"
    )
    evidence_ids = [
        candidate_evidence_id,
        reference_id,
        *qualification_100k,
        *qualification_500k,
    ]
    _require_evidence_ids(evidence_ids, evidence_by_id, "evidence/reference set")
    if candidate_evidence_id not in trajectory["result"]["evidence_ids"]:
        raise ValueError("READY candidate evidence is absent from trajectory result")

    candidate_contract = _nonempty_text(
        readiness.get("candidate_contract_identity"), "candidate contract identity"
    )
    reference_contract = _nonempty_text(
        readiness.get("reference_contract_identity"), "reference contract identity"
    )
    source_config_identity = _nonempty_text(
        readiness.get("source_config_identity"), "source/config identity"
    )
    paired_comparison_ref = _nonempty_text(
        readiness.get("paired_comparison_ref"), "paired comparison"
    )
    role_history_refs = _nonempty_list(
        readiness.get("role_history_refs"), "role history"
    )
    cost_event_ids = _nonempty_list(
        readiness.get("native_cost_event_ids"), "native cost evidence"
    )
    limitations = readiness.get("known_limitations")
    if not isinstance(limitations, list) or any(not isinstance(item, str) for item in limitations):
        raise ValueError("READY requires an explicit limitations list")
    full_scale_question = _nonempty_text(
        readiness.get("full_scale_question"), "exact full-scale question"
    )

    missing_costs = sorted(set(cost_event_ids) - set(cost_by_id))
    if missing_costs:
        raise ValueError(f"READY native cost IDs are dangling: {missing_costs}")
    if not any(_has_measured_native_cost(cost_by_id[event_id]) for event_id in cost_event_ids):
        raise ValueError("READY requires measured native device or CPU cost")

    candidate = evidence_by_id[candidate_evidence_id]
    if candidate["outcome"].get("transfer_status") not in {
        "qualified",
        "ready_for_desktop",
    }:
        raise ValueError("READY candidate transfer status is not qualified")
    if candidate["outcome"].get("comparison_status") in {
        "pending",
        "incompatible_with_strict_v4",
        "not_evaluated",
    }:
        raise ValueError("READY candidate lacks a strict completed comparison")
    if candidate.get("legacy_contract") != candidate_contract:
        raise ValueError("READY candidate contract identity mismatch")
    reference = evidence_by_id[reference_id]
    if reference.get("legacy_contract") != reference_contract:
        raise ValueError("READY reference contract identity mismatch")

    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("READY candidate has no artifact locators")
    artifact_refs = []
    artifact_hashes = {}
    for artifact in artifacts:
        locator = _nonempty_text(artifact.get("locator"), "artifact locator")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"READY artifact lacks SHA256: {artifact.get('name')}")
        if artifact.get("availability") not in _DURABLE_AVAILABILITY:
            raise ValueError(f"READY artifact is not durably verified: {artifact.get('name')}")
        artifact_refs.append(locator)
        artifact_hashes[locator] = digest.lower()

    state = trajectory["state_at_start"]
    source_commit = _nonempty_text(state.get("source_commit"), "source commit")
    if source_commit == "unknown_retrospective":
        raise ValueError("READY source commit cannot be unknown")
    if state.get("source_config_identity") != source_config_identity:
        raise ValueError("READY source/config identity mismatch")
    contract_refs = _nonempty_list(state.get("contract_refs"), "contract pointers")
    if reference_id not in state.get("reference_ids", []):
        raise ValueError("READY compatible reference is absent from trajectory state")

    package = {
        "schema": "molgap-ready-for-desktop-v1",
        "package_id": f"ready-{trajectory['trajectory_id']}",
        "trajectory_id": trajectory["trajectory_id"],
        "candidate_evidence_id": candidate_evidence_id,
        "qualification_100k_evidence_ids": sorted(set(qualification_100k)),
        "qualification_500k_evidence_ids": sorted(set(qualification_500k)),
        "funnel_requires_500k": funnel_requires_500k,
        "source_commit": source_commit,
        "candidate_contract_identity": candidate_contract,
        "reference_contract_identity": reference_contract,
        "source_config_identity": source_config_identity,
        "contract_refs": contract_refs,
        "reference_ids": [reference_id],
        "comparison_refs": [paired_comparison_ref],
        "role_history_refs": role_history_refs,
        "native_cost_event_ids": sorted(set(cost_event_ids)),
        "cost_summary_ref": "research_memory/derived/cost_ledger.json",
        "artifact_refs": sorted(set(artifact_refs)),
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "transfer_status": "qualified",
        "known_limitations": limitations,
        "desktop_question": full_scale_question,
    }
    return validate_ready_package(package)
