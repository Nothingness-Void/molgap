"""Role-use aggregation preserving coarse historical semantics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def validate_observed_role_truth(
    declarations: list[dict[str, str]], role_events: list[dict[str, Any]],
) -> None:
    """Check new terminal claims against source-verified events, never role plans."""
    from .schemas import ROLE_ACCESS_KINDS, validate_role_event

    observed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in role_events:
        validate_role_event(event)
        if event["access_kind"] == "selection_used" and not event["selection_used"]:
            raise ValueError("observed selection event contradicts selection_used=false")
        observed[event["role_name"]].append(event)
    requirements = {
        "consumed": ROLE_ACCESS_KINDS,
        "read": {"prediction_input", "labels_read"},
        "used": {"training_membership", "metric_computed", "selection_used", "external_submission"},
        **{kind: {kind} for kind in ROLE_ACCESS_KINDS},
    }
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ValueError("role_use must be an explicit role-to-state mapping")
        for role, state in declaration.items():
            if not isinstance(role, str) or not role or not isinstance(state, str):
                raise ValueError("invalid terminal role declaration")
            events = observed.get(role, [])
            if state in {"untouched", "not_applicable"}:
                if events:
                    raise ValueError(f"role declared {state} but observed access exists: {role}")
            elif state in requirements:
                matched = any(event["access_kind"] in requirements[state] or
                              (state in {"used", "selection_used"} and event["selection_used"])
                              for event in events)
                if not matched:
                    raise ValueError(f"role declared {state} without matching observed event: {role}")
            elif state not in {"unknown", "unavailable"}:
                raise ValueError(f"unsupported terminal role declaration: {role}={state}")


def build_role_reuse_index(
    role_events: list[dict[str, Any]], evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    identities: dict[str, list[dict[str, Any]]] = defaultdict(list)
    explicitly_consumed_role_names: set[str] = set()
    for event in sorted(role_events, key=lambda item: item["role_event_id"]):
        identity = ":".join(
            (event["dataset_identity"], event["row_manifest_hash"], event["role_name"])
        )
        identities[identity].append(
            {
                "trajectory_id": event["trajectory_id"],
                "role_event_id": event["role_event_id"],
                "access_kind": event["access_kind"],
                "selection_used": event["selection_used"],
                "semantics": "explicit",
            }
        )
        explicitly_consumed_role_names.add(event["role_name"])

    coarse = []
    protected_untouched: dict[str, list[str]] = defaultdict(list)
    for envelope in sorted(evidence, key=lambda item: item["evidence_id"]):
        for role, state in sorted(envelope["role_use"].items()):
            row = {
                "evidence_id": envelope["evidence_id"],
                "role_name": role,
                "state": state,
                "semantics": "coarse_historical",
                "role_identity": None,
            }
            coarse.append(row)
            if (
                role in {"official_validation", "test_dev", "test_challenge"}
                and state == "untouched"
                and role not in explicitly_consumed_role_names
            ):
                protected_untouched[role].append(envelope["evidence_id"])

    reused = {
        identity: rows
        for identity, rows in sorted(identities.items())
        if len({row["trajectory_id"] for row in rows}) > 1
    }
    repeatedly_selected = {
        identity: rows
        for identity, rows in sorted(identities.items())
        if sum(bool(row["selection_used"]) for row in rows) > 1
    }
    return {
        "format": "molgap-rml-role-reuse-index-v1",
        "explicit_role_event_count": len(role_events),
        "identities": {key: value for key, value in sorted(identities.items())},
        "reused_role_identities": reused,
        "repeatedly_selected_role_identities": repeatedly_selected,
        "coarse_historical_role_records": coarse,
        "ambiguous_historical_role_record_count": len(coarse),
        "protected_roles_reported_untouched": {
            key: sorted(value) for key, value in sorted(protected_untouched.items())
        },
        "full_training_membership_conflicts": sorted(
            role
            for role in explicitly_consumed_role_names
            if any(
                row["role_name"] == role and row["state"] == "untouched"
                for row in coarse
            )
        ),
    }
