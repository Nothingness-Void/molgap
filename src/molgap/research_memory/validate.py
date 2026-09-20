"""RML validation and repository-pointer safety."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from molgap.evidence_pointers import (
    load_json_object as load_json,
)
from .paths import resolve_repo_pointer, verify_bound_artifact

from molgap.v5_common import (
    reference_bundle_digest,
    validate_comparison_prelaunch,
    validate_comparison_readiness,
    validate_reference_bundle,
    validate_server_comparison_prelaunch,
    validate_target_transform_asset,
    validate_v5_evidence_envelope,
)

from .discovery import DiscoveredRecords, discover_records
from .schemas import (
    validate_id,
    validate_cost_event,
    validate_ready_package,
    validate_role_event,
    validate_trace_manifest,
    validate_trajectory,
)


def _validate_pointers(root: Path, pointers: Iterable[str]) -> None:
    for pointer in pointers:
        if not isinstance(pointer, str) or not pointer.strip():
            raise ValueError("record pointer must be non-empty text")
        resolve_repo_pointer(root, pointer)


def _trajectory_pointers(record: Mapping[str, Any]) -> list[str]:
    state = record["state_at_start"]
    pointers = [*state["contract_refs"], *state["role_snapshot_refs"]]
    budget_ref = state.get("budget_snapshot_ref")
    if isinstance(budget_ref, str) and budget_ref:
        pointers.append(budget_ref)
    for action in record["actions"]:
        pointers.extend(action["evidence_refs"])
    pointers.extend(record["result"]["evidence_refs"])
    pointers.append(record["decision"]["decision_ref"])
    readiness = record.get("readiness")
    if isinstance(readiness, Mapping):
        comparison_ref = readiness.get("paired_comparison_ref")
        if isinstance(comparison_ref, str) and comparison_ref:
            pointers.append(comparison_ref)
        role_refs = readiness.get("role_history_refs")
        if isinstance(role_refs, list):
            pointers.extend(role_refs)
    comparison_readiness_ref = record.get("comparison_readiness_ref")
    if isinstance(comparison_readiness_ref, str) and comparison_readiness_ref:
        pointers.append(comparison_readiness_ref)
    return pointers


def validate_repository_records(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    discovered = discover_records(root)
    records: dict[str, list[tuple[Path, dict[str, Any]]]] = {
        "evidence": [],
        "trajectories": [],
        "costs": [],
        "roles": [],
        "traces": [],
        "ready": [],
        "comparison_prelaunch": [],
        "comparison_readiness": [],
        "reference_bundles": [],
        "target_transform_assets": [],
    }
    ids: dict[str, dict[str, Path]] = {
        "evidence_id": {},
        "trajectory_id": {},
        "hypothesis_id": {},
        "cost_event_id": {},
        "role_event_id": {},
        "package_id": {},
        "reference_bundle_id": {},
        "target_transform_asset_id": {},
    }

    def unique(kind: str, value: str, path: Path) -> None:
        prior = ids[kind].get(value)
        if prior is not None:
            raise ValueError(f"duplicate {kind} {value}: {prior} and {path}")
        ids[kind][value] = path

    for path in discovered.evidence:
        record = load_json(path)
        validate_v5_evidence_envelope(record, repo_root=root)
        validate_id(record["evidence_id"], f"{path}:evidence_id")
        unique("evidence_id", record["evidence_id"], path)
        records["evidence"].append((path, record))
    for path in discovered.trajectories:
        record = validate_trajectory(load_json(path))
        unique("trajectory_id", record["trajectory_id"], path)
        unique("hypothesis_id", record["hypothesis"]["hypothesis_id"], path)
        _validate_pointers(root, _trajectory_pointers(record))
        records["trajectories"].append((path, record))
    for path in discovered.costs:
        record = validate_cost_event(load_json(path))
        unique("cost_event_id", record["cost_event_id"], path)
        _validate_pointers(root, [record["evidence_ref"]])
        records["costs"].append((path, record))
    for path in discovered.roles:
        record = validate_role_event(load_json(path))
        unique("role_event_id", record["role_event_id"], path)
        _validate_pointers(root, [record["evidence_ref"]])
        records["roles"].append((path, record))
    for path in discovered.traces:
        record = validate_trace_manifest(load_json(path))
        _validate_pointers(
            root,
            [
                record["contract_ref"],
                record["presentation_semantics_ref"],
                record["trace_artifact_ref"],
                record["terminal_evidence_ref"],
            ],
        )
        records["traces"].append((path, record))
        if "trace_artifact_sha256" in record:
            from .trace import load_canonical_trace, validate_manifest_trace
            verify_bound_artifact(root, record["trace_artifact_ref"], record["trace_artifact_sha256"])
            validate_manifest_trace(record, load_canonical_trace(resolve_repo_pointer(root, record["trace_artifact_ref"])))
    for path in discovered.ready:
        record = validate_ready_package(load_json(path))
        unique("package_id", record["package_id"], path)
        _validate_pointers(
            root,
            [
                *record["contract_refs"],
                *record["comparison_refs"],
                *record["role_history_refs"],
                record["cost_summary_ref"],
                *record["artifact_refs"],
            ],
        )
        records["ready"].append((path, record))
    for path in discovered.comparison_prelaunch:
        record = validate_comparison_prelaunch(load_json(path))
        records["comparison_prelaunch"].append((path, record))
    for path in discovered.comparison_readiness:
        record = validate_comparison_readiness(
            load_json(path),
            evidence_verifier=lambda pointer, digest: verify_bound_artifact(
                root, pointer, digest
            ),
        )
        records["comparison_readiness"].append((path, record))
    for path in discovered.target_transform_assets:
        record = validate_target_transform_asset(load_json(path))
        unique("target_transform_asset_id", record["asset_id"], path)
        records["target_transform_assets"].append((path, record))
    for path in discovered.reference_bundles:
        record = validate_reference_bundle(load_json(path))
        unique("reference_bundle_id", record["reference_bundle_id"], path)
        _validate_pointers(
            root,
            [
                record["contract_ref"],
                record["runtime_certificate_ref"],
                record["row_manifest_ref"],
                record["target_manifest_ref"],
                record["trace_manifest_ref"],
                record["role_history_ref"],
                record["target_transform_asset_ref"],
                record["cost_records_ref"],
                record["acceptance_ref"],
                record["decision_ref"],
            ],
        )
        records["reference_bundles"].append((path, record))

    target_transform_by_path = {
        path.resolve(): record for path, record in records["target_transform_assets"]
    }
    for path, record in records["reference_bundles"]:
        transform_path = resolve_repo_pointer(root, record["target_transform_asset_ref"])
        if transform_path is None or transform_path not in target_transform_by_path:
            raise ValueError(
                f"{path}:target_transform_asset_ref must point to a discovered "
                "experiments/**/target_transform.json record"
            )
        transform = target_transform_by_path[transform_path]
        identity = record["comparison_identity"]
        if identity["target_transform_identity"] != transform["asset_id"]:
            raise ValueError(f"{path}:target-transform identity does not match asset_id")
        if identity["target_transform_asset_sha256"] != transform["asset_sha256"]:
            raise ValueError(f"{path}:target-transform SHA does not match validated asset")
        if identity["target_identity"] != transform["target_identity"]:
            raise ValueError(f"{path}:target identity does not match transform asset")

    trajectory_by_id = {
        record["trajectory_id"]: record for _, record in records["trajectories"]
    }
    evidence_ids = {record["evidence_id"] for _, record in records["evidence"]}
    action_ids = {
        record["trajectory_id"]: {action["action_id"] for action in record["actions"]}
        for _, record in records["trajectories"]
    }
    def require_ids(path: Path, field: str, values: Iterable[str], known: set[str]) -> None:
        missing = sorted(set(values) - known)
        if missing:
            raise ValueError(f"{path}:{field} contains missing IDs: {missing}")

    cost_ids = {record["cost_event_id"] for _, record in records["costs"]}
    reference_bundles_by_id = {
        record["reference_bundle_id"]: record
        for _, record in records["reference_bundles"]
    }
    reference_bundle_ids = set(reference_bundles_by_id)
    for path, record in records["comparison_prelaunch"]:
        reference_bundle_id = record.get("reference_bundle_id")
        if (
            reference_bundle_id is not None
            and reference_bundle_id not in reference_bundle_ids
        ):
            raise ValueError(
                f"{path}:reference_bundle_id references missing ID: "
                f"{reference_bundle_id}"
            )
        validate_server_comparison_prelaunch(
            record,
            experiment_purpose=record["experiment_purpose"],
            reference_bundle=(
                reference_bundles_by_id[reference_bundle_id]
                if reference_bundle_id is not None
                else None
            ),
        )
    for path, record in records["comparison_readiness"]:
        if not record["strict_ready"]:
            continue
        reference_bundle_id = record.get("reference_bundle_id")
        if reference_bundle_id not in reference_bundles_by_id:
            raise ValueError(
                f"{path}:strict readiness references missing bundle ID: "
                f"{reference_bundle_id}"
            )
        actual_bundle = reference_bundles_by_id[reference_bundle_id]
        if record.get("reference_id") != actual_bundle["reference_id"]:
            raise ValueError(f"{path}:strict readiness reference identity differs from bundle")
        if record.get("reference_bundle_sha256") != reference_bundle_digest(actual_bundle):
            raise ValueError(f"{path}:strict readiness reference bundle SHA differs")
        bundle_binding_refs = {
            "runtime_certificate": "runtime_certificate_ref",
            "row_manifest": "row_manifest_ref",
            "target_manifest": "target_manifest_ref",
            "trace_manifest": "trace_manifest_ref",
            "role_history": "role_history_ref",
            "target_transform_asset": "target_transform_asset_ref",
            "cost_records": "cost_records_ref",
            "acceptance": "acceptance_ref",
            "decision": "decision_ref",
        }
        for artifact, bundle_field in bundle_binding_refs.items():
            if record["reference_artifact_bindings"][artifact]["ref"] != actual_bundle[
                bundle_field
            ]:
                raise ValueError(
                    f"{path}:strict readiness {artifact} binding differs from bundle"
                )
    for path, record in records["trajectories"]:
        missing_parents = sorted(
            set(record["state_at_start"]["parent_trajectory_ids"]) - set(trajectory_by_id)
        )
        if missing_parents:
            raise ValueError(
                f"{path}:state_at_start.parent_trajectory_ids contains missing IDs: "
                f"{missing_parents}"
            )
        require_ids(
            path,
            "state_at_start.prior_trajectory_ids",
            record["state_at_start"].get("prior_trajectory_ids", []),
            set(trajectory_by_id),
        )
        require_ids(
            path,
            "hypothesis.supporting_evidence_ids",
            record["hypothesis"]["supporting_evidence_ids"],
            evidence_ids,
        )
        require_ids(
            path,
            "state_at_start.prior_evidence_ids",
            record["state_at_start"]["prior_evidence_ids"],
            evidence_ids,
        )
        require_ids(
            path,
            "state_at_start.reference_ids",
            record["state_at_start"]["reference_ids"],
            evidence_ids,
        )
        require_ids(
            path,
            "result.evidence_ids",
            record["result"]["evidence_ids"],
            evidence_ids,
        )
        for action in record["actions"]:
            require_ids(
                path,
                f"actions.{action['action_id']}.cost_event_ids",
                action["cost_event_ids"],
                cost_ids,
            )
        readiness = record.get("readiness")
        if isinstance(readiness, Mapping):
            require_ids(
                path,
                "readiness.candidate_evidence_id",
                [readiness["candidate_evidence_id"]],
                evidence_ids,
            )
            require_ids(
                path,
                "readiness.compatible_reference_id",
                [readiness["compatible_reference_id"]],
                evidence_ids,
            )
            require_ids(
                path,
                "readiness.qualification_100k_evidence_ids",
                readiness["qualification_100k_evidence_ids"],
                evidence_ids,
            )
            require_ids(
                path,
                "readiness.qualification_500k_evidence_ids",
                readiness["qualification_500k_evidence_ids"],
                evidence_ids,
            )
            require_ids(
                path,
                "readiness.native_cost_event_ids",
                readiness["native_cost_event_ids"],
                cost_ids,
            )
        comparison_ref = record.get("comparison_readiness_ref")
        if isinstance(comparison_ref, str):
            resolve_repo_pointer(root, comparison_ref)
        reference_bundle_id = record.get("reference_bundle_id")
        if reference_bundle_id is not None and reference_bundle_id not in reference_bundle_ids:
            raise ValueError(
                f"{path}:reference_bundle_id references missing ID: {reference_bundle_id}"
            )
        if record["record_mode"] == "prospective":
            expected_cost = record["hypothesis"]["expected_native_cost_ref"]
            validate_id(expected_cost, f"{path}:hypothesis.expected_native_cost_ref")
            require_ids(
                path,
                "hypothesis.expected_native_cost_ref",
                [expected_cost],
                cost_ids,
            )
    for kind in ("costs", "roles"):
        for path, record in records[kind]:
            trajectory_id = record["trajectory_id"]
            if trajectory_id not in trajectory_by_id:
                raise ValueError(f"{path}:trajectory_id references missing ID: {trajectory_id}")
            if record["action_id"] not in action_ids[trajectory_id]:
                raise ValueError(
                    f"{path}:action_id references missing action: "
                    f"{trajectory_id}:{record['action_id']}"
                )
    for path, record in records["traces"]:
        if record["trajectory_id"] not in trajectory_by_id:
            raise ValueError(
                f"{path}:trajectory_id references missing ID: {record['trajectory_id']}"
            )
        if record["reference_id"] not in evidence_ids:
            raise ValueError(
                f"{path}:reference_id references missing ID: {record['reference_id']}"
            )
    for path, record in records["ready"]:
        trajectory_id = record["trajectory_id"]
        if trajectory_id not in trajectory_by_id:
            raise ValueError(
                f"{path}:trajectory_id references missing ID: {trajectory_id}"
            )
        trajectory = trajectory_by_id[trajectory_id]
        if trajectory["record_mode"] != "prospective":
            raise ValueError(f"{path}:trajectory_id must reference a prospective trajectory")
        if trajectory["owner"] != "server":
            raise ValueError(f"{path}:trajectory_id must reference a server-owned trajectory")
        if trajectory["decision"]["outcome"] != "READY_FOR_DESKTOP":
            raise ValueError(f"{path}:trajectory decision is not READY_FOR_DESKTOP")
        if trajectory["decision"].get("final") is not True:
            raise ValueError(f"{path}:trajectory decision is not final")
        if record["candidate_evidence_id"] not in evidence_ids:
            raise ValueError(
                f"{path}:candidate_evidence_id references missing ID: "
                f"{record['candidate_evidence_id']}"
            )
        require_ids(path, "reference_ids", record["reference_ids"], evidence_ids)
        require_ids(
            path,
            "qualification_100k_evidence_ids",
            record["qualification_100k_evidence_ids"],
            evidence_ids,
        )
        require_ids(
            path,
            "qualification_500k_evidence_ids",
            record["qualification_500k_evidence_ids"],
            evidence_ids,
        )
        require_ids(
            path,
            "native_cost_event_ids",
            record["native_cost_event_ids"],
            cost_ids,
        )
        readiness = trajectory.get("readiness")
        if not isinstance(readiness, Mapping):
            raise ValueError(f"{path}:trajectory has no readiness block")
        exact_fields = {
            "candidate_evidence_id": "candidate_evidence_id",
            "candidate_contract_identity": "candidate_contract_identity",
            "reference_contract_identity": "reference_contract_identity",
            "source_config_identity": "source_config_identity",
            "funnel_requires_500k": "funnel_requires_500k",
        }
        for package_field, readiness_field in exact_fields.items():
            if record[package_field] != readiness.get(readiness_field):
                raise ValueError(
                    f"{path}:{package_field} does not match trajectory readiness"
                )
        list_fields = (
            "qualification_100k_evidence_ids",
            "qualification_500k_evidence_ids",
            "role_history_refs",
            "native_cost_event_ids",
        )
        for field in list_fields:
            if sorted(record[field]) != sorted(readiness.get(field, [])):
                raise ValueError(f"{path}:{field} does not match trajectory readiness")
        if record["reference_ids"] != [readiness.get("compatible_reference_id")]:
            raise ValueError(f"{path}:reference_ids do not match trajectory readiness")
        if record["comparison_refs"] != [readiness.get("paired_comparison_ref")]:
            raise ValueError(f"{path}:comparison_refs do not match trajectory readiness")
        candidate = next(
            value for _, value in records["evidence"]
            if value["evidence_id"] == record["candidate_evidence_id"]
        )
        reference = next(
            value for _, value in records["evidence"]
            if value["evidence_id"] == record["reference_ids"][0]
        )
        if candidate.get("legacy_contract") != record["candidate_contract_identity"]:
            raise ValueError(f"{path}:candidate contract identity mismatch")
        if reference.get("legacy_contract") != record["reference_contract_identity"]:
            raise ValueError(f"{path}:reference contract identity mismatch")
        cost_by_id = {
            value["cost_event_id"]: value for _, value in records["costs"]
        }
        if not any(
            cost_by_id[cost_id]["measurement"][unit]["status"] == "measured"
            for cost_id in record["native_cost_event_ids"]
            for unit in ("device_hours", "cpu_hours")
        ):
            raise ValueError(f"{path}:native cost has no measured device or CPU hours")
    return {"root": root, "discovered": discovered, "records": records}
