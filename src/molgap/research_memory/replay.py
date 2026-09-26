"""Build replay worlds without weakening the existing strict grouping gate."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from molgap.evidence_pointers import load_json_object
from .paths import repo_local_path, resolve_repo_pointer, verify_bound_artifact
from .backtest import _comparison_key, build_screening_backtest
from .trace import load_canonical_trace, json_bytes, trace_digest, validate_manifest_trace
from .schemas import validate_trace_manifest
from .paired import accepted_reference_evidence, pair_binding


def _terminal_label(trajectory: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any] | None:
    """Only a frozen contract result supplies a screen label; never scale truth."""
    outcome = trajectory["decision"]["outcome"]
    scientific = evidence["outcome"]["scientific_status"]
    if outcome == "POSITIVE_UNDER_CONTRACT" and scientific == "POSITIVE_UNDER_CONTRACT":
        return {"winner": True, "promotion_passed": True}
    if outcome == "NEGATIVE_UNDER_CONTRACT" and scientific == "NEGATIVE_UNDER_CONTRACT":
        return {"winner": False, "promotion_passed": False}
    return None


def _reference_binding_is_valid(
    manifest: dict[str, Any], trajectory: dict[str, Any], root: Path | None = None
) -> bool:
    """Bind candidates prospectively and references to their own accepted identity.

    A candidate must name the reference in its state-at-start snapshot.  A
    canonical reference trajectory cannot prospectively reference itself; its
    equivalent immutable binding is the accepted evidence identity frozen in
    its terminal result.  Keeping these cases separate prevents this historical
    recovery rule from weakening the candidate release gate.
    """
    paired = pair_binding(trajectory)
    if paired is not None and manifest["backtest_eligibility"]["eligible"]:
        if root is None or manifest["comparison_role"] != paired["comparison_role"]:
            return False
        try:
            accepted_reference_id, _ = accepted_reference_evidence(root, trajectory)
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return manifest["reference_id"] == accepted_reference_id
    if manifest["comparison_role"] == "candidate":
        return manifest["reference_id"] in trajectory["state_at_start"]["reference_ids"]
    if manifest["comparison_role"] == "reference":
        return manifest["reference_id"] in trajectory["result"]["evidence_ids"]
    return False


def _canonical_reference_manifest(root: Path, path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Use a hash-bound migration sidecar without rewriting accepted V5 bytes."""
    binding_path = repo_local_path(root, path.parent / "trace_migration.json")
    if manifest["comparison_role"] != "reference" or not binding_path.is_file():
        return manifest
    binding = load_json_object(binding_path)
    if binding.get("format") != "molgap-rml-reference-trace-migration-v1":
        raise ValueError("unsupported reference trace migration")
    if resolve_repo_pointer(root, binding["original_manifest_ref"]) != path.resolve():
        raise ValueError("migration does not bind its original manifest")
    verify_bound_artifact(root, binding["original_manifest_ref"], binding["original_manifest_sha256"])
    verify_bound_artifact(root, binding["canonical_manifest_ref"], binding["canonical_manifest_sha256"])
    migrated = validate_trace_manifest(load_json_object(resolve_repo_pointer(root, binding["canonical_manifest_ref"])))
    mutable = {"trace_artifact_ref", "trace_artifact_sha256", "trace_fields"}
    if {k: v for k, v in manifest.items() if k not in mutable} != {
        k: v for k, v in migrated.items() if k not in mutable
    }:
        raise ValueError("canonical migration changed frozen reference semantics")
    evidence = load_json_object(resolve_repo_pointer(root, manifest["terminal_evidence_ref"]))
    if evidence["evidence_id"] != manifest["reference_id"] or not any(
        a["locator"] == binding["source_trace_ref"] == manifest["trace_artifact_ref"]
        and a.get("sha256") == binding["source_trace_sha256"] for a in evidence["artifacts"]
    ):
        raise ValueError("migration source is not bound by frozen reference evidence")
    verify_bound_artifact(root, migrated["trace_artifact_ref"], migrated["trace_artifact_sha256"])
    canonical = load_canonical_trace(resolve_repo_pointer(root, migrated["trace_artifact_ref"]))
    validate_manifest_trace(migrated, canonical)
    if not any(p.get("source") == binding["source_trace_ref"] and
               p.get("sha256") == binding["source_trace_sha256"]
               for p in canonical.get("provenance", [])):
        raise ValueError("canonical trace lost its original source provenance")
    return migrated


def build_replay_pool(root: Path, records: dict[str, list]) -> dict[str, Any]:
    trajectories = {t["trajectory_id"]: t for _, t in records["trajectories"]}
    evidence_by_path = {p.resolve(): e for p, e in records["evidence"]}
    manifests = [_canonical_reference_manifest(root, p, m) for p, m in records["traces"]]
    grouping = build_screening_backtest(manifests)
    keys = {g["comparability_key"] for g in grouping["included_comparable_groups"]}
    excluded = list(grouping["excluded_traces"])
    eligible = []
    from collections import Counter
    identities = Counter((m["trajectory_id"], m["run_id"]) for m in manifests)
    for manifest in sorted(manifests, key=lambda m: (m["trajectory_id"], m["run_id"])):
        key = _comparison_key(manifest)
        identity = (manifest["trajectory_id"], manifest["run_id"])
        trajectory = trajectories[manifest["trajectory_id"]]
        if not _reference_binding_is_valid(manifest, trajectory, root):
            excluded.append({"trajectory_id": identity[0], "run_id": identity[1],
                             "reasons": ["reference_not_frozen_in_trajectory"]})
            continue
        if identities[identity] > 1:
            excluded.append({"trajectory_id": identity[0], "run_id": identity[1],
                             "reasons": ["ambiguous_duplicate_replay_run_identity"]})
            continue
        if key not in keys:
            continue
        reasons = []
        if not manifest["backtest_eligibility"]["eligible"] or manifest["backtest_eligibility"]["exclusion_reasons"]:
            continue
        artifact = resolve_repo_pointer(root, manifest["trace_artifact_ref"])
        canonical = None
        if artifact is None:
            reasons.append("canonical_trace_not_locally_retained")
        elif manifest.get("trace_artifact_sha256") is None:
            reasons.append("canonical_trace_hash_unbound")
        else:
            # New bound artifacts are strict: corruption is an error, not a
            # historical capability exclusion that could hide tampering.
            canonical = load_canonical_trace(artifact)
            if trace_digest(artifact) != manifest["trace_artifact_sha256"]:
                raise ValueError("canonical trace digest mismatch")
            if (canonical["trajectory_id"], canonical["run_id"]) != identity:
                raise ValueError("canonical trace identity mismatch")
        evidence_path = resolve_repo_pointer(root, manifest["terminal_evidence_ref"])
        evidence = evidence_by_path.get(evidence_path)
        if evidence is None or evidence["evidence_id"] not in trajectory["result"]["evidence_ids"]:
            reasons.append("terminal_evidence_not_bound_to_trajectory")
        if manifest["comparison_role"] == "reference" and (
            evidence is None or evidence["evidence_id"] != manifest["reference_id"]
            or evidence_path not in {
                resolve_repo_pointer(root, ref) for ref in trajectory["result"]["evidence_refs"]
            }
        ):
            reasons.append("reference_terminal_evidence_identity_mismatch")
        if trajectory["decision"]["outcome"] == "ACTIVE":
            reasons.append("trajectory_not_terminal")
        if reasons:
            excluded.append({"trajectory_id": identity[0], "run_id": identity[1], "reasons": reasons})
            continue
        rows = canonical["observations"]
        axis = {"optimizer_steps": "optimizer_step", "presentations": "sample_presentations"}.get(manifest["x_axis"])
        observations = [row for row in rows if axis and row[axis] is not None]
        if not observations:
            excluded.append({"trajectory_id": identity[0], "run_id": identity[1],
                             "reasons": ["no_observed_cross_run_axis"]})
            continue
        expected = manifest["exposure"]["optimizer_steps" if axis == "optimizer_step" else "sample_presentations"]
        if observations[-1][axis] > expected:
            raise ValueError("trace exceeds manifest terminal exposure")
        costs = [c for _, c in records["costs"] if c["trajectory_id"] == identity[0] and c["run_id"] == identity[1]]
        hardware = sorted({c["hardware"] for c in costs})
        measurement_complete = bool(costs) and all(c["measurement"]["device_hours"]["status"] == "measured" for c in costs)
        native_cost = sum(c["measurement"]["device_hours"]["value"] for c in costs) if measurement_complete and len(hardware) == 1 else None
        eligible.append({
            "trajectory_id": identity[0], "run_id": identity[1], "family_id": trajectory["family_id"],
            "comparability_key": key, "comparability_identity": manifest["comparability_identity"],
            "reference_id": manifest["reference_id"], "comparison_role": manifest["comparison_role"],
            "model_identity": manifest["model_identity"], "axis": axis,
            "prefix_observations": observations, "metric_semantics": canonical["metric_semantics"],
            "terminal_endpoint": expected, "terminal_outcome": trajectory["decision"]["outcome"],
            "terminal_label": _terminal_label(trajectory, evidence),
            "native_cost": {"device_hours": native_cost, "hardware": hardware,
                            "measurements": [{"cost_event_id": c["cost_event_id"],
                                              **c["measurement"]["device_hours"]}
                                             for c in sorted(costs, key=lambda c: c["cost_event_id"])],
                            "event_ids": sorted(c["cost_event_id"] for c in costs)},
            "policy_version_at_execution": {k: trajectory.get("decision_state", {}).get(k)
                                            for k in ("policy_id", "policy_version")},
            "decision_state": trajectory.get("decision_state"),
            "evidence_ids": [evidence["evidence_id"]], "exclusion_reasons": [],
            "record_mode": trajectory["record_mode"],
            "capability": ("historical_partial" if trajectory["record_mode"] == "retrospective_partial"
                           else "complete" if observations[-1][axis] == expected else "partial"),
            "trace_sha256": manifest["trace_artifact_sha256"],
            "terminal_evidence_sha256": hashlib.sha256(json_bytes(evidence)).hexdigest(),
            "action_replay": _action_replay(root, trajectory),
        })
    # A reference with metadata but no canonical observations cannot establish a
    # replay world. Regroup after capability filtering, retaining strict keys.
    valid_keys = {key for key in keys if {e["comparison_role"] for e in eligible if e["comparability_key"] == key} == {"candidate", "reference"}}
    entries = []
    for entry in eligible:
        if entry["comparability_key"] not in valid_keys:
            excluded.append({"trajectory_id": entry["trajectory_id"], "run_id": entry["run_id"],
                             "reasons": ["no_canonical_candidate_reference_pair"]})
        else:
            entries.append(entry)
    action_entries = []
    for trajectory in sorted(trajectories.values(), key=lambda t: t["trajectory_id"]):
        action = _action_replay(root, trajectory)
        if action is None:
            continue
        inputs = action["inputs"]
        if trajectory["decision"]["outcome"] in {"ACTIVE", "INFRASTRUCTURE_ONLY", "NO_TRAIN", "STOP_FOR_COST", "INCONCLUSIVE"}:
            continue
        action_entries.append({
            "trajectory_id": trajectory["trajectory_id"], "run_id": inputs["run_id"],
            "family_id": trajectory["family_id"], "comparison_role": "candidate",
            "reference_id": inputs["reference_id"],
            "comparability_identity": inputs["comparability_identity"],
            "comparability_key": _comparison_key(inputs),
            "decision_state": trajectory["decision_state"], "action_replay": action,
            "native_cost": {"hardware": inputs.get("hardware", []), "device_hours": None},
        })
    return {"format": "molgap-rml-replay-pool-v1", "entries": entries, "action_entries": action_entries,
            "comparable_group_count": len(valid_keys),
            "exclusions": sorted(excluded, key=lambda x: (x["trajectory_id"], x["run_id"])),
            "source_digest": hashlib.sha256(json_bytes({"entries": entries, "action_entries": action_entries, "exclusions": excluded})).hexdigest()}


def _action_replay(root: Path, trajectory: dict[str, Any]) -> dict[str, Any] | None:
    """Optional frozen action evidence for scale/research policies, never inferred.

    Decision-time inputs and later truth are separate hash-bound documents.
    The snapshot names only IDs known in the prospective decision_state.
    """
    binding = trajectory.get("action_replay")
    if binding is None:
        return None
    result = {}
    for kind in ("inputs", "truth"):
        verify_bound_artifact(root, binding[kind + "_ref"], binding[kind + "_sha256"])
        result[kind] = load_json_object(resolve_repo_pointer(root, binding[kind + "_ref"]))
    state = trajectory.get("decision_state")
    if state is None:
        raise ValueError("action replay requires a frozen decision state")
    for kind, document in result.items():
        if document.get("trajectory_id") != trajectory["trajectory_id"]:
            raise ValueError("action replay trajectory mismatch")
    if not set(result["inputs"]["evidence_ids"]) <= set(state["known_evidence_ids"]):
        raise ValueError("action inputs contain future evidence")
    if result["inputs"].get("state_timestamp") != state["state_timestamp"]:
        raise ValueError("action inputs do not belong to frozen decision time")
    if state.get("action_inputs_sha256") != binding["inputs_sha256"]:
        raise ValueError("action input digest was not frozen prospectively")
    if not set(result["truth"]["evidence_ids"]) <= set(trajectory["result"]["evidence_ids"]):
        raise ValueError("action truth evidence is not a frozen trajectory result")
    if result["truth"].get("policy_type") != result["inputs"].get("policy_type"):
        raise ValueError("action truth/input policy types differ")
    result["bindings"] = binding
    return result
