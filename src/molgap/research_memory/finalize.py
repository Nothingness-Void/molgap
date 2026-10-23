"""Fail-closed terminal metadata ingestion; no model or platform imports.

A terminal.json package supplies already accepted V5 evidence, explicit observed
cost/role events, a terminal decision and repository-local SHA256 bindings.
One directory rename publishes all records. Discovery overlays only receipts
whose complete contents still match their publication hashes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from molgap.evidence_pointers import load_json_object
from .paths import bound_artifact_path, repo_local_path, resolve_repo_pointer, verify_bound_artifact
from .paired import (accepted_reference_evidence, pair_binding, reference_trajectory,
                     terminal_reference_evidence, validate_pair_observation,
                     validate_terminal_continuation)
from .roles import validate_observed_role_truth
from molgap.v5_common import validate_v5_evidence_envelope
from .schemas import validate_cost_event, validate_role_event, validate_trace_manifest, validate_trajectory
from .trace import atomic_write, file_digest, json_bytes, load_canonical_trace, sync_directory, validate_manifest_trace

FINALIZER_VERSION = "2"


def _local(root: Path, pointer: str) -> Path:
    path = resolve_repo_pointer(root, pointer)
    if path is None:
        raise ValueError(f"finalize requires retained local metadata: {pointer}")
    return path


def _bindings(root: Path, bindings: dict[str, str]) -> None:
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("terminal package requires explicit artifact_hashes")
    for pointer, digest in bindings.items():
        verify_bound_artifact(root, pointer, digest)


def prospective_bytes(directory: Path, receipt: dict[str, Any] | None = None) -> bytes:
    receipt = receipt or load_json_object(directory / "finalization.json")
    if receipt.get("format") == "molgap-rml-finalization-v1":
        return (directory / "prospective_snapshot.json").read_bytes()
    # The original prospective input remains canonical; bind it instead of copying it.
    root = next(parent for parent in directory.parents if (parent / "experiments").is_dir())
    path = repo_local_path(root, receipt["prospective_ref"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["prospective_sha256"]:
        raise ValueError("prospective input changed after finalization")
    return raw


def load_terminal_input(directory: Path) -> dict[str, Any]:
    """Read legacy embedded inputs or v2 pointers to the same published records."""
    package = load_json_object(directory / "terminal_input.json")
    if package.get("format") != "molgap-rml-terminal-input-v2":
        return package
    bindings = package.pop("published_components")
    package["format"] = "molgap-rml-terminal-package-v1"
    for key, entries in bindings.items():
        if key not in {"evidence", "decision", "costs", "roles"}:
            raise ValueError("unsupported terminal component")
        values = []
        for binding in entries:
            path = repo_local_path(directory, binding["ref"])
            if file_digest(path) != binding["sha256"]:
                raise ValueError("terminal component hash mismatch")
            value = load_json_object(path)
            if key == "decision":
                value = value["decision"]
            values.append(value)
        if key in {"evidence", "decision"} and len(values) != 1:
            raise ValueError("terminal input requires one evidence and decision")
        package[key] = values[0] if key in {"evidence", "decision"} else values
    if not {"evidence", "decision"} <= package.keys():
        raise ValueError("incomplete terminal components")
    return package


def verified_receipt(directory: Path) -> dict[str, Any]:
    receipt = load_json_object(repo_local_path(directory, "finalization.json"))
    if receipt.get("format") not in {"molgap-rml-finalization-v1", "molgap-rml-finalization-v2"}:
        raise ValueError("unsupported finalization receipt")
    actual_files = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual_files != set(receipt["published_hashes"]) | {"finalization.json"}:
        raise ValueError("finalized directory inventory differs from receipt")
    if not set(receipt["replacements"]) <= set(receipt["published_hashes"]):
        raise ValueError("replacement is not a published record")
    for relative, digest in receipt["published_hashes"].items():
        path = repo_local_path(directory, relative)
        if file_digest(path) != digest:
            raise ValueError(f"finalized package changed: {relative}")
    required = {"trajectory.json", "v5_evidence.json", "terminal_input.json"}
    if receipt["format"] == "molgap-rml-finalization-v1":
        required.add("prospective_snapshot.json")
    if not required <= set(receipt["published_hashes"]):
        raise ValueError("incomplete finalization publication")
    source_digest = hashlib.sha256(json_bytes({
        "trajectory_sha256": hashlib.sha256(prospective_bytes(directory, receipt)).hexdigest(),
        "terminal": load_terminal_input(directory),
        "trace_sha256": file_digest(directory / "trace.json") if "trace.json" in receipt["published_hashes"] else None,
    })).hexdigest()
    if source_digest != receipt["source_artifact_digest"]:
        raise ValueError("finalization source digest mismatch")
    return receipt


def _qualification_amendment(root: Path, previous: Path, frozen_bytes: bytes,
                             package: dict, trace_bytes: bytes | None) -> None:
    """Requalification may add proof, never revise the accepted scientific result."""
    receipt = verified_receipt(previous)
    old = load_terminal_input(previous)
    if prospective_bytes(previous, receipt) != frozen_bytes:
        raise ValueError("qualification revision changed the original prospective bytes")
    for field in ("trajectory_id", "run_id", "action_id", "decision",
                  "acceptance_ref", "roles", "costs", "role_use"):
        if package.get(field) != old.get(field):
            raise ValueError("qualification revision changed accepted " + field)
    for field in (old["evidence"].keys() | package["evidence"].keys()) - {"artifacts"}:
        if package["evidence"].get(field) != old["evidence"].get(field):
            raise ValueError("qualification revision changed accepted evidence " + field)
    for field in ("artifacts",):
        if any(item not in package["evidence"].get(field, [])
               for item in old["evidence"].get(field, [])):
            raise ValueError("qualification revision removed accepted evidence artifacts")
    if any(package["artifact_hashes"].get(pointer) != digest
           for pointer, digest in old["artifact_hashes"].items()):
        raise ValueError("qualification revision changed accepted artifact bindings")
    old_manifest, new_manifest = old.get("trace_manifest"), package.get("trace_manifest")
    if old_manifest is not None or new_manifest is not None:
        if not isinstance(old_manifest, dict) or not isinstance(new_manifest, dict) or any(
                old_manifest.get(field) != new_manifest.get(field)
                for field in (old_manifest.keys() | new_manifest.keys()) - {"backtest_eligibility"}):
            raise ValueError("qualification revision changed accepted trace semantics")
    old_trace = (previous / "trace.json").read_bytes() if receipt["trace_status"] == "available" else None
    if trace_bytes != old_trace:
        raise ValueError("qualification revision changed accepted trace bytes")


def current_finalized_directory(repo_root: str | Path, trajectory_or_directory: str | Path) -> Path:
    """Resolve the unique append-only qualification tip; forks never pick a winner."""
    root = Path(repo_root).resolve()
    path = repo_local_path(root, trajectory_or_directory)
    if path.is_file():
        path = path.parent
    if path.name == "rml_finalized":
        receipt = verified_receipt(path)
        owner = (repo_local_path(root, receipt["prospective_ref"]).parent
                 if receipt.get("prospective_ref") else path.parent)
    else:
        owner = path
    original = repo_local_path(root, owner / "rml_finalized")
    revisions = sorted((owner / "qualification_revisions").glob("*/rml_finalized"))
    if not original.is_dir():
        if revisions:
            raise ValueError("qualification revision has no original finalization")
        return original
    initial = verified_receipt(original)
    frozen = prospective_bytes(original, initial)
    directories = {original, *revisions}
    children = {}
    for directory in revisions:
        directory = repo_local_path(root, directory)
        receipt = verified_receipt(directory)
        package = load_terminal_input(directory)
        pointer = package.get("supersedes_finalization_ref")
        if not isinstance(pointer, str) or pointer not in package.get("artifact_hashes", {}):
            raise ValueError("qualification revision requires a hash-bound superseded receipt")
        parent = repo_local_path(root, pointer)
        if parent.name != "finalization.json" or parent.parent not in directories:
            raise ValueError("qualification revision parent is outside its prospective chain")
        verify_bound_artifact(root, pointer, package["artifact_hashes"][pointer])
        if (receipt["trajectory_id"] != initial["trajectory_id"]
                or prospective_bytes(directory, receipt) != frozen):
            raise ValueError("qualification revision changed its prospective identity")
        _qualification_amendment(root, parent.parent, frozen, package,
            (directory / "trace.json").read_bytes() if receipt["trace_status"] == "available" else None)
        if parent.parent in children:
            raise ValueError("qualification finalization fork requires explicit reconciliation")
        children[parent.parent] = directory
    current, visited = original, {original}
    while current in children:
        current = children[current]
        if current in visited:
            raise ValueError("qualification finalization chain contains a cycle")
        visited.add(current)
    if visited != directories:
        raise ValueError("qualification finalization chain has an unreachable revision")
    return current


def _validate_staged(root: Path, staging: Path, destination: Path) -> None:
    """Validate the staged records and cross-links against their eventual view."""
    from .validate import _trajectory_pointers, validate_repository_records

    receipt = verified_receipt(staging)
    baseline = validate_repository_records(root, verify_comparison_artifacts=False)
    discovered = baseline["discovered"]
    replacements = set(receipt["replacements"].values())
    ids = {}
    for kind, field in (("trajectories", "trajectory_id"), ("evidence", "evidence_id"),
                         ("costs", "cost_event_id"), ("roles", "role_event_id")):
        ids[field] = {load_json_object(p)[field] for p in getattr(discovered, kind)
                      if p.relative_to(root).as_posix() not in replacements}
    validators = {"trajectory.json": (validate_trajectory, "trajectory_id"),
                  "v5_evidence.json": (validate_v5_evidence_envelope, "evidence_id"),
                  "trace_manifest.json": (validate_trace_manifest, None)}
    records = {}
    for name in receipt["published_hashes"]:
        record = load_json_object(staging / name)
        if name == "trajectory.json":
            from .plan import resolve_decision_state
            record = resolve_decision_state(root, record)
        records[name] = record
        validator, field = validators.get(name, (None, None))
        if name.startswith("costs/"):
            validator, field = validate_cost_event, "cost_event_id"
        if name.startswith("roles/"):
            validator, field = validate_role_event, "role_event_id"
        if validator:
            validator(record)
        if field:
            if record[field] in ids[field]:
                raise ValueError(f"staged package duplicates {field}: {record[field]}")
            ids[field].add(record[field])

    def pointer(value: str) -> None:
        prefix = destination.relative_to(root).as_posix() + "/"
        if value.startswith(prefix):
            relative = value[len(prefix):]
            if relative not in receipt["published_hashes"]:
                raise ValueError("staged pointer does not name a published artifact")
        else:
            resolve_repo_pointer(root, value)

    trajectory = records["trajectory.json"]
    for value in _trajectory_pointers(trajectory):
        pointer(value)
    state = trajectory["state_at_start"]
    bundle_id = trajectory.get("reference_bundle_id")
    bundles = {b["reference_bundle_id"]: b for _, b in baseline["records"]["reference_bundles"]}
    if bundle_id is not None and bundle_id not in bundles:
        raise ValueError("staged trajectory references unknown reference bundle")
    for field, values in (
        ("trajectory_id", state["parent_trajectory_ids"] + state.get("prior_trajectory_ids", [])),
        ("evidence_id", state["reference_ids"] + state["prior_evidence_ids"] +
         trajectory["hypothesis"]["supporting_evidence_ids"] + trajectory["result"]["evidence_ids"]),
        ("cost_event_id", [trajectory["hypothesis"]["expected_native_cost_ref"]] +
         [c for a in trajectory["actions"] for c in a["cost_event_ids"]]),
    ):
        if not set(values) <= ids[field]:
            raise ValueError(f"unresolved staged {field} references")
    if "trace_manifest.json" in records:
        manifest = records["trace_manifest.json"]
        validate_manifest_trace(manifest, records["trace.json"])
        for field in ("trace_artifact_ref", "terminal_evidence_ref", "contract_ref", "presentation_semantics_ref"):
            pointer(manifest[field])
        if manifest["reference_id"] is not None and manifest["reference_id"] not in ids["evidence_id"]:
            raise ValueError("unresolved trace reference")


def finalize(repo_root: str | Path, trajectory: str | Path, terminal: str | Path,
             trace: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    trajectory_path = repo_local_path(root, trajectory)
    if trajectory_path.is_dir():
        trajectory_path = repo_local_path(root, trajectory_path / "trajectory.json")
    trajectory_path.relative_to(root / "experiments")
    if "rml_finalized" in trajectory_path.relative_to(root).parts:
        raise ValueError("pass the original prospective trajectory, not its finalized overlay")
    frozen_bytes = trajectory_path.read_bytes()
    from .plan import resolve_decision_state
    frozen_raw = json.loads(frozen_bytes)
    frozen = validate_trajectory(resolve_decision_state(root, frozen_raw))
    paired = pair_binding(frozen)
    if paired is not None:
        reference_trajectory(root, frozen)
    if frozen["record_mode"] != "prospective" or frozen["owner"] not in {"desktop", "server"}:
        raise ValueError("finalize requires a desktop/server-owned prospective trajectory")
    source = repo_local_path(root, terminal)
    if source.is_dir():
        source = repo_local_path(root, source / "terminal.json")
    package = load_json_object(source)
    if package.get("format") != "molgap-rml-terminal-package-v1":
        raise ValueError("unsupported terminal package")
    if package.get("closure_blockers"):
        raise ValueError("terminal package has unresolved closure blockers")
    if package.get("trajectory_id") != frozen["trajectory_id"]:
        raise ValueError("terminal trajectory identity mismatch")
    if not isinstance(package.get("finalized_at"), str) or not package["finalized_at"]:
        raise ValueError("explicit terminal finalization timestamp required")
    trace_bytes = None
    if trace is not None:
        trace_path = bound_artifact_path(root, trace)
        canonical = load_canonical_trace(trace_path)
        trace_bytes = trace_path.read_bytes()
        if (canonical["trajectory_id"], canonical["run_id"]) != (
            frozen["trajectory_id"], package.get("run_id")
        ):
            raise ValueError("trace run identity mismatch")
    _bindings(root, package.get("artifact_hashes"))
    digest = hashlib.sha256(json_bytes({
        "trajectory_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "terminal": package,
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest() if trace_bytes is not None else None,
    })).hexdigest()
    original_destination = repo_local_path(root, trajectory_path.parent / "rml_finalized")
    supersedes = package.get("supersedes_finalization_ref")
    previous = None
    if supersedes is not None:
        if not isinstance(supersedes, str) or supersedes not in package["artifact_hashes"]:
            raise ValueError("qualification revision requires a hash-bound superseded receipt")
        old_receipt = repo_local_path(root, supersedes)
        if old_receipt.name != "finalization.json":
            raise ValueError("qualification revision must supersede an exact finalization receipt")
        previous = old_receipt.parent
        current = current_finalized_directory(root, trajectory_path)
        destination = repo_local_path(root, trajectory_path.parent /
            "qualification_revisions" / digest / "rml_finalized")
        if destination.is_dir():
            receipt = verified_receipt(destination)
            if receipt["source_artifact_digest"] != digest:
                raise ValueError("FINALIZATION_CONFLICT: immutable qualification inputs changed")
            return {**verified_receipt(current), "status": "ALREADY_FINALIZED"}
        if current != previous:
            raise ValueError("qualification revision must supersede the unique current finalization")
        _qualification_amendment(root, previous, frozen_bytes, package, trace_bytes)
    else:
        destination = original_destination
        if destination.exists():
            receipt = verified_receipt(destination)
            if receipt["source_artifact_digest"] != digest:
                raise ValueError("FINALIZATION_CONFLICT: immutable terminal inputs changed")
            return {**verified_receipt(current_finalized_directory(root, trajectory_path)),
                    "status": "ALREADY_FINALIZED"}
    if frozen["decision"]["outcome"] != "ACTIVE":
        raise ValueError("cannot finalize an already terminal trajectory")
    run_id = package["run_id"]
    actions = {row["action_id"]: row for row in frozen["actions"]}
    action_id = package["action_id"]
    if action_id not in actions or run_id not in actions[action_id]["run_ids"]:
        raise ValueError("terminal run/action was not frozen in trajectory")
    evidence = copy.deepcopy(package["evidence"])
    validate_v5_evidence_envelope(evidence, repo_root=root)
    # Acceptance is consumed as recorded, never recomputed from predictions.
    acceptance_ref = package["acceptance_ref"]
    acceptance = load_json_object(_local(root, acceptance_ref))
    if acceptance_ref not in package["artifact_hashes"]:
        raise ValueError("acceptance metadata must be hash bound")
    if acceptance.get("evidence_id") != evidence["evidence_id"] or acceptance.get("outcome") != evidence["outcome"]:
        raise ValueError("acceptance/evidence outcome binding mismatch")
    if acceptance.get("run_id") != run_id:
        raise ValueError("acceptance run mismatch")
    for artifact in evidence["artifacts"]:
        if artifact.get("sha256") is None:
            raise ValueError("new finalization evidence requires artifact hashes")
        if package["artifact_hashes"].get(artifact["locator"]) != artifact["sha256"]:
            raise ValueError("evidence artifact is not in verified input bindings")
    for pointer in evidence["authority"]["pointers"]:
        if pointer not in package["artifact_hashes"]:
            raise ValueError("unbound evidence authority")
    prefix = destination.relative_to(root).as_posix()
    updated = copy.deepcopy(frozen)
    updated["decision"] = copy.deepcopy(package["decision"])
    if updated["decision"]["outcome"] == "ACTIVE":
        raise ValueError("terminal decision is still ACTIVE")
    if updated["decision"] != acceptance.get("trajectory_decision"):
        raise ValueError("terminal decision must match retained acceptance metadata")
    if updated["decision"]["decision_ref"] not in package["artifact_hashes"]:
        raise ValueError("terminal decision authority must be bound")
    comparison_ref = package.get("comparison_readiness_ref")
    comparison = None
    if comparison_ref is not None:
        from molgap.v5_common import validate_comparison_readiness
        if comparison_ref not in package["artifact_hashes"]:
            raise ValueError("terminal comparison metadata must be hash bound")
        comparison = validate_comparison_readiness(
            load_json_object(_local(root, comparison_ref)),
            evidence_verifier=lambda pointer, sha: verify_bound_artifact(root, pointer, sha),
        )
        if comparison.get("reference_id") not in frozen["state_at_start"]["reference_ids"]:
            if paired is None or paired["comparison_role"] != "candidate" or (
                comparison.get("reference_id") != accepted_reference_evidence(root, frozen)[0]
            ):
                raise ValueError("terminal comparison does not use a frozen reference")
        if frozen.get("reference_bundle_id") is not None and comparison["reference_bundle_id"] != frozen["reference_bundle_id"]:
            raise ValueError("terminal comparison changed the frozen reference bundle")
        updated.update(comparison_class=comparison["comparison_class"],
                       comparison_readiness_ref=comparison_ref,
                       comparison_blockers=comparison["blocker_codes"],
                       reference_bundle_id=comparison["reference_bundle_id"])
    updated["result"] = {"evidence_ids": [evidence["evidence_id"]],
                         "evidence_refs": [prefix + "/v5_evidence.json"]}
    if package.get("action_replay") is not None:
        updated["action_replay"] = copy.deepcopy(package["action_replay"])
        from .replay import _action_replay
        _action_replay(root, updated)
    files = {"v5_evidence.json": json_bytes(evidence)}
    replacements = {"trajectory.json": trajectory_path.relative_to(root).as_posix()}
    original_evidence = repo_local_path(root, trajectory_path.parent / "v5_evidence.json")
    if original_evidence.is_file():
        if load_json_object(original_evidence)["evidence_id"] != evidence["evidence_id"]:
            raise ValueError("existing evidence has a different identity")
        replacements["v5_evidence.json"] = original_evidence.relative_to(root).as_posix()
    for kind, validator, id_field in (("costs", validate_cost_event, "cost_event_id"),
                                       ("roles", validate_role_event, "role_event_id")):
        events = package.get(kind, [])
        if not isinstance(events, list):
            raise ValueError(f"{kind} must be an array of observed events")
        for event in events:
            validator(event)
            if (event["trajectory_id"], event["action_id"], event["run_id"]) != (
                frozen["trajectory_id"], action_id, run_id
            ):
                raise ValueError(f"{kind} event identity mismatch")
            pointer = event["evidence_ref"]
            if pointer not in package["artifact_hashes"]:
                raise ValueError(f"{kind} event source is not bound")
            observed = load_json_object(_local(root, pointer))
            if event not in observed.get(kind, []):
                raise ValueError(f"{kind} event absent from observed source metadata")
            relative = f"{kind}/{event[id_field]}.json"
            if relative in files:
                raise ValueError("duplicate terminal event")
            files[relative] = json_bytes(event)
            for old in repo_local_path(root, trajectory_path.parent / kind).glob("*.json"):
                old = repo_local_path(root, old)
                if load_json_object(old).get(id_field) == event[id_field]:
                    replacements[relative] = old.relative_to(root).as_posix()
            if kind == "costs":
                for action in updated["actions"]:
                    if action["action_id"] == action_id:
                        action["cost_event_ids"] = sorted(set(action["cost_event_ids"]) | {event[id_field]})
    validate_observed_role_truth(
        [evidence.get("role_use", {}), package.get("role_use", {}), acceptance.get("role_use", {})],
        package.get("roles", []),
    )
    if not package.get("costs"):
        missing_cost = {
            "schema": "molgap-cost-event-v1", "cost_event_id": "cost-finalize-" + digest[:24],
            "trajectory_id": frozen["trajectory_id"], "action_id": action_id, "run_id": run_id,
            "attempt_id": "unknown", "platform": "unknown", "hardware": "unknown",
            "category": "other", "evidence_ref": prefix + "/v5_evidence.json",
            "measurement": {unit: {"status": "measurement_missing", "value": None}
                            for unit in ("device_hours", "cpu_hours", "wall_hours", "queue_hours")},
        }
        validate_cost_event(missing_cost)
        files["costs/measurement_missing.json"] = json_bytes(missing_cost)
        for action in updated["actions"]:
            if action["action_id"] == action_id:
                action["cost_event_ids"] = sorted(set(action["cost_event_ids"]) | {missing_cost["cost_event_id"]})
    if trace_bytes is not None:
        manifest = copy.deepcopy(package["trace_manifest"])
        validate_trace_manifest(manifest)
        eligible = manifest["backtest_eligibility"]["eligible"]
        if manifest["reference_id"] is None and (paired is not None or frozen["state_at_start"]["reference_ids"]):
            raise ValueError("unreferenced trace cannot discard a frozen reference or same-run binding")
        if package.get("continuation_ref") is not None:
            validate_terminal_continuation(root, package, require_strict=eligible)
        if paired is not None:
            observation = validate_pair_observation(package.get("same_run_observation"))
            if (observation["spec_identity"], observation["logical_run_id"], observation["source_commit"]) != (
                paired["spec_identity"], paired["logical_run_id"], frozen["state_at_start"]["source_commit"]
            ):
                raise ValueError("same-run observation differs from frozen prospective identity")
            if manifest["comparison_role"] != paired["comparison_role"]:
                raise ValueError("trace role differs from frozen same-run arm role")
            if paired["comparison_role"] == "reference":
                if manifest["reference_id"] != evidence["evidence_id"] or evidence["outcome"]["execution_status"] != "complete":
                    raise ValueError("same-run reference trace must bind its complete accepted evidence")
            else:
                reference_id = (accepted_reference_evidence(root, frozen)[0] if eligible
                                else terminal_reference_evidence(root, frozen)[0])
                if manifest["reference_id"] != reference_id:
                    raise ValueError("same-run candidate trace does not bind its accepted control")
                if eligible and (comparison is None or not comparison["strict_ready"]
                                 or comparison["reference_id"] != reference_id):
                    raise ValueError("replay-eligible candidate trace requires strict V5 comparison")
            if not eligible and not manifest["backtest_eligibility"]["exclusion_reasons"]:
                raise ValueError("ineligible same-run trace must explain its replay exclusion")
            if paired["comparison_role"] == "candidate":
                reference_path = resolve_repo_pointer(root, paired["reference_trajectory_ref"])
                assert reference_path is not None
                reference_terminal = load_terminal_input(current_finalized_directory(root, reference_path))
                reference_observation = validate_pair_observation(reference_terminal.get("same_run_observation"))
                if reference_observation != observation:
                    if eligible:
                        raise ValueError("same-run candidate and control have different observed platform jobs")
                    candidate_origin = (validate_terminal_continuation(root, package)
                        if package.get("continuation_ref") is not None else observation)
                    reference_origin = (validate_terminal_continuation(root, reference_terminal)
                        if reference_terminal.get("continuation_ref") is not None else reference_observation)
                    if candidate_origin != reference_origin:
                        raise ValueError("same-run candidate and control have different observed platform jobs")
        elif eligible and (comparison is None or not comparison["strict_ready"]):
            raise ValueError("new eligible trace requires existing strict V5 comparison acceptance")
        if (manifest["trajectory_id"], manifest["run_id"]) != (frozen["trajectory_id"], run_id):
            raise ValueError("manifest identity mismatch")
        if manifest["contract_ref"] not in frozen["state_at_start"]["contract_refs"]:
            raise ValueError("manifest contract not frozen")
        if (paired is None
                and manifest["reference_id"] is not None
                and manifest["reference_id"] not in frozen["state_at_start"]["reference_ids"]):
            raise ValueError("manifest reference not frozen")
        manifest.update(trace_artifact_ref=prefix + "/trace.json",
                        trace_artifact_sha256=hashlib.sha256(trace_bytes).hexdigest(),
                        terminal_evidence_ref=prefix + "/v5_evidence.json")
        validate_trace_manifest(manifest)
        validate_manifest_trace(manifest, canonical)
        files["trace.json"] = trace_bytes
        files["trace_manifest.json"] = json_bytes(manifest)
        old = repo_local_path(root, trajectory_path.parent / "trace_manifest.json")
        if old.exists():
            replacements["trace_manifest.json"] = old.relative_to(root).as_posix()
    elif package.get("trace_manifest") is not None:
        raise ValueError("trace manifest supplied without a canonical artifact")
    # Preserve compact prospective state in the terminal overlay as well.
    if "decision_state" in frozen_raw:
        updated["decision_state"] = frozen_raw["decision_state"]
    validate_trajectory(updated)
    files["trajectory.json"] = json_bytes(updated)
    compact = copy.deepcopy(package)
    compact["format"] = "molgap-rml-terminal-input-v2"
    components = {"evidence": ["v5_evidence.json"], "decision": ["trajectory.json"]}
    # Absence differs from an explicit empty array in the bound source digest.
    for kind, id_field in (("costs", "cost_event_id"), ("roles", "role_event_id")):
        if kind in package:
            components[kind] = [f"{kind}/{event[id_field]}.json" for event in package[kind]]
    compact["published_components"] = {key: [{"ref": name, "sha256": hashlib.sha256(files[name]).hexdigest()}
        for name in names] for key, names in components.items()}
    for key in components:
        compact.pop(key, None)
    files["terminal_input.json"] = json_bytes(compact)
    if previous is not None:
        old_receipt = verified_receipt(previous)
        for name in files:
            if name in old_receipt["published_hashes"]:
                replacements[name] = (previous / name).relative_to(root).as_posix()
    receipt = {
        "format": "molgap-rml-finalization-v2", "finalization_id": "finalize-" + digest,
        "prospective_ref": trajectory_path.relative_to(root).as_posix(),
        "prospective_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "finalizer_version": FINALIZER_VERSION, "source_artifact_digest": digest,
        "finalized_at": package["finalized_at"], "trajectory_id": frozen["trajectory_id"],
        "outcome": updated["decision"]["outcome"], "replacements": replacements,
        "published_hashes": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "input_artifact_hashes": package["artifact_hashes"],
        "trace_status": "available" if trace_bytes is not None else "unavailable",
        "cost_status": "observed_records" if package.get("costs") else "measurement_missing",
        "role_status": "observed_records" if package.get("roles") else "unavailable",
    }
    # Staging lives outside discovery's experiments glob and on the same volume.
    staging_root = repo_local_path(root, "research_memory/.staging")
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="finalize-", dir=staging_root))
    try:
        for name, data in files.items():
            atomic_write(staging / name, data)
        atomic_write(staging / "finalization.json", json_bytes(receipt))
        _validate_staged(root, staging, destination)
        _bindings(root, package["artifact_hashes"])
        if trajectory_path.read_bytes() != frozen_bytes:
            raise ValueError("prospective trajectory changed during finalization")
        # Rename cannot overwrite a nonempty committed directory: competing
        # writers either publish once or fail without a partial visible state.
        lock = None
        if previous is not None:
            revision_root = repo_local_path(root, trajectory_path.parent / "qualification_revisions")
            revision_root.mkdir(parents=True, exist_ok=True)
            lock = revision_root / ".publication.lock"
            lock.mkdir()
        try:
            if previous is not None and current_finalized_directory(root, trajectory_path) != previous:
                raise ValueError("qualification revision parent changed before publication")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.rename(staging, destination)
        finally:
            if lock is not None:
                lock.rmdir()
        sync_directory(destination.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {**receipt, "status": "FINALIZED"}
