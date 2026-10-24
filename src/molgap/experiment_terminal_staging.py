"""Stage existing terminal-package metadata from independently inspected outputs.

Only compact metadata is published in the checkout. Retained tensors and traces
keep their original custody and exact digests. Staging never finalizes RML.
"""
from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path

from .common.io import publish_immutable_bytes
from .experiment_family_workflow import (
    _artifact_path, _check_context, _frozen_action_attempt, _json, _runtime,
    inspect_terminal_output,
)
from .research_memory.paths import artifact_locator, bound_artifact_path, repo_local_path, verify_bound_artifact
from .research_memory.roles import validate_observed_role_truth
from .research_memory.schemas import TRAJECTORY_OUTCOMES, validate_cost_event, validate_role_event, validate_trajectory
from .research_memory.trace import file_digest, json_bytes, load_canonical_trace
from .screen_policy import canonical_fingerprint


PROTECTED_ROLES = ("official_validation", "test_dev", "test_challenge")


def _per_arm(value, arm_id):
    return value.get(arm_id) if isinstance(value, dict) else value


def _role_input(value, arm_id):
    """Select an arm role audit while preserving one inline audit object."""
    if isinstance(value, dict) and arm_id in value:
        return value[arm_id]
    return value


def _materialize_inline_role_audit(root: Path, value, context, trajectory_path: Path,
                                   attempt_id: str, arm_id: str, where=None,
                                   action_id=None):
    """Persist an inline observed role audit before it enters terminal evidence."""
    if value is None or isinstance(value, (str, Path)):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("role audit must be a repository path or mapping")
    audit = dict(value)
    if audit.get("context") is None:
        # The caller supplies observed role facts; the resolved RunContext is
        # the only authority for machine/run identity.  Fill that identity
        # without deriving any role state from it.
        audit["context"] = context.to_dict()
    elif audit["context"] != context.to_dict():
        try:
            _check_context(audit["context"], context)
        except ValueError as exc:
            raise ValueError("Observed role audit differs from the frozen run/source/attempt context") from exc
    if "roles" not in audit:
        audit["roles"] = []
    role_use = audit.get("role_use")
    if not isinstance(role_use, Mapping):
        raise ValueError("Role audit requires observed role_use and existing-schema roles")
    for role in PROTECTED_ROLES:
        if role_use.get(role) not in {"untouched", "consumed", "not_applicable", "unknown"}:
            raise ValueError("Role audit lacks explicit protected-role truth: " + role)
    identity = canonical_fingerprint(audit)
    target = (trajectory_path.parent / "terminal_staging" / attempt_id /
              ("role_audit-" + arm_id + "-" + identity[:16] + ".json"))
    # Inline role events belong to this immutable audit object. Rebind only
    # their evidence source pointer; preserve every observed role fact.
    roles = audit.get("roles")
    if not isinstance(roles, list):
        raise ValueError("Role audit requires observed role_use and existing-schema roles")
    if any(not isinstance(event, Mapping) for event in roles):
        raise ValueError("role audit roles must be existing role-event objects")
    for event in roles:
        validate_role_event(event)
        if where is not None and action_id is not None and (
                event["trajectory_id"], event["run_id"], event["action_id"]) != (
                where["trajectory_id"], where["run_id"], action_id):
            raise ValueError("Observed role event differs from its bound run/action/source audit")
    validate_observed_role_truth([audit.get("role_use", {})], roles)
    audit["roles"] = [
        {**event, "evidence_ref": target.relative_to(root).as_posix()}
        for event in roles
    ]
    publish_immutable_bytes(target, json_bytes(audit))
    return target.relative_to(root).as_posix()


def _cost_events(native, context, where, action_id, source_ref):
    """Translate only measured native seconds; retain everything else as unknown."""
    events = []
    for index, cost in enumerate(native):
        if (cost["status"] != "measured" or cost["unit"] != "seconds"
                or cost["hardware"].lower() in {"unknown", "unavailable", "unspecified"}):
            continue
        field = {("wall_seconds", "process_wall"): "wall_hours",
                 ("device_seconds", "allocated_device"): "device_hours"}.get(
                     (cost["metric"], cost["semantics"]))
        # Busy time is not allocated device hours, and CPU/wall are not interchangeable.
        if field is None:
            continue
        measurement = {name: {"value": None, "status": "measurement_missing"}
                       for name in ("device_hours", "cpu_hours", "wall_hours", "queue_hours")}
        measurement[field] = {"value": cost["value"] / 3600, "status": "measured"}
        event = {"schema": "molgap-cost-event-v1",
            "cost_event_id": f"cost-terminal-{where['trajectory_id']}-{index}",
            "trajectory_id": where["trajectory_id"], "action_id": action_id, "run_id": where["run_id"],
            "attempt_id": context["attempt_id"] or "unknown", "platform": context["platform"],
            "hardware": cost["hardware"], "category": "training", "evidence_ref": source_ref,
            "measurement": measurement}
        events.append(validate_cost_event(event))
    return events


def _roles(root, audit_ref, context, where, action_id):
    if audit_ref is None:
        return {role: "unknown" for role in PROTECTED_ROLES}, [], ["observed_protected_role_audit_missing"], None
    audit_path = repo_local_path(root, audit_ref)
    audit_ref = audit_path.relative_to(root).as_posix()
    audit = _json(audit_path)
    if audit.get("context") != context.to_dict():
        try:
            _check_context(audit.get("context"), context)
        except ValueError as exc:
            raise ValueError("Observed role audit differs from the frozen run/source/attempt context") from exc
    role_use, events = audit.get("role_use"), audit.get("roles")
    if not isinstance(role_use, dict) or not isinstance(events, list):
        raise ValueError("Role audit requires observed role_use and existing-schema roles")
    for role in PROTECTED_ROLES:
        if role_use.get(role) not in {"untouched", "consumed", "not_applicable", "unknown"}:
            raise ValueError("Role audit lacks explicit protected-role truth: " + role)
    for event in events:
        validate_role_event(event)
        if (event["trajectory_id"], event["run_id"], event["action_id"], event["evidence_ref"]) != (
                where["trajectory_id"], where["run_id"], action_id, audit_ref):
            raise ValueError("Observed role event differs from its bound run/action/source audit")
    validate_observed_role_truth([role_use], events)
    blockers = (["observed_protected_role_audit_incomplete"]
                if any(role_use[role] == "unknown" for role in PROTECTED_ROLES) else [])
    return role_use, events, blockers, audit_ref


def _runtime_role_audit(output_root: Path, observed: Mapping):
    """Load the runtime's hash-checked role-audit sidecar when present.

    ``inspect_session_output`` verifies the sidecar bytes and exposes only its
    relative path and digest.  Terminal staging consumes that exact artifact;
    it never reconstructs role facts from the recipe or from a hand-authored
    terminal argument.
    """
    entry = observed.get("role_audit") if isinstance(observed, Mapping) else None
    if entry is None:
        return None
    if (not isinstance(entry, Mapping) or set(entry) - {"path", "sha256", "role_use", "event_count"}
            or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str)):
        raise ValueError("Runtime role audit sidecar binding is invalid")
    path = _artifact_path(output_root, entry["path"])
    if file_digest(path) != entry["sha256"]:
        raise ValueError("Runtime role audit sidecar changed after mechanical inspection")
    return _json(path)


def _scientific_input(value, arm_id):
    """Select a per-arm owner package without mistaking a package for a map."""
    if isinstance(value, Mapping) and arm_id in value:
        return value[arm_id]
    return value


def _validate_scientific_package(root: Path, value, *, where, action,
                                 observed, output_root, outcome, decision_ref):
    """Check an owner terminal package against this run's inspected bytes."""
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ValueError("scientific acceptance must be a repository terminal package path or per-arm path map")
    package_path = repo_local_path(root, value)
    package = _json(package_path)
    if package.get("format") != "molgap-rml-terminal-package-v1":
        raise ValueError("scientific acceptance must use molgap-rml-terminal-package-v1")
    if package.get("closure_blockers"):
        raise ValueError("scientific acceptance package has unresolved closure blockers")
    if (package.get("trajectory_id"), package.get("run_id"), package.get("action_id")) != (
            where["trajectory_id"], where["run_id"], action["action_id"]):
        raise ValueError("scientific acceptance package differs from the frozen trajectory/run/action")
    decision = package.get("decision")
    if not isinstance(decision, Mapping) or decision.get("final") is not True:
        raise ValueError("scientific acceptance requires a final owner decision")
    if decision.get("outcome") in (None, "ACTIVE") or decision.get("outcome") != outcome:
        raise ValueError("scientific acceptance outcome differs from the requested terminal outcome")
    if not isinstance(decision.get("decision_ref"), str) or not decision["decision_ref"]:
        raise ValueError("scientific acceptance decision authority is missing")
    decision_path = repo_local_path(root, decision["decision_ref"])
    expected_decision = repo_local_path(root, decision_ref).relative_to(root).as_posix()
    if decision_path.relative_to(root).as_posix() != expected_decision:
        raise ValueError("scientific acceptance decision authority differs from the requested decision")
    acceptance_ref = package.get("acceptance_ref")
    if not isinstance(acceptance_ref, str):
        raise ValueError("scientific acceptance package lacks acceptance_ref")
    acceptance_path = repo_local_path(root, acceptance_ref)
    acceptance = _json(acceptance_path)
    evidence = package.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("scientific acceptance package lacks V5 evidence")
    from molgap.v5_common import validate_v5_evidence_envelope
    validate_v5_evidence_envelope(evidence, repo_root=root)
    if (acceptance.get("evidence_id") != evidence.get("evidence_id")
            or acceptance.get("run_id") != package.get("run_id")
            or acceptance.get("outcome") != evidence.get("outcome")
            or acceptance.get("trajectory_decision") != decision):
        raise ValueError("scientific acceptance metadata does not match its owner package")
    if acceptance.get("trajectory_id") not in (None, package.get("trajectory_id")):
        raise ValueError("scientific acceptance trajectory identity mismatch")
    roles = package.get("roles", [])
    role_use = package.get("role_use", {})
    if not isinstance(roles, list) or not isinstance(role_use, Mapping):
        raise ValueError("scientific acceptance package has invalid role observations")
    from .research_memory.roles import validate_observed_role_truth
    validate_observed_role_truth([role_use, evidence.get("role_use", {}), acceptance.get("role_use", {})], roles)
    bindings = package.get("artifact_hashes")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError("scientific acceptance package lacks artifact_hashes")
    for locator, digest in bindings.items():
        if not isinstance(locator, str) or not locator:
            raise ValueError("scientific acceptance artifact locator must be a nonempty string")
        verify_bound_artifact(root, locator, digest)
    if acceptance_ref not in bindings or file_digest(acceptance_path) != bindings[acceptance_ref]:
        raise ValueError("scientific acceptance metadata is not hash bound")
    if decision_path.relative_to(root).as_posix() not in bindings:
        raise ValueError("scientific acceptance decision authority is not hash bound")

    current = {
        "predictions": observed["artifacts"]["predictions"],
        "selected_model": observed["artifacts"]["selected_model"],
        "trace": observed["artifacts"]["trace"],
    }
    for role, current_binding in current.items():
        current_path = _artifact_path(output_root, current_binding["path"])
        matches = []
        for artifact in evidence.get("artifacts", []):
            locator = artifact.get("locator")
            if not isinstance(locator, str) or not locator:
                continue
            if (bound_artifact_path(root, locator) == current_path
                    and artifact.get("sha256") == current_binding["sha256"]):
                matches.append(artifact)
        if not matches:
            raise ValueError("scientific acceptance artifact differs from inspected " + role)
        locator = matches[0]["locator"]
        if bindings.get(locator) != current_binding["sha256"]:
            raise ValueError("scientific acceptance " + role + " artifact is not hash bound")
    manifest = package.get("trace_manifest")
    if not isinstance(manifest, Mapping) or manifest.get("trace_artifact_ref") is None:
        raise ValueError("scientific acceptance package lacks trace_manifest")
    trace_locator = manifest["trace_artifact_ref"]
    if not isinstance(trace_locator, str) or not trace_locator:
        raise ValueError("scientific acceptance trace locator must be a nonempty string")
    if bound_artifact_path(root, trace_locator) != _artifact_path(output_root, current["trace"]["path"]):
        raise ValueError("scientific acceptance trace manifest differs from inspected trace")
    if manifest.get("trace_artifact_sha256") != current["trace"]["sha256"]:
        raise ValueError("scientific acceptance trace manifest hash differs from inspected trace")
    return {"path": package_path, "package": package, "acceptance": acceptance,
            "acceptance_path": acceptance_path}


def _diagnostic_binding(root, resolved, report, attempt_id):
    if report is None:
        return None
    from .experiment_attribution import ATTRIBUTION_FORMAT
    if (report.get("format") != ATTRIBUTION_FORMAT
            or report.get("spec_identity") != resolved["spec"].identity
            or report.get("run_id") != resolved["spec"].to_dict()["logical_run_id"]
            or report.get("attempt_id") != attempt_id):
        raise ValueError("Diagnostic report differs from the frozen run/attempt")
    summary = report.get("written_files", {}).get("analysis_summary")
    if summary is None:
        # Standalone API callers may supply an in-memory report. Publish it once
        # beside the original run, then share that immutable pointer across arms.
        run = Path(resolved["run_directory"]).absolute()
        digest = canonical_fingerprint(report)
        summary = run / "reports" / ("terminal_diagnostic_" + digest[:16] + ".json")
        publish_immutable_bytes(summary, json_bytes(report))
    path = Path(summary)
    path = path if path.is_absolute() else root / path
    if canonical_fingerprint(_json(path)) != canonical_fingerprint(report):
        raise ValueError("Retained diagnostic summary differs from the supplied report")
    # An explicit working report may be replaced by a later analysis. Bind one
    # immutable shared copy for terminal evidence, rather than that mutable name.
    identity = canonical_fingerprint({key: value for key, value in report.items() if key != "written_files"})
    own_names = {"diagnostic_" + identity + ".json",
                 "terminal_diagnostic_" + canonical_fingerprint(report)[:16] + ".json"}
    if path.name not in own_names:
        path = Path(resolved["run_directory"]) / "reports" / ("terminal_diagnostic_" + canonical_fingerprint(report) + ".json")
        publish_immutable_bytes(path, json_bytes(report))
    locator = artifact_locator(root, path)
    digest = file_digest(path)
    verify_bound_artifact(root, locator, digest)
    return {"locator": locator, "sha256": digest, "spec_identity": resolved["spec"].identity,
            "logical_run_id": resolved["spec"].to_dict()["logical_run_id"]}


def stage_verified_terminal_packages(profile, repo_root, run_id, attempt_id, *,
        retrieved_root=None, supplied=None, locations=None, outcome=None, decision_ref=None,
        diagnostic_report=None, role_audit=None, resolved=None, reports=None,
        package_dir=None, scientific_acceptance=None):
    """Publish drafts, or explicit decision packages, without canonical RML mutation.

    ``resolved`` and ``reports`` share observations inside one local acceptance
    transaction. Independent callers resolve and inspect again. An optional
    repository-local role audit binds its exact ``context``, observed ``role_use``
    and existing-schema ``roles``; recipe authorization is never an audit.
    An owner supplied ``molgap-rml-terminal-package-v1`` may be reused after
    its decision, role, evidence, and current artifact bindings are checked.
    """
    if resolved is None:
        from .experiment_run import resolve_run_acceptance_inputs
        resolved = resolve_run_acceptance_inputs(profile, repo_root, run_id, attempt_id,
            retrieved_root=retrieved_root, supplied=supplied, locations=locations)
    root = Path(repo_root).resolve()
    spec = resolved["spec"]
    outputs = {arm_id: {**item,
        **({"context": resolved["contexts"][arm_id]} if "context" not in item else {}),
        **({"target_identity": resolved["target_identities"][arm_id]}
           if arm_id in resolved.get("target_identities", {}) else {})}
        for arm_id, item in resolved["outputs"].items()}
    locations = locations if locations is not None else resolved["locations"]
    frozen_options = resolved.get("options", {})
    acceptance_plan = resolved.get("acceptance_plan")
    comparison = (acceptance_plan.get("comparison")
                  if isinstance(acceptance_plan, Mapping) else None)
    frozen_package = package_dir or frozen_options.get("package_dir")
    verification = frozen_options.get("_verification")
    arm_ids = {arm["arm_id"] for arm in spec.to_dict()["arms"]}
    if set(outputs) != arm_ids or set(locations) != arm_ids:
        raise ValueError("Terminal staging requires every independent frozen arm")
    if reports is None:
        reports = {arm_id: inspect_terminal_output(item) for arm_id, item in outputs.items()}
    if set(reports) != arm_ids:
        raise ValueError("Terminal staging requires every independent mechanical report")
    if any(report["status"] != "MECHANICALLY_VERIFIED" for report in reports.values()):
        return {"status": "BLOCKED", "arms": reports, "paths": {}, "locations": {},
                "blockers": ["mechanical_output_verification_failed"], "executed": False}
    diagnostic_binding = _diagnostic_binding(root, resolved, diagnostic_report, attempt_id)
    staged, paths, all_blockers = {}, {}, []
    for arm_id in sorted(arm_ids):
        item, report, where = outputs[arm_id], reports[arm_id], dict(locations[arm_id])
        context = item["context"]
        arm = next(arm for arm in spec.to_dict()["arms"] if arm["arm_id"] == arm_id)
        if context.spec_identity != spec.identity or context.arm_identity != canonical_fingerprint(arm):
            raise ValueError("Terminal context differs from the frozen Spec arm")
        if report.get("context") != context.to_dict():
            raise ValueError("Mechanical report context differs from the frozen launch")
        trajectory_path = repo_local_path(root, where["trajectory"])
        trajectory = validate_trajectory(_json(trajectory_path))
        # A frozen source package is the only authority allowed to restore
        # missing generated contract metadata.  Restore exact pinned bytes
        # before constructing the terminal package so finalization can bind
        # them without reopening or overwriting generated files.
        from .research_memory.terminal_wiring import _restore_frozen_generated_metadata
        restored = _restore_frozen_generated_metadata(
            root, trajectory, frozen_package, _verification=verification
        )
        if trajectory["trajectory_id"] != where["trajectory_id"] or trajectory["record_mode"] != "prospective":
            raise ValueError("Terminal staging requires the owning prospective trajectory")
        actions = [action for action in trajectory["actions"] if where["run_id"] in action["run_ids"]]
        if len(actions) != 1:
            raise ValueError("Terminal run must bind exactly one frozen prospective action")
        action = actions[0]
        action_attempt_id = _frozen_action_attempt(action)
        if action["source_commit"] != context.source_commit:
            raise ValueError("Terminal source/attempt differs from the frozen prospective action")
        observed = report["observed"]
        output_root = Path(item["output_dir"]).absolute()
        manifest_path = _artifact_path(output_root, "output_manifest.json")
        manifest = _json(manifest_path)
        _check_context(manifest.get("context"), context)
        _runtime(manifest.get("runtime"), context, item["expected"].get("precision", "fp32"))
        if (manifest.get("artifacts") != observed["artifacts"] or manifest.get("costs") != observed["costs"]
                or manifest.get("progress") != observed["progress"]):
            raise ValueError("Inspected output manifest changed before terminal staging")
        artifacts, hashes = [], {}
        hashes.update(restored)
        for name, binding in observed["artifacts"].items():
            path = _artifact_path(output_root, binding["path"])
            locator = artifact_locator(root, path)
            verify_bound_artifact(root, locator, binding["sha256"])
            hashes[locator] = binding["sha256"]
            artifacts.append({"name": "training_trace" if name == "trace" else name,
                "role": "training_trace" if name == "trace" else name,
                "locator": locator, "sha256": binding["sha256"],
                "availability": "machine_local_verified" if locator.startswith("external://") else "durable_local_verified"})
            if name == "trace":
                trace_path, trace_locator = path, locator
        trace = load_canonical_trace(trace_path)
        if (trace["trajectory_id"], trace["run_id"]) != (where["trajectory_id"], where["run_id"]):
            raise ValueError("Actual canonical trace differs from the prospective owner/run")
        explicit_outcome, explicit_decision = _per_arm(outcome, arm_id), _per_arm(decision_ref, arm_id)
        blockers = []
        if explicit_outcome is None or explicit_decision is None:
            blockers.append("explicit_outcome_and_decision_ref_required")
        if explicit_outcome is not None and explicit_outcome not in TRAJECTORY_OUTCOMES - {"ACTIVE"}:
            raise ValueError("Expected an explicit existing terminal trajectory outcome")
        if (explicit_outcome == "READY_FOR_DESKTOP"
                or (explicit_outcome in {"POSITIVE_UNDER_CONTRACT", "POSITIVE_BELOW_GATE"}
                    and comparison is None)):
            blockers.append("positive_or_ready_requires_owning_scientific_acceptance")
        if explicit_decision is not None:
            explicit_decision = repo_local_path(root, explicit_decision).relative_to(root).as_posix()
            hashes[explicit_decision] = file_digest(root / explicit_decision)
        owner_input = _scientific_input(scientific_acceptance, arm_id)
        if owner_input is not None:
            if role_audit is not None:
                raise ValueError("scientific acceptance already supplies the terminal role observations")
            if explicit_outcome is None or explicit_decision is None:
                raise ValueError("scientific acceptance requires explicit outcome and decision reference")
            owner = _validate_scientific_package(
                root, owner_input, where=where, action=action,
                observed=observed, output_root=output_root, outcome=explicit_outcome,
                decision_ref=explicit_decision,
            )
            owner_path = owner["path"]
            owner_ref = owner_path.relative_to(root).as_posix()
            staged[arm_id] = {**where, "terminal": owner_ref, "trace": trace_locator}
            paths[arm_id] = {"terminal": str(owner_path),
                "observed": None, "acceptance": str(owner["acceptance_path"]),
                "closure_ready": True, "blockers": []}
            continue
        # A producer sidecar is the authoritative observed role record when
        # present. Explicit audits remain a compatibility path for outputs
        # produced before the runtime sidecar existed.
        runtime_role_audit = _runtime_role_audit(output_root, observed)
        role_input = (runtime_role_audit if runtime_role_audit is not None
                      else _role_input(role_audit, arm_id))
        role_input = _materialize_inline_role_audit(
            root, role_input, context, trajectory_path, attempt_id, arm_id,
            where, action["action_id"]
        )
        role_use, roles, role_blockers, audit_ref = _roles(
            root, role_input, context, where, action["action_id"]
        )
        blockers.extend(role_blockers)
        if audit_ref is not None:
            hashes[audit_ref] = file_digest(root / audit_ref)
        manifest_locator = artifact_locator(root, manifest_path)
        hashes[manifest_locator] = file_digest(manifest_path)
        if diagnostic_binding is not None:
            hashes[diagnostic_binding["locator"]] = diagnostic_binding["sha256"]
        # Input-addressed metadata avoids overwriting a draft when a later audit
        # or explicit decision completes it. Retried identical staging is stable.
        identity = canonical_fingerprint({"report": report, "outcome": explicit_outcome,
            "decision_ref": explicit_decision, "bindings": hashes, "role_use": role_use,
            "roles": roles, "diagnostic": diagnostic_binding})
        directory = repo_local_path(root, trajectory_path.parent / "terminal_staging" / attempt_id / identity[:16])
        pointer = lambda name: (directory / name).relative_to(root).as_posix()
        objective_ref, acceptance_ref = pointer("observed.json"), pointer("acceptance.json")
        terminal_ref = pointer("terminal.json")
        costs = _cost_events(observed["costs"], {"attempt_id": action_attempt_id, "platform": context.platform},
            where, action["action_id"], objective_ref)
        objective = {"context": context.to_dict(), "mechanical_verification": report,
            "native_costs": observed["costs"], "costs": costs, "roles": [],
            "missing_evidence": ["scientific_comparison_not_evaluated", "runtime_qualification_not_evaluated",
                "replay_readiness_not_evaluated", *role_blockers], "diagnostic_report": diagnostic_binding}
        publish_immutable_bytes(directory / "observed.json", json_bytes(objective))
        hashes[objective_ref] = file_digest(directory / "observed.json")
        closure_ready = not blockers
        final_decision = {"outcome": explicit_outcome if closure_ready else "ACTIVE", "final": closure_ready,
            "decision_ref": explicit_decision, "next_allowed_actions": [], "reopen_conditions": []}
        dimensions = {"execution_status": "complete", "artifact_status": "mechanically_verified",
            "comparison_status": "NOT_EVALUATED", "scientific_status": explicit_outcome if closure_ready else "NOT_EVALUATED",
            "transfer_status": "NOT_EVALUATED", "budget_decision": "NOT_EVALUATED", "full_handoff_status": "NOT_EVALUATED"}
        evidence_id = "ev-terminal-" + where["trajectory_id"]
        acceptance = {"run_id": where["run_id"], "evidence_id": evidence_id, "outcome": dimensions,
            "trajectory_decision": final_decision, "role_use": role_use,
            "mechanical_status": "MECHANICALLY_VERIFIED", "closure_blockers": blockers}
        publish_immutable_bytes(directory / "acceptance.json", json_bytes(acceptance))
        hashes[acceptance_ref] = file_digest(directory / "acceptance.json")
        evidence = {"format": "molgap-v5-evidence-envelope-v1", "contract": "MOLGAP-COMMON-V5-FINAL",
            "evidence_id": evidence_id, "track": trajectory["track"], "scope": "terminal_output_inspection",
            "legacy_contract": trajectory["state_at_start"]["contract_refs"][0],
            "outcome": dimensions, "authority": {"pointers": [explicit_decision or objective_ref]},
            "artifacts": artifacts, "role_use": role_use,
            "metrics": {"source_ref": objective_ref, "unit": "eV",
                "development_mae_eV": observed.get("development_mae_eV"),
                "selected_weights": observed.get("selected_weights"),
                "selected_optimizer_step": observed.get("selected_optimizer_step"),
                "missing_reason": None if observed.get("development_mae_eV") is not None else "Development metric not observed"},
            "migration": {"migrated_at": "terminal_staging", "verification_scope": "retained output bytes and explicit decision",
                "training_executed": False, "inference_executed": False, "scientific_reinterpretation": False}}
        from .research_memory.terminal_wiring import build_default_trace_manifest
        trace_manifest = build_default_trace_manifest(root, trajectory, {"run_id": where["run_id"]}, trace)
        trace_manifest.update(trace_artifact_ref=trace_locator, trace_artifact_sha256=hashes[trace_locator],
            terminal_evidence_ref=terminal_ref)
        for field in trace_manifest["trace_fields"]:
            trace_manifest["trace_fields"][field] = any(row.get(field) is not None for row in trace["observations"])
        package = {"format": "molgap-rml-terminal-package-v1", "trajectory_id": where["trajectory_id"],
            "run_id": where["run_id"], "action_id": action["action_id"], "acceptance_ref": acceptance_ref,
            "decision": final_decision, "evidence": evidence, "costs": costs, "roles": roles,
            "role_use": role_use, "artifact_hashes": hashes, "trace_manifest": trace_manifest,
            "closure_blockers": blockers, "requested_outcome": explicit_outcome}
        from .research_memory.paired import pair_binding
        paired = pair_binding(trajectory)
        if paired is not None:
            trace_manifest["comparison_role"] = paired["comparison_role"]
            trace_manifest["reference_id"] = (evidence_id if paired["comparison_role"] == "reference"
                else "ev-terminal-" + paired["reference_trajectory_id"])
            package["same_run_observation"] = {"schema": "molgap-same-run-observation-v1",
                "spec_identity": spec.identity, "logical_run_id": context.logical_run_id,
                "platform_name": context.platform, "platform_run_reference": context.run_reference,
                "attempt_id": action_attempt_id, "source_commit": context.source_commit,
                "source_package_sha256": context.source_archive_sha256}
        existing = directory / "terminal.json"
        package["finalized_at"] = (_json(existing)["finalized_at"] if existing.is_file()
            else datetime.now(timezone.utc).isoformat())
        publish_immutable_bytes(existing, json_bytes(package))
        staged[arm_id] = {**where, "terminal": terminal_ref, "trace": trace_locator}
        paths[arm_id] = {"terminal": str(existing), "observed": str(directory / "observed.json"),
            "acceptance": str(directory / "acceptance.json"), "closure_ready": not blockers, "blockers": blockers}
        all_blockers.extend(f"{arm_id}: {blocker}" for blocker in blockers)
    result = {"status": "STAGED_DRAFT" if all_blockers else "STAGED_CLOSURE_READY",
        "arms": reports, "paths": paths, "locations": staged, "blockers": all_blockers, "executed": False}
    if comparison is not None and scientific_acceptance is None:
        from .family_pair_acceptance import complete_pair_terminal_packages
        result = complete_pair_terminal_packages(root, spec, comparison=comparison,
            outputs=outputs, staged=result)
        pair = spec.to_dict()["prospective"].get("same_run_replay")
        if pair is not None:
            order = [pair["reference_arm_id"], *pair["candidate_arm_ids"]]
            result["locations"] = {arm_id: result["locations"][arm_id] for arm_id in order}
    return result
