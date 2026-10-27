"""Local run-ID adapters over frozen inputs and existing lifecycle owners.

Retrieval, submission and scientific decisions stay with their owning workflows.
This module only resolves the original identity and delegates an explicit action.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .experiment_family_workflow import _json
from .experiment_input import locate_run
from .experiment_launch import _safe_local
from .experiment_spec import ExperimentSpec


class RunReconciliationRequired(ValueError):
    """The original operation journal or launch still needs its owning adapter."""


def _validate_strict_inputs(inputs, outputs):
    if type(inputs) is not dict or set(inputs) - set(outputs):
        raise ValueError("Strict inputs must name only frozen run arms")
    for supplied in inputs.values():
        if type(supplied) is not dict or set(supplied) - {
                "native_root", "comparison_readiness_ref", "actual_resume_root",
                "recovery_repair", "repair_source_root", "target_transform_asset"}:
            raise ValueError("Unknown strict acceptance location override")


def resolve_run_acceptance_inputs(*args, **kwargs):
    """Resolve frozen expectations and retained all-arm acceptance locations."""
    from .experiment_run_acceptance import resolve_run_acceptance_inputs as resolve
    return resolve(*args, **kwargs)


def stage_verified_terminal_packages(*args, **kwargs):
    """Stage through the existing terminal owner without importing it at lookup."""
    from .experiment_terminal_staging import stage_verified_terminal_packages as stage
    return stage(*args, **kwargs)


def _original_run(profile, repo_root, run_id, attempt_id, *, _verification=None):
    located = locate_run(profile, repo_root, run_id, attempt_id,
        **({"_verification": _verification} if _verification is not None else {}))
    run = Path(located["run_directory"])

    def original(name):
        record = located["files"][name]
        if record is None:
            raise ValueError("Missing original run file: " + name)
        path = Path(record["path"])
        _safe_local(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Original run file changed during resolution: " + name)
        return path

    spec = ExperimentSpec.from_json(original("inputs/experiment_spec.json").read_text(encoding="utf-8"))
    if spec.identity != located["spec_identity"]:
        raise ValueError("Original Spec changed during resolution")
    binding = _json(original("inputs/input_binding.json"))
    plan = _json(original("inputs/workflow_plan.json"))
    journal_record = located["files"]["operations/submission_state.json"]
    journal = _json(original("operations/submission_state.json")) if journal_record else None
    # A retained receipt must never bypass the original journal's uncertainty.
    if journal is not None and journal.get("phase") != "BOUND":
        raise RunReconciliationRequired("Reconcile the original submission journal before local lifecycle actions")
    receipt = located["receipt"]
    if receipt is None or receipt["submission_state"] not in {"ACCEPTED", "RECONCILED_EXISTING"}:
        raise RunReconciliationRequired("An original reconciled observed launch receipt is required")
    report = _json(original("prepared/workflow_report.json"))
    identity = located["package_identity"]
    source = receipt["binding"]["source_commit"]
    if (report.get("status") != "PREPARED_FOR_PLATFORM"
            or report.get("spec_identity") != spec.identity
            or report.get("package_identity") != identity
            or report.get("source_commit") != source
            or (binding.get("source_commit") is not None and binding["source_commit"] != source)):
        raise ValueError("Original preparation/package/source binding changed")
    if journal is not None:
        bound = journal.get("binding", {})
        response = journal.get("kernel_response", {})
        reference = receipt["canonical_platform_reference"]["value"]
        version = receipt["platform_version"]["value"]
        if (bound.get("package_identity") != identity
                or bound.get("authority_ref") != binding.get("authority_ref")
                or bound.get("monitor_paths") != binding.get("monitor_paths")
                or bound.get("kernel") != reference
                or response.get("kernel") != reference
                or str(response.get("version_number")) != version
                or ("platform" in plan and bound.get("dataset") != plan["platform"]["source_dataset"])):
            raise ValueError("Original journal differs from the frozen run/receipt")
    return spec, dict(repo_root=Path(repo_root).absolute(), package_dir=run / "prepared/package",
                      expected_package_identity=identity,
                      receipt_path=original("operations/launch_receipt.json"),
                      **({"_verification": _verification} if _verification is not None else {})), run / "prepared"


def inspect_run_output(profile, repo_root, run_id, attempt_id, *, arm_id, artifact_root, expectations):
    """Inspect an explicit retrieved arm with independent frozen expectations."""
    from .experiment_family_workflow import RunContext, inspect_output
    spec, options, _ = _original_run(profile, repo_root, run_id, attempt_id)
    context = RunContext.from_launch(spec, options["receipt_path"], options["package_dir"],
        expected_package_identity=options["expected_package_identity"], arm_id=arm_id)
    root = Path(artifact_root).absolute()
    _safe_local(root)
    return inspect_output(root, context=context, expected=expectations)


def accept_run_workflow(profile, repo_root, run_id, attempt_id, *, supplied=None, locations=None,
                        retrieved_root=None, execute=False, execution_root=None,
                        diagnose=False, stage_terminal=False, outcome=None, decision_ref=None,
                        diagnostic_output=None, n_bootstrap=0, seed=42, role_audit=None,
                        scientific_acceptance=None, strict_inputs=None,
                        continuation_receipt=None, physical_runs=None,
                        resume_prepared=None, resume_authority=None):
    """Inspect saved outputs and optionally diagnose, draft or explicitly close RML."""
    from .experiment_run_acceptance import resolve_run_acceptance_inputs
    from .experiment_workflow import accept_workflow
    if (outcome is None) != (decision_ref is None):
        raise ValueError("A scientific outcome and decision reference must be provided together")
    if role_audit is not None and not (stage_terminal or outcome is not None):
        raise ValueError("A role audit requires terminal staging")
    if scientific_acceptance is not None and outcome is None:
        raise ValueError("Scientific acceptance requires an explicit outcome and decision reference")
    if not diagnose and (diagnostic_output is not None or n_bootstrap or seed != 42):
        raise ValueError("Diagnostic options require diagnose=True")
    if (resume_prepared is None) != (resume_authority is None):
        raise ValueError("Resume preparation and explicit recovery authority are required together")
    if resume_prepared is not None and (continuation_receipt is None
            or not (stage_terminal or outcome is not None)):
        raise ValueError("Resume closure requires a continued receipt and terminal staging")
    automatic = diagnose or stage_terminal or outcome is not None or strict_inputs is not None
    if supplied is not None and locations is not None and retrieved_root is None and not automatic:
        # Preserve explicit legacy plans owned by Server-only acceptance adapters.
        spec, options, _ = _original_run(profile, repo_root, run_id, attempt_id)
        return accept_workflow(spec, supplied=supplied, locations=locations, execute=execute,
                               execution_root=execution_root, **options)
    resolution = dict(retrieved_root=retrieved_root, supplied=supplied, locations=locations)
    if continuation_receipt is not None:
        resolution["continuation_receipt"] = continuation_receipt
    resolved = resolve_run_acceptance_inputs(profile, repo_root, run_id, attempt_id, **resolution)
    if strict_inputs is not None:
        _validate_strict_inputs(strict_inputs, resolved["outputs"])
    if resume_prepared is not None:
        from .experiment_workflow_resume import resolve_terminal_resume
        resolved["continuations"] = resolve_terminal_resume(resolved, resume_prepared,
            resume_authority, strict_inputs=strict_inputs)
    spec, options = resolved["spec"], resolved["options"]
    identity = {"run_id": run_id, "attempt_id": attempt_id, "spec_identity": spec.identity}
    # Ordinary inspection does not depend on a future terminal input package.
    inspections = {}
    result = accept_workflow(spec, supplied=resolved["outputs"],
        locations=None if automatic or locations is None else resolved["locations"],
        execute=False if automatic or locations is None else execute,
        execution_root=execution_root, target_identities=resolved.get("target_identities"),
        _inspections=inspections, **options)
    result.update(identity)
    if physical_runs is not None:
        from .experiment_allocation import aggregate_physical_allocation_costs
        result["physical_allocation_costs"] = aggregate_physical_allocation_costs(physical_runs)
    if result["status"] == "BLOCKED":
        return result
    read_only_strict = (strict_inputs is not None and locations is not None
                        and not execute and not stage_terminal and outcome is None)
    canonical_inputs = False
    if strict_inputs is not None and outcome is None and decision_ref is None and role_audit is None:
        from .research_memory.finalize import (current_finalized_directory,
            verified_receipt, load_terminal_input)
        canonical, canonical_eligible = {}, True
        for arm_id, where in resolved["locations"].items():
            retained = resolved.get("_canonical_finalizations", {}).get(arm_id)
            if retained is None:
                current = current_finalized_directory(repo_root, where["trajectory"])
                if not current.is_dir():
                    break
                verified_receipt(current)
                terminal = load_terminal_input(current)
            else:
                current, terminal = retained
            canonical_eligible = (canonical_eligible
                and terminal["trace_manifest"]["backtest_eligibility"]["eligible"])
            canonical[arm_id] = {**where,
                "terminal": (current / "terminal_input.json").relative_to(
                    Path(repo_root).resolve()).as_posix(),
                "trace": terminal["trace_manifest"]["trace_artifact_ref"]}
        if len(canonical) == len(resolved["outputs"]):
            resolved["locations"] = canonical
            canonical_inputs = True
            read_only_strict = read_only_strict or canonical_eligible
    # The first mechanical pass is the only producer inspection in this
    # transaction. Strict validation reuses its report/bytes independently.
    resolved["mechanical_reports"] = result["arms"]
    target_transforms = None
    same_run = spec.to_dict().get("prospective", {}).get("same_run_replay")
    if strict_inputs is not None and not read_only_strict and same_run is not None:
        from .experiment_family_workflow import resolve_strict_target_transforms
        target_transforms = resolve_strict_target_transforms(repo_root, resolved, strict_inputs)
        if target_transforms["status"] == "BLOCKED":
            return {**result, "status": "BLOCKED", "executed": False,
                "strict_comparison": target_transforms, "blockers": target_transforms["blockers"]}
    native, native_cache = None, None
    if canonical_inputs and not read_only_strict:
        inspected = inspect_run_strict_evidence(repo_root, resolved, strict_inputs)
        native = {arm: row["native_qualification"]
                  for arm, row in inspected["arms"].items()}
        if any(value["status"] != "VERIFIED" for value in native.values()):
            return {**result, "status": "BLOCKED", "executed": False,
                "strict_validation": inspected,
                "blockers": ["native_proof_dependencies_missing"]}
        native_cache = {arm: (value, resolved["contexts"][arm].to_dict())
                        for arm, value in native.items()}
    if strict_inputs is not None and not read_only_strict and native_cache is None:
        from .family_training_runner import inspect_retained_native_qualification
        from .experiment_family_workflow import _artifact_path, _json_bytes
        native, native_contexts = {}, {}
        for arm_id, item in resolved["outputs"].items():
            output = Path(item["output_dir"])
            supplied_strict = strict_inputs.get(arm_id, {})
            continuation = resolved.get("continuations", {}).get(arm_id)
            native_root = supplied_strict.get("native_root", output)
            native_context = resolved["contexts"][arm_id]
            resume_options = {}
            if continuation is not None:
                from .experiment_family_workflow import RunContext
                native_root = continuation["native_root"]
                native_context = RunContext.from_launch(spec, continuation["origin_receipt"],
                    options["package_dir"],
                    expected_package_identity=options["expected_package_identity"], arm_id=arm_id)
                resume_options = {
                    "resume_directory": continuation["actual_resume_root"],
                    "resume_checkpoint_sha256": _json(continuation["resume_bundle"])
                        ["artifacts"]["checkpoint"]["sha256"],
                }
            native_contexts[arm_id] = native_context
            provenance_path = _artifact_path(output, "runtime_provenance.json")
            provenance_bytes = provenance_path.read_bytes()
            native[arm_id] = inspect_retained_native_qualification(native_root,
                recipe_path=_artifact_path(output, "training_contract.json"),
                provenance=_json_bytes(provenance_bytes), context=native_context,
                **resume_options)
            original_digest = hashlib.sha256(provenance_bytes).hexdigest()
            original = next((dependency for dependency in native[arm_id]["dependencies"]
                if Path(dependency["local_path"]).resolve() == provenance_path.resolve()), None)
            if original is not None:
                if original.get("verified_sha256") != original_digest:
                    raise ValueError("Native qualification provenance changed during verification")
            else:
                native[arm_id]["dependencies"].append({"path": provenance_path.name,
                    "local_path": str(provenance_path), "sha256": original_digest,
                    "verified_sha256": original_digest, "present": True,
                    "size_bytes": len(provenance_bytes)})
                native[arm_id]["known_local_bytes"] += len(provenance_bytes)
        if any(value["status"] != "VERIFIED" for value in native.values()):
            return {**result, "status": "BLOCKED", "executed": False,
                "strict_validation": {"status": "BLOCKED", "native": native},
                "blockers": ["native_proof_dependencies_missing"]}
        # Continuation qualification inspects the frozen origin proof.  Keep
        # that context in the cache; the successor's physical observation is
        # checked separately by validate_terminal_continuation.
        native_cache = {arm: (value, native_contexts[arm].to_dict())
                        for arm, value in native.items()}
    if diagnose:
        from .experiment_attribution import diagnose_run, _write_run_diagnosis
        diagnostic = diagnose_run(profile, repo_root, run_id, attempt_id,
            retrieved_root=retrieved_root, supplied=supplied, output=diagnostic_output,
            n_bootstrap=n_bootstrap, seed=seed, resolved=resolved, reports=result["arms"])
        if diagnostic_output is None:
            from .screen_policy import canonical_fingerprint
            path = resolved["run_directory"] / "reports" / ("diagnostic_" + canonical_fingerprint(diagnostic) + ".json")
            diagnostic = _write_run_diagnosis(diagnostic, path, resolved)
        result["diagnosis"] = diagnostic
        if diagnostic.get("status") == "BLOCKED":
            return {**result, "status": "BLOCKED"}
    if read_only_strict:
        strict = inspect_run_strict_evidence(repo_root, resolved, strict_inputs)
        return {**result,
            "status": result["status"] if strict["status"] == "VERIFIED" else "BLOCKED",
            "strict_validation": strict, "strict_mode": "CANONICAL_READ_ONLY",
            "executed": False,
            "replay_readiness": "VERIFIED" if strict["status"] == "VERIFIED" else "BLOCKED",
            "blockers": strict["blockers"]}
    if (stage_terminal or outcome is not None) and not canonical_inputs:
        from .experiment_terminal_staging import stage_verified_terminal_packages
        staged = stage_verified_terminal_packages(profile, repo_root, run_id, attempt_id,
            retrieved_root=retrieved_root, supplied=supplied, locations=locations,
            outcome=outcome, decision_ref=decision_ref, role_audit=role_audit,
            diagnostic_report=result.get("diagnosis"), resolved=resolved, reports=result["arms"],
            **({"scientific_acceptance": scientific_acceptance} if scientific_acceptance is not None else {}))
        result["terminal_staging"] = staged
        if execute:
            if staged.get("blockers") or outcome is None:
                return {**result, "status": "BLOCKED", "executed": False}
            closed = accept_workflow(spec, supplied=resolved["outputs"], locations=staged["locations"],
                execute=True, execution_root=execution_root, _inspections=inspections,
                target_identities=resolved.get("target_identities"), **options)
            if closed["status"] == "COMPLETE" and resolved["acceptance_plan"].get("comparison") is not None:
                from .experiment_family_workflow import family_replay_status
                from .family_pair_acceptance import register_session_reference
                closed.update(family_replay_status(repo_root,
                    [item["trajectory_id"] for item in staged["locations"].values()]))
                if closed["replay_readiness"] == "REPLAY_READY":
                    registered = register_session_reference(repo_root, spec, staged)
                    if registered is not None:
                        closed["accepted_reference"] = registered
            return {**closed, **identity, "diagnosis": result.get("diagnosis"), "terminal_staging": staged}
    if strict_inputs is not None:
        from .experiment_family_workflow import qualify_same_run_terminal_packages
        active_locations = (result.get("terminal_staging", {}).get("locations") or resolved["locations"])
        if same_run is not None:
            qualification = qualify_same_run_terminal_packages(repo_root, resolved, active_locations, native,
                reports=result["arms"], strict_inputs=strict_inputs, _target_transforms=target_transforms)
        else:
            # Historical references use their retained comparison bindings;
            # they cannot be translated into a same-run reference declaration.
            qualification = {"locations": active_locations, "qualification_mode": "RETAINED_REFERENCE"}
        if qualification.get("status") == "BLOCKED":
            return {**result, "status": "BLOCKED", "executed": False,
                "strict_comparison": qualification, "blockers": qualification["blockers"]}
        qualified = {**resolved, "locations": qualification["locations"]}
        result["strict_comparison"] = qualification
        if qualification.get("canonical_reused"):
            strict = inspect_run_strict_evidence(repo_root, qualified, strict_inputs,
                _native_reports=native_cache)
            return {**result,
                "status": "COMPLETE" if strict["status"] == "VERIFIED" else "BLOCKED",
                "strict_validation": strict, "strict_mode": "CANONICAL_READ_ONLY",
                "executed": False, "blockers": strict["blockers"]}
        result["strict_validation"] = inspect_run_strict_evidence(repo_root, qualified,
            strict_inputs, require_finalized=False, _native_reports=native_cache)
        if result["strict_validation"]["status"] != "VERIFIED":
            return {**result, "status": "BLOCKED", "executed": False,
                "blockers": result["strict_validation"]["blockers"]}
        if execute:
            closed = accept_workflow(spec, supplied=resolved["outputs"],
                locations=qualification["locations"], execute=True,
                execution_root=execution_root, _inspections=inspections,
                target_identities=resolved.get("target_identities"), **options)
            result.update(closed)
            result["strict_validation"] = inspect_run_strict_evidence(repo_root,
                qualified, strict_inputs, _native_reports=native_cache)
            if result["strict_validation"]["status"] != "VERIFIED":
                result["status"] = "BLOCKED"
        return result
    elif execute and automatic:
        if locations is None:
            raise ValueError("Use --stage-terminal with explicit --outcome/--decision-ref, or reviewed --locations")
        closed = accept_workflow(spec, supplied=resolved["outputs"], locations=resolved["locations"],
            execute=True, execution_root=execution_root, _inspections=inspections,
            target_identities=resolved.get("target_identities"), **options)
        return {**closed, **identity, "diagnosis": result.get("diagnosis")}
    elif execute and locations is None:
        raise ValueError("Use --stage-terminal with explicit --outcome/--decision-ref, or reviewed --locations")
    return result


def inspect_run_strict_evidence(repo_root, resolved, inputs, *, require_finalized=True,
                                _native_reports=None):
    """Recheck native, comparison and replay evidence through existing owners."""
    from .comparison_readiness import validate_comparison_readiness
    from .experiment_family_workflow import _artifact_path
    from .family_training_runner import inspect_retained_native_qualification
    from .research_memory.finalize import (current_finalized_directory,
        verified_receipt, load_terminal_input, _bindings)
    from .research_memory.paths import repo_local_path, bound_artifact_path, verify_bound_artifact
    from .research_memory.replay import build_replay_pool
    from .research_memory.schemas import validate_trace_manifest, validate_trajectory
    from .research_memory.trace import file_digest, load_canonical_trace, validate_manifest_trace
    from .v4_runtime import normalized_source_sha256
    _validate_strict_inputs(inputs, resolved["outputs"])
    root = Path(repo_root).resolve()
    arms, blockers = {}, []
    records = {kind: [] for kind in ("trajectories", "evidence", "traces", "costs",
                                     "comparison_readiness")}
    for arm_id, item in sorted(resolved["outputs"].items()):
        supplied = inputs.get(arm_id, {})
        output = Path(item["output_dir"]).absolute()
        row = {"native_qualification": {"status": "BLOCKED"},
            "comparison": {"status": "BLOCKED"}, "replay": {"status": "BLOCKED"},
            "blockers": []}
        arms[arm_id] = row
        terminal = continuation_record = metadata_error = None
        try:
            trajectory_path = repo_local_path(root, resolved["locations"][arm_id]["trajectory"])
            finalized = trajectory_path.parent / "rml_finalized"
            if require_finalized:
                finalized = current_finalized_directory(root, trajectory_path)
                verified_receipt(finalized)
                terminal = load_terminal_input(finalized)
                trajectory = validate_trajectory(_json(finalized / "trajectory.json"))
                manifest = validate_trace_manifest(_json(finalized / "trace_manifest.json"))
            else:
                from .research_memory.plan import resolve_decision_state
                terminal = _json(repo_local_path(root, resolved["locations"][arm_id]["terminal"]))
                trajectory = validate_trajectory(resolve_decision_state(root, _json(trajectory_path)))
                manifest = validate_trace_manifest(terminal["trace_manifest"])
            if terminal.get("continuation_ref") is not None:
                pointer = terminal["continuation_ref"]
                verify_bound_artifact(root, pointer, terminal["artifact_hashes"][pointer])
                continuation_record = _json(bound_artifact_path(root, pointer))
        except (ValueError, OSError, KeyError) as exc:
            metadata_error = exc
        try:
            context = resolved["contexts"][arm_id]
            if _native_reports is None:
                native_root, native_context, resume_options = supplied.get("native_root", output), context, {}
                if continuation_record is not None:
                    from .experiment_family_workflow import RunContext
                    native_root = supplied.get("native_root", bound_artifact_path(root,
                        continuation_record["native_certificate"]["path"]).parent)
                    native_context = RunContext(**_json(bound_artifact_path(root,
                        continuation_record["source_context"]["path"]))["context"])
                    resume_options = {
                        "resume_checkpoint_sha256": continuation_record["checkpoint"]["sha256"],
                        "resume_directory": supplied.get("actual_resume_root", bound_artifact_path(
                            root, continuation_record["actual_resume_certificate"]["path"]).parent),
                    }
                row["native_qualification"] = inspect_retained_native_qualification(
                    native_root, recipe_path=_artifact_path(output, "training_contract.json"),
                    provenance=_json(_artifact_path(output, "runtime_provenance.json")),
                    context=native_context, **resume_options)
            else:
                cached, observed_context = _native_reports[arm_id]
                expected_context = context.to_dict()
                if continuation_record is not None:
                    expected_context = _json(bound_artifact_path(root,
                        continuation_record["source_context"]["path"]))["context"]
                if cached["status"] != "VERIFIED" or observed_context != expected_context:
                    raise ValueError("Native verification transaction context changed")
                if normalized_source_sha256(_artifact_path(output, "training_contract.json")) != context.training_recipe_sha256:
                    raise ValueError("Native verification transaction recipe changed")
                for dependency in cached["dependencies"]:
                    if file_digest(Path(dependency["local_path"])) != dependency["verified_sha256"]:
                        raise ValueError("Native verification transaction dependency changed")
                row["native_qualification"] = cached
            if row["native_qualification"]["status"] != "VERIFIED":
                row["blockers"].extend(row["native_qualification"].get("blockers", []))
        except (ValueError, OSError, KeyError) as exc:
            row["native_qualification"]["error"] = str(exc)
            row["blockers"].append("native_qualification_invalid")
        try:
            if metadata_error is not None:
                raise metadata_error
            _bindings(root, terminal.get("artifact_hashes"))
            if require_finalized and "mechanical_reports" in resolved:
                from .experiment_family_workflow import _verify_canonical_output
                _verify_canonical_output(root, resolved["spec"], context, terminal,
                    resolved["mechanical_reports"][arm_id]["observed"], output)
            if terminal.get("continuation_ref") is not None:
                from .research_memory.paired import validate_terminal_continuation
                cached = ((row["native_qualification"], continuation_record["checkpoint"]["sha256"])
                          if row["native_qualification"]["status"] == "VERIFIED" else None)
                validate_terminal_continuation(root, terminal, require_strict=True,
                    _native_verification=cached)
            verify_bound_artifact(root, manifest["trace_artifact_ref"], manifest["trace_artifact_sha256"])
            validate_manifest_trace(manifest,
                load_canonical_trace(bound_artifact_path(root, manifest["trace_artifact_ref"])))
            if require_finalized:
                records["trajectories"].append((finalized / "trajectory.json", trajectory))
                records["evidence"].append((finalized / "v5_evidence.json", _json(finalized / "v5_evidence.json")))
                records["traces"].append((finalized / "trace_manifest.json", manifest))
                for path in sorted((finalized / "costs").glob("*.json")):
                    records["costs"].append((path, _json(path)))
            comparison_ref = supplied.get("comparison_readiness_ref", terminal.get("comparison_readiness_ref"))
            if comparison_ref is None:
                if manifest["comparison_role"] == "reference":
                    row["comparison"] = {"status": "NOT_APPLICABLE", "reason": "accepted_reference_arm"}
                else:
                    row["blockers"].append("strict_comparison_readiness_missing")
            else:
                digest = terminal.get("artifact_hashes", {}).get(comparison_ref)
                if digest is None or terminal.get("comparison_readiness_ref") != comparison_ref:
                    raise ValueError("Strict comparison is not bound to the accepted terminal input")
                verify_bound_artifact(root, comparison_ref, digest)
                comparison = validate_comparison_readiness(_json(bound_artifact_path(root, comparison_ref)),
                    evidence_verifier=lambda pointer, sha: verify_bound_artifact(root, pointer, sha))
                if require_finalized:
                    records["comparison_readiness"].append((bound_artifact_path(root, comparison_ref), comparison))
                if comparison["candidate_id"] != terminal["evidence"]["evidence_id"]:
                    raise ValueError("Strict comparison candidate differs from the accepted terminal result")
                row["comparison"] = {"status": "VERIFIED" if comparison["strict_ready"] else "BLOCKED",
                    "comparison_class": comparison["comparison_class"],
                    "blockers": comparison["blocker_codes"]}
                if not comparison["strict_ready"]:
                    row["blockers"].extend(comparison["blocker_codes"] or ["strict_comparison_not_ready"])
            if not manifest["backtest_eligibility"]["eligible"]:
                row["blockers"].extend(manifest["backtest_eligibility"]["exclusion_reasons"])
        except (ValueError, OSError, KeyError) as exc:
            row["replay"]["error"] = str(exc)
            row["blockers"].append("accepted_replay_evidence_missing_or_invalid")
    if not require_finalized:
        for arm_id, row in arms.items():
            row["replay"] = {"status": "PENDING_FINALIZATION",
                "reason": "canonical pool is checked after terminal closure"}
            row["blockers"] = sorted(set(row["blockers"]))
            blockers.extend(arm_id + ":" + value for value in row["blockers"])
        return {"status": "BLOCKED" if blockers else "VERIFIED", "arms": arms,
            "blockers": blockers, "replay_readiness": "PENDING_FINALIZATION",
            "ready_for_desktop": "NOT_EVALUATED"}
    try:
        pool = build_replay_pool(root, records)
        eligible = {entry["trajectory_id"] for entry in pool["entries"]}
        for arm_id, row in arms.items():
            identity = resolved["locations"][arm_id]["trajectory_id"]
            exclusions = [entry for entry in pool["exclusions"] if entry["trajectory_id"] == identity]
            row["replay"] = {**row["replay"], "status": "VERIFIED" if identity in eligible else "BLOCKED",
                "exclusions": exclusions}
            if identity not in eligible:
                row["blockers"].append("strict_replay_pool_membership_missing")
    except (ValueError, OSError, KeyError) as exc:
        for row in arms.values():
            row["replay"]["error"] = str(exc)
            row["blockers"].append("strict_replay_pool_invalid")
    for arm_id, row in arms.items():
        row["blockers"] = sorted(set(row["blockers"]))
        blockers.extend(arm_id + ":" + value for value in row["blockers"])
    return {"status": "BLOCKED" if blockers else "VERIFIED", "arms": arms,
        "blockers": blockers, "scientific_acceptance": "NOT_EVALUATED",
        "ready_for_desktop": "NOT_EVALUATED", "training_executed": False,
        "inference_executed": False, "network_accessed": False}


def build_run_resume(profile, repo_root, run_id, attempt_id, *, arm_id, source_output, output, recovery_repair=None):
    """Export one already retrieved prefix through the original recovery owner."""
    from .experiment_workflow_resume import build_workflow_resume
    spec, options, prepared = _original_run(profile, repo_root, run_id, attempt_id)
    output = _resume_output(repo_root, prepared.parent, output)
    return build_workflow_resume(spec, prepared=prepared, arm_id=arm_id,
                                 source_output=source_output, output=output, recovery_repair=recovery_repair, **options)


def prepare_run_resume(profile, repo_root, run_id, attempt_id, *, plan, output):
    """Stage explicit validated bundles without publishing or submitting."""
    from .experiment_workflow_resume import prepare_resumed_workflow
    spec, options, prepared = _original_run(profile, repo_root, run_id, attempt_id)
    output = _resume_output(repo_root, prepared.parent, output)
    return prepare_resumed_workflow(spec, prepared=prepared, plan=plan, output=output, **options)


def _resume_output(repo_root, run, output):
    path = Path(output).absolute()
    _safe_local(path)
    checkout = Path(repo_root).resolve()
    protected = (checkout, *(run / name for name in ("inputs", "prepared", "operations")))
    if any(path.is_relative_to(root) or root.is_relative_to(path) for root in protected):
        raise ValueError("Resume output must be outside the checkout and original frozen run records")
    if path.exists():
        raise ValueError("Resume output requires a fresh destination")
    return path
