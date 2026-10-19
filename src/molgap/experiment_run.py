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


def _original_run(profile, repo_root, run_id, attempt_id):
    located = locate_run(profile, repo_root, run_id, attempt_id)
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
                      receipt_path=original("operations/launch_receipt.json")), run / "prepared"


def inspect_run_output(profile, repo_root, run_id, attempt_id, *, arm_id, artifact_root, expectations):
    """Inspect an explicit retrieved arm with independent frozen expectations."""
    from .experiment_family_workflow import RunContext, inspect_output
    spec, options, _ = _original_run(profile, repo_root, run_id, attempt_id)
    context = RunContext.from_launch(spec, options["receipt_path"], options["package_dir"],
        expected_package_identity=options["expected_package_identity"], arm_id=arm_id)
    root = Path(artifact_root).absolute()
    _safe_local(root)
    return inspect_output(root, context=context, expected=expectations)


def accept_run_workflow(profile, repo_root, run_id, attempt_id, *, supplied, locations,
                        execute=False, execution_root=None):
    """Delegate an explicit original acceptance plan; no inferred verdicts."""
    from .experiment_workflow import accept_workflow
    spec, options, _ = _original_run(profile, repo_root, run_id, attempt_id)
    return accept_workflow(spec, supplied=supplied, locations=locations, execute=execute,
                           execution_root=execution_root, **options)


def build_run_resume(profile, repo_root, run_id, attempt_id, *, arm_id, source_output, output):
    """Export one already retrieved prefix through the original recovery owner."""
    from .experiment_workflow_resume import build_workflow_resume
    spec, options, prepared = _original_run(profile, repo_root, run_id, attempt_id)
    output = _resume_output(repo_root, prepared.parent, output)
    return build_workflow_resume(spec, prepared=prepared, arm_id=arm_id,
                                 source_output=source_output, output=output, **options)


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
