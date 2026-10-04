"""Local lifecycle orchestration over existing package, release and RML owners.

Preparation never submits. Acceptance never invents a scientific decision.
All intermediate files are retained when a stage fails, especially partially
published prospective records. A fresh output directory is required per attempt.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from .experiment_execution import check_family_recipes, validate_execution_plan
from .experiment_family_workflow import (
    _json, _artifact_path, check_acceptance_plan, prepare_terminal_outputs,
    build_verified_terminal_descriptor, close_verified_outputs,
)
from .experiment_launch import _safe_local, build_launch_receipt
from .experiment_package import build_experiment_source_package, _allowlist
from .experiment_preflight import check_release_inputs, _atomic
from .experiment_prospective import plan_prospective
from .experiment_spec import ExperimentSpec


# Static local preparation adapters. New platforms implement the same hooks.
_PLATFORM_ADAPTERS = MappingProxyType({"kaggle": "molgap.kaggle_workflow"})


def _platform_adapter(name):
    from importlib import import_module
    if name not in _PLATFORM_ADAPTERS:
        raise ValueError("No registered local workflow preparation adapter for this platform")
    return import_module(_PLATFORM_ADAPTERS[name])


def prepare_workflow(spec: ExperimentSpec, repo_root: Path, plan: dict, output: Path) -> dict:
    """Validate and stage once, publish prospective RML only after static gates.

    source_files is an explicit augmentation of the reviewed shared inventory;
    it never discovers files at runtime or silently copies another experiment.
    Caller supplies existing contract, initialization and scientific plan inputs.
    """
    platform_name = spec.to_dict()["platform"]["name"]
    backend = _platform_adapter(platform_name)
    if type(plan) is not dict or set(plan) != {"format", "spec_identity", "source_files", "arms", "acceptance_plan", platform_name} or plan["format"] != "molgap-experiment-workflow-v1" or plan["spec_identity"] != spec.identity:
        raise ValueError("Workflow plan/Spec mismatch")
    repo_root, output = Path(repo_root).absolute(), Path(output).absolute()
    _safe_local(repo_root)
    _safe_local(output)
    if output.exists():
        raise ValueError("Workflow output must be a fresh directory; reconcile before another attempt")
    metadata = backend.validate_plan(spec, plan[platform_name])
    declaration = spec.to_dict()
    bindings = {b["arm_id"]: b for b in declaration["prospective"].get("arms", [])}
    if type(plan["arms"]) is not list:
        raise ValueError("Expected explicit workflow arms")
    jobs, initial_states, recipes = [], {}, {}
    for arm in plan["arms"]:
        if type(arm) is not dict or set(arm) != {"arm_id", "device", "recipe", "initial_state"} or arm["arm_id"] not in bindings:
            raise ValueError("Expected known arm/device/recipe/initial_state")
        arm_id = arm["arm_id"]
        if arm_id in initial_states:
            raise ValueError("Duplicate workflow arm")
        initial = Path(arm["initial_state"]).absolute()
        _safe_local(initial)
        if not initial.is_file():
            raise ValueError("Missing frozen initialization file")
        initial_states[arm_id], recipes[arm_id] = initial, arm["recipe"]
        jobs.append({"arm_id": arm_id, "device": arm["device"], "recipe": arm["recipe"],
                     "initial_state": f"initial_states/{arm_id}.pt", "trajectory_id": bindings[arm_id]["trajectory_id"]})
    validated_jobs = validate_execution_plan(spec, jobs)
    acceptance_path = _artifact_path(repo_root, plan["acceptance_plan"])
    acceptance = check_acceptance_plan(spec, repo_root, _json(acceptance_path))
    if acceptance["status"] == "BLOCKED":
        return {"status": "BLOCKED", "stage": "acceptance_plan", "acceptance": acceptance,
                "prospective_published": False, "submitted": False}
    # The inventory is maintained with adapters, not assembled by each caller.
    from .experiment_source_inventory import SHARED_SOURCE_FILES
    source_files = _allowlist(sorted(set(SHARED_SOURCE_FILES) | set(plan["source_files"]) | set(recipes.values())))
    output.mkdir(parents=True)
    report = {"format": "molgap-workflow-preparation-v1", "status": "PREPARING",
              "spec_identity": spec.identity, "submitted": False, "stages": []}
    def record(stage, result):
        report["stages"].append({"stage": stage, "result": result})
        _atomic(output / "workflow_report.json", report)
    try:
        package = output / "package"
        manifest = build_experiment_source_package(spec, repo_root, source_files, package)
        record("package", manifest)
        record("family_recipes", check_family_recipes(spec, repo_root, recipes))
        stage = backend.stage_inputs(repo_root=repo_root, output=output, package=package,
            manifest=manifest, spec=spec, platform_plan=plan[platform_name], metadata=metadata,
            initial_states=initial_states, jobs=jobs, acceptance_plan_path=plan["acceptance_plan"])
        staged, kernel = stage["input_root"], stage["kernel_dir"]
        release = check_release_inputs(spec, package,
            expected_package_identity=manifest["package_identity"], recipe_files=recipes,
            initial_states={a: staged / "initial_states" / (a + ".pt") for a in initial_states},
            required_modules=sorted({module for j in validated_jobs for module in j["required_modules"]} |
                                    {"molgap.experiment_training_worker"}),
            input_root=staged)
        _atomic(output / "release_report.json", release)
        record("check_release", release)
        if release["errors"]:
            report.update(status="BLOCKED", stage="check_release", prospective_published=False)
            return report
        prospective, code = plan_prospective(spec, repo_root)
        record("prospective", prospective)
        if code:
            report.update(status="RECONCILIATION_REQUIRED", stage="prospective", prospective_published="inspect_plan_result")
            return report
        backend.freeze_inputs(stage, {arm_id: _artifact_path(repo_root, binding["output"] + "/trajectory.json")
                                    for arm_id, binding in bindings.items()})
        # No second clean-import run: source/recipe/init bytes remain unchanged.
        # Bind the final entry/metadata/prospective files with the same helper
        # used by check_release_inputs and the platform's before-POST recheck.
        backend.bind_release(stage, spec, release)
        _atomic(output / "release_report.json", release)
        record("final_release_binding", release)
        record("launch_receipt", build_launch_receipt(spec, package, expected_package_identity=manifest["package_identity"]))
        report.update(status="PREPARED_FOR_PLATFORM", prospective_published=True,
                      package_identity=manifest["package_identity"], source_commit=manifest["source_commit"],
                      release_report=str(output / "release_report.json"), kernel_dir=str(kernel),
                      source_dataset_dir=str(staged), accelerator=stage["accelerator"],
                      limitations=["Local preparation only; workload skill publishes and submits.",
                                   "Remote all-arm runtime qualification is still mandatory."])
        return report
    except Exception as exc:
        report.update(status="ERROR_REQUIRES_RECONCILIATION", error=str(exc))
        raise
    finally:
        _atomic(output / "workflow_report.json", report)


def accept_workflow(spec, repo_root, supplied, *, receipt_path, package_dir,
                    expected_package_identity, locations, execute=False,
                    target_identities=None):
    """Inspect every arm, build the existing descriptor, then delegate closure."""
    from .experiment_family_workflow import inspect_terminal_output
    outputs = prepare_terminal_outputs(spec, repo_root, supplied, receipt_path=receipt_path,
        package_dir=package_dir, expected_package_identity=expected_package_identity)
    if target_identities is not None:
        if type(target_identities) is not dict or set(target_identities) - set(outputs):
            raise ValueError("Target identity bindings name unknown workflow arms")
        for arm_id, binding in target_identities.items():
            outputs[arm_id]["target_identity"] = binding
    reports = {arm_id: inspect_terminal_output(item)
               for arm_id, item in outputs.items()}
    if any(r["status"] == "BLOCKED" for r in reports.values()):
        return {"status": "BLOCKED", "arms": reports, "executed": False}
    descriptor = build_verified_terminal_descriptor(repo_root, spec, outputs=outputs, locations=locations)
    result = close_verified_outputs(repo_root, spec, descriptor, outputs=outputs, execute=execute)
    return {**result, "descriptor": descriptor.to_dict()}
