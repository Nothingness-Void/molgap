"""Compose existing prospective and release staging APIs, without a submitter."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .experiment_launch import _safe_local
from .experiment_package import _allowlist, _name, _source, _spec
from .experiment_prospective import check_prospective, plan_prospective
from .experiment_spec import SCHEMA_VERSION_V2, _digest, _unique_object
from .experiment_staging import UploadArtifact, _artifact_name, stage_release_inputs, validate_staging_inputs
from .research_memory.trace import atomic_write, json_bytes
from .v4_bundle import _assert_clean_paths, _payload, _tracked_paths

from types import MappingProxyType

from .experiment_execution import check_family_recipes, validate_execution_plan
from .experiment_family_workflow import (
    _json, _artifact_path, check_acceptance_plan, prepare_terminal_outputs,
    build_verified_terminal_descriptor, close_verified_outputs,
)
from .experiment_launch import _safe_local, build_launch_receipt
from .experiment_package import build_experiment_source_package, _allowlist
from .experiment_preflight import check_release_inputs, _atomic
from .experiment_spec import ExperimentSpec


WORKFLOW_FORMAT = "molgap-release-workflow-v1"
_FIELDS = {
    "format", "spec_identity", "source_paths", "artifacts", "recipe_files",
    "initial_states", "required_modules", "entry_template", "kernel_metadata",
    "dataset_metadata", "pickle_inputs",
}


def _path(root: Path, value: str) -> Path:
    if type(value) is not str or not value:
        raise ValueError("Workflow paths must be explicit nonempty strings")
    path = Path(value)
    path = path if path.is_absolute() else root / path
    _safe_local(path)
    return path.absolute()


def release_workflow_inputs(spec, repo_root: Path, raw: str) -> dict:
    """Parse transport configuration, never infer a protocol or fabricate RML."""
    spec = _spec(spec)
    config = json.loads(raw, object_pairs_hook=_unique_object)
    if type(config) is not dict or set(config) != _FIELDS or config["format"] != WORKFLOW_FORMAT:
        raise ValueError("Missing/unknown release workflow fields or format")
    json_bytes(config)  # Reject NaN/Infinity before publishing prospective records.
    if config["spec_identity"] != spec.identity:
        raise ValueError("Release workflow differs from the frozen Spec")
    for name in ("source_paths", "required_modules", "pickle_inputs"):
        values = config[name]
        if type(values) is not list or any(type(value) is not str or not value for value in values):
            raise ValueError(f"Invalid workflow list: {name}")
        if name != "pickle_inputs" and not values:
            raise ValueError(f"Empty workflow list: {name}")
    for name in ("artifacts", "recipe_files", "initial_states"):
        if type(config[name]) is not dict:
            raise ValueError(f"Invalid workflow mapping: {name}")
    _allowlist(config["source_paths"])
    arms = {arm["arm_id"] for arm in spec.to_dict()["arms"]}
    if set(config["recipe_files"]) != arms or set(config["initial_states"]) - arms:
        raise ValueError("Workflow recipe/initialization arm bindings differ from Spec")
    for value in config["recipe_files"].values():
        if _name(value) not in config["source_paths"]:
            raise ValueError("Workflow recipes must be explicit packaged source")
    if any(type(name) is not str or name not in config["artifacts"] for name in config["initial_states"].values()):
        raise ValueError("Workflow initialization must name a staged artifact")
    if config["dataset_metadata"] is not None and type(config["dataset_metadata"]) is not dict:
        raise ValueError("Dataset metadata must be explicit object or null")
    artifacts = {}
    for name, value in config["artifacts"].items():
        _artifact_name(name)
        if type(value) is not dict or set(value) != {"path", "sha256"}:
            raise ValueError("Each artifact requires path and SHA256")
        _digest(value["sha256"], "workflow.artifact.sha256")
        artifacts[name] = UploadArtifact(_path(repo_root, value["path"]), value["sha256"])
    return {
        "relative_paths": config["source_paths"], "artifacts": artifacts,
        "recipe_files": config["recipe_files"], "initial_states": config["initial_states"],
        "required_modules": config["required_modules"],
        "entry_template": _path(repo_root, config["entry_template"]),
        "kernel_metadata": _path(repo_root, config["kernel_metadata"]),
        "dataset_metadata": config["dataset_metadata"],
        "pickle_inputs": [_path(repo_root, value) for value in config["pickle_inputs"]],
    }


def prepare_experiment_release(spec, repo_root: Path, workflow_raw: str, output: Path) -> tuple[dict, int]:
    """Plan all arms, assemble upload inputs and retain a single local receipt.

    A partial plan or a failed staging attempt is retained, never deleted or
    automatically replanned. Caller must reconcile before a retry. This does
    not select an account, POST a kernel, release compute or finalize evidence.
    """
    started = perf_counter()
    spec = _spec(spec)
    if spec.to_dict()["schema_version"] != SCHEMA_VERSION_V2:
        raise ValueError("prepare-release requires per-arm Spec v2 planning")
    repo_root, output = Path(repo_root).absolute(), Path(output).absolute()
    _safe_local(repo_root)
    _safe_local(output)
    inputs = release_workflow_inputs(spec, repo_root, workflow_raw)
    precheck = check_experiment_preparation(spec, repo_root, workflow_raw, output=output)
    timings = {"preparation_check_seconds": perf_counter() - started}
    output.mkdir(parents=True, exist_ok=False)
    result = {"spec_identity": spec.identity, "submitted": False, "compute_released": False,
              "preparation_check": precheck, "timings": timings}
    phase_started = perf_counter()
    phase = "prospective_publication_seconds"
    try:
        planned, code = plan_prospective(spec, repo_root)
        timings[phase] = perf_counter() - phase_started
        result["prospective"] = planned
        if code:
            result["status"] = "PLANNING_BLOCKED_RECONCILE_BEFORE_RETRY"
        else:
            phase, phase_started = "package_stage_release_check_seconds", perf_counter()
            staged = stage_release_inputs(spec, repo_root, output=output / "release", **inputs)
            timings[phase] = perf_counter() - phase_started
            result["staging"] = staged
            code = 1 if staged["errors"] else 0
            result["status"] = "LOCAL_PREPARATION_BLOCKED" if code else "LOCAL_PREPARATION_COMPLETE"
        timings["total_local_preparation_seconds"] = perf_counter() - started
        atomic_write(output / "workflow.json", json_bytes(result))
        return result, code
    except Exception as exc:
        timings[phase] = perf_counter() - phase_started
        timings["total_local_preparation_seconds"] = perf_counter() - started
        atomic_write(output / "workflow.json", json_bytes({
            **result, "status": "PREPARATION_ERROR_RECONCILE_BEFORE_RETRY", "error": str(exc),
        }))
        raise


def check_experiment_preparation(spec, repo_root: Path, workflow_raw: str, *,
                                output: Path | None = None) -> dict:
    """Read-only integrated input/plan check; not a replacement for check-release.

    Validate actual prospective semantics, selected source and upload file bytes
    before publishing RML. No cache loading, tensor deserialization, model
    execution or platform API calls occur; upload files are streamed for SHA.
    """
    started = perf_counter()
    spec = _spec(spec)
    if spec.to_dict()["schema_version"] != SCHEMA_VERSION_V2:
        raise ValueError("check-preparation requires per-arm Spec v2 planning")
    root = Path(repo_root).absolute()
    _safe_local(root)
    inputs = release_workflow_inputs(spec, root, workflow_raw)
    validate_staging_inputs(spec, root, inputs["relative_paths"],
        artifacts=inputs["artifacts"], recipe_files=inputs["recipe_files"],
        initial_states=inputs["initial_states"], entry_template=inputs["entry_template"],
        kernel_metadata=inputs["kernel_metadata"], output=output)
    names = _allowlist(inputs["relative_paths"])
    for name in names:
        path = _source(root.resolve(), name)
        if path.suffix == ".py":
            ast.parse(_payload(path), filename=name)
    for arm in spec.to_dict()["arms"]:
        arm_id = arm["arm_id"]
        recipe = _source(root.resolve(), inputs["recipe_files"][arm_id])
        if hashlib.sha256(_payload(recipe)).hexdigest() != arm["training"]["recipe"]["sha256"]:
            raise ValueError(f"Preparation recipe SHA differs from packaged LF bytes: {arm_id}")
        if arm["initialization"]["state_sha256"] is not None and arm_id not in inputs["initial_states"]:
            raise ValueError(f"Preparation missing pinned initialization binding: {arm_id}")
    for module in inputs["required_modules"]:
        base = "src/" + module.replace(".", "/")
        if base + ".py" not in names and base + "/__init__.py" not in names:
            raise ValueError(f"Preparation required module absent from source allowlist: {module}")
    missing = set(names) - _tracked_paths(root)
    if missing:
        raise ValueError(f"Preparation source is untracked: {sorted(missing)}")
    source_commit = _assert_clean_paths(root, names)
    prospective = check_prospective(spec, root)
    if prospective["batch"]["source_commit"] != source_commit:
        raise ValueError("HEAD changed during preparation check")
    return {"status": "LOCAL_PREPARATION_CHECKED_ONLY", "spec_identity": spec.identity,
            "source_commit": source_commit, "prospective": prospective,
            "upload_file_sha256": {name: artifact.sha256 for name, artifact in inputs["artifacts"].items()},
            "elapsed_seconds": perf_counter() - started,
            "published": False, "submitted": False, "compute_released": False,
            "release_report_required": True}


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
            initial_states=initial_states, jobs=jobs)
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
                    expected_package_identity, locations, execute=False):
    """Inspect every arm, build the existing descriptor, then delegate closure."""
    from .experiment_family_workflow import inspect_output
    outputs = prepare_terminal_outputs(spec, repo_root, supplied, receipt_path=receipt_path,
        package_dir=package_dir, expected_package_identity=expected_package_identity)
    reports = {arm_id: inspect_output(item["output_dir"], context=item["context"], expected=item["expected"])
               for arm_id, item in outputs.items()}
    if any(r["status"] == "BLOCKED" for r in reports.values()):
        return {"status": "BLOCKED", "arms": reports, "executed": False}
    descriptor = build_verified_terminal_descriptor(repo_root, spec, outputs=outputs, locations=locations)
    result = close_verified_outputs(repo_root, spec, descriptor, outputs=outputs, execute=execute)
    return {**result, "descriptor": descriptor.to_dict()}
