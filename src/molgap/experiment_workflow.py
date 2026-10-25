"""Compose existing prospective and release staging APIs, without a submitter."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .experiment_launch import _safe_local, build_launch_receipt
from .experiment_package import _allowlist, _name, _source, _spec, build_experiment_source_package
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


def _arm_inputs(repo_root, arms, bindings):
    """Bind worker inputs relative to the selected repository for both front doors."""
    if type(arms) is not list:
        raise ValueError("Expected explicit workflow arms")
    jobs, recipes, initial_states = [], {}, {}
    for arm in arms:
        if (type(arm) is not dict
                or set(arm) != {"arm_id", "device", "recipe", "initial_state"}
                or arm["arm_id"] not in bindings):
            raise ValueError("Expected known arm/device/recipe/initial_state")
        arm_id = arm["arm_id"]
        if arm_id in initial_states:
            raise ValueError("Duplicate workflow arm")
        initial = _path(repo_root, arm["initial_state"])
        if not initial.is_file():
            raise ValueError("Missing frozen initialization file")
        initial_states[arm_id], recipes[arm_id] = initial, arm["recipe"]
        jobs.append({"arm_id": arm_id, "device": arm["device"], "recipe": arm["recipe"],
                     "initial_state": f"initial_states/{arm_id}.pt",
                     "trajectory_id": bindings[arm_id]["trajectory_id"]})
    from .experiment_package import initial_state_paths
    paths = initial_state_paths(initial_states)
    for job in jobs:
        job["initial_state"] = paths[job["arm_id"]]
    return jobs, recipes, initial_states


def prepare_family_training(spec, repo_root, plan, output):
    """Stage the shared worker for any platform; skills add platform release/metadata.

    No reference, scientific acceptance or full-scale admission is inferred.
    Those owning gates remain prerequisites for an authorized job submission.
    """
    from .experiment_package import stage_source_inputs
    from .experiment_source_inventory import registered_source_files
    from .experiment_launch import publish_immutable_bytes
    from .research_memory.trace import file_digest
    from .family_training_runner import INTERFACES
    if (type(plan) is not dict or set(plan) != {"format", "spec_identity", "arms", "source_files"}
            or plan["format"] != "molgap-family-training-preparation-v1" or plan["spec_identity"] != spec.identity):
        raise ValueError("Shared preparation plan/Spec mismatch")
    repo_root, output = Path(repo_root).absolute(), Path(output).absolute()
    _safe_local(repo_root)
    _safe_local(output)
    if output.exists() or any(output.is_relative_to(repo_root / name) for name in ("experiments", "research_memory")):
        raise ValueError("Use a fresh ignored staging directory, not canonical research trees")
    declaration = spec.to_dict()
    if any((arm["family"]["name"], arm["family"]["version"]) not in INTERFACES.values() for arm in declaration["arms"]):
        raise ValueError("Shared preparation requires shared K1/GPTrans interfaces")
    bindings = {entry["arm_id"]: entry for entry in declaration["prospective"].get("arms", [])}
    jobs, recipes, initial_states = _arm_inputs(repo_root, plan["arms"], bindings)
    validated = validate_execution_plan(spec, jobs)
    check_family_recipes(spec, repo_root, recipes)
    source_files = _allowlist(sorted(set(registered_source_files(spec, plan["source_files"])) | set(recipes.values())))
    output.mkdir(parents=True)
    package = output / "package"
    manifest = build_experiment_source_package(spec, repo_root, source_files, package)
    staged = stage_source_inputs(package, output / "inputs", initial_states,
        initial_paths={job["arm_id"]: job["initial_state"] for job in jobs})
    release = check_release_inputs(spec, package, expected_package_identity=manifest["package_identity"],
        recipe_files=recipes, initial_states={job["arm_id"]: staged / job["initial_state"] for job in jobs},
        required_modules=sorted({module for job in validated for module in job["required_modules"]}
            | {"molgap.experiment_training_worker", "molgap.family_training_outputs"}), input_root=staged)
    _atomic(output / "release_report.json", release)
    if release["errors"]:
        return {"status": "BLOCKED", "release": release, "submitted": False, "prospective_published": False}
    prospective, code = plan_prospective(spec, repo_root, execution_source_commit=manifest["source_commit"])
    _atomic(output / "prospective_report.json", prospective)
    if code:
        return {"status": "RECONCILIATION_REQUIRED", "prospective": prospective, "submitted": False}
    hashes = {}
    for arm_id, binding in bindings.items():
        path = _artifact_path(repo_root, binding["output"] + "/trajectory.json")
        target = staged / "prospective" / arm_id / "trajectory.json"
        publish_immutable_bytes(target, path.read_bytes())
        hashes[arm_id] = file_digest(target)
    report = {"status": "SOURCE_INPUTS_PREPARED", "spec_identity": spec.identity,
        "expected_package_identity": manifest["package_identity"], "jobs": jobs,
        "prospective_sha256": hashes, "platform": declaration["platform"]["name"],
        "submitted": False, "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_EVALUATED"}
    _atomic(output / "prepared_training.json", report)
    return report


def _workflow_inputs(spec, repo_root, plan):
    """Resolve the same declaration for read-only admission and preparation."""
    platform_name = spec.to_dict()["platform"]["name"]
    backend = _platform_adapter(platform_name)
    fields = {"format", "spec_identity", "arms", "acceptance_plan", platform_name}
    runtime = type(plan) is dict and plan.get("source_profile") == "runtime-v1"
    fields |= {"source_profile", "source_records"} if runtime else {"source_files"}
    generated = runtime and "generated_records" in plan
    if generated:
        fields |= {"generated_records", "configuration_source"}
    if type(plan) is not dict or not fields <= set(plan) or set(plan) - fields - {"dataset_manifests"} or plan["format"] != "molgap-experiment-workflow-v1" or plan["spec_identity"] != spec.identity:
        raise ValueError("Workflow plan/Spec mismatch")
    if type(plan["source_records"] if runtime else plan["source_files"]) is not list:
        raise ValueError("Workflow source_files must be an explicit path list")
    repo_root = Path(repo_root).absolute()
    _safe_local(repo_root)
    metadata = backend.validate_plan(spec, plan[platform_name])
    declaration = spec.to_dict()
    bindings = {b["arm_id"]: b for b in declaration["prospective"].get("arms", [])}
    jobs, recipes, initial_states = _arm_inputs(repo_root, plan["arms"], bindings)
    validated_jobs = validate_execution_plan(spec, jobs)
    dataset_manifests = plan.get("dataset_manifests", {})
    if type(dataset_manifests) is not dict or set(dataset_manifests) - set(recipes):
        raise ValueError("Dataset manifest bindings must name known workflow arms")
    dataset_manifests = {arm: str(_artifact_or_local_path(repo_root, value))
                         for arm, value in dataset_manifests.items()}
    acceptance_path = _artifact_path(repo_root, plan["acceptance_plan"])
    acceptance = check_acceptance_plan(spec, repo_root, _json(acceptance_path), require_training_targets=True)
    if acceptance["status"] == "BLOCKED":
        return {"status": "BLOCKED", "stage": "acceptance_plan", "acceptance": acceptance,
                "prospective_published": False, "submitted": False}
    # The inventory is maintained with adapters, not assembled by each caller.
    from .experiment_source_inventory import registered_source_files
    if runtime:
        from .experiment_source_inventory import runtime_source_files
        records = [*plan["source_records"], *recipes.values()]
        if generated:
            records += [plan["acceptance_plan"], *(b["plan_spec_ref"] for b in bindings.values())]
        derived = plan.get("generated_records", {})
        if generated:
            from .experiment_package import _generation_bindings
            _generation_bindings(derived, plan["configuration_source"], [*records, *derived])
            if any(hashlib.sha256(_payload(_source(repo_root.resolve(), n))).hexdigest() != h
                   for n, h in derived.items()):
                raise ValueError("Generated configuration record bytes changed")
        source_files = sorted(set(runtime_source_files(repo_root, [p for p in records if p not in derived])) | set(derived))
    else:
        source_files = registered_source_files(spec, [*plan["source_files"], *recipes.values()])
    return dict(metadata=metadata, bindings=bindings, jobs=jobs, initial_states=initial_states,
                recipes=recipes, validated_jobs=validated_jobs, source_files=source_files,
                dataset_manifests=dataset_manifests)


def _artifact_or_local_path(repo_root, value):
    if type(value) is not str:
        raise ValueError("Expected a local dataset manifest path")
    path = Path(value)
    path = path if path.is_absolute() else Path(repo_root) / path
    _safe_local(path.absolute())
    if not path.is_file():
        raise ValueError("Missing frozen dataset manifest")
    return path.absolute()


def _workflow_checks(spec, repo_root, inputs):
    from .experiment_execution import training_adapter
    def manifest_bindings():
        required = {arm["arm_id"] for arm in spec.to_dict()["arms"]
                    if training_adapter(arm).module == "molgap.family_training_runner"}
        if required - set(inputs["dataset_manifests"]):
            raise ValueError("Shared family release requires explicit dataset_manifests bindings")
        return {"bound_arms": sorted(inputs["dataset_manifests"])}
    observations, blockers = {}, []
    for name, check in (("family_recipes", lambda: check_family_recipes(spec, repo_root, inputs["recipes"])),
                        ("dataset_manifests", manifest_bindings),
                        ("prospective", lambda: check_prospective(spec, Path(repo_root).absolute()))):
        try:
            observations[name] = check()
        except (ValueError, TypeError, RuntimeError, OSError) as error:
            blockers.append({"check": name, "error_type": type(error).__name__, "message": str(error)})
    return observations, blockers


def check_workflow_readiness(spec, repo_root, plan):
    """Check recipes, profilers, acceptance and RML policy before any writes."""
    from .experiment_execution import workflow_capabilities
    started = perf_counter()
    inputs = _workflow_inputs(spec, repo_root, plan)
    if inputs.get("status") == "BLOCKED":
        return inputs
    observations, blockers = _workflow_checks(spec, repo_root, inputs)
    return {"status": "BLOCKED" if blockers else "WORKFLOW_INPUTS_READY", "spec_identity": spec.identity,
            "checks": observations, "blockers": blockers, "capabilities": workflow_capabilities(spec),
            "elapsed_seconds": perf_counter() - started,
            "published": False, "submitted": False, "native_qualification": "NOT_EVALUATED"}


def prepare_workflow(spec: ExperimentSpec, repo_root: Path, plan: dict, output: Path, *, source_cache=None, _content_verification=None, _verification=None) -> dict:
    """Prepare one frozen declaration with no platform submission."""
    from .experiment_package import _PackageVerification

    verification = _verification if _verification is not None else _PackageVerification()
    started = perf_counter()
    repo_root, output = Path(repo_root).absolute(), Path(output).absolute()
    _safe_local(output)
    if output.exists():
        raise ValueError("Workflow output must be a fresh directory; reconcile before another attempt")
    backend = _platform_adapter(spec.to_dict()["platform"]["name"])
    platform_name = spec.to_dict()["platform"]["name"]
    inputs = _workflow_inputs(spec, repo_root, plan)
    if inputs.get("status") == "BLOCKED":
        return inputs
    metadata, bindings, jobs, initial_states, recipes, validated_jobs, source_files = (
        inputs[key] for key in ("metadata", "bindings", "jobs", "initial_states", "recipes",
                               "validated_jobs", "source_files"))
    # All cheap owner/policy checks precede the staging directory and package.
    observations, blockers = _workflow_checks(spec, repo_root, inputs)
    if blockers:
        return {"status": "BLOCKED", "stage": "workflow_readiness", "checks": observations,
                "blockers": blockers, "prospective_published": False, "submitted": False}
    output.mkdir(parents=True)
    report = {"format": "molgap-workflow-preparation-v1", "status": "PREPARING",
              "spec_identity": spec.identity, "submitted": False, "stages": [],
              "prospective_published": False, "publication_intent": False,
              "timings": {"input_validation_seconds": perf_counter() - started}}
    phase_started = perf_counter()
    failure = None
    def record(stage, result):
        nonlocal phase_started
        report["timings"][stage + "_seconds"] = perf_counter() - phase_started
        phase_started = perf_counter()
        report["stages"].append({"stage": stage, "result": result})
    try:
        record("family_recipes", observations["family_recipes"])
        record("prospective_check", observations["prospective"])
        package = output / "package"
        options = {"_verification": verification}
        if source_cache is not None:
            options["source_cache"] = source_cache
        if "generated_records" in plan:
            options.update(generated_records={n: _payload(_source(repo_root.resolve(), n))
                for n in plan["generated_records"]}, configuration_source=plan["configuration_source"])
        manifest = build_experiment_source_package(spec, repo_root, source_files, package, **options)
        record("package", manifest)
        stage = backend.stage_inputs(repo_root=repo_root, output=output, package=package,
            manifest=manifest, spec=spec, platform_plan=plan[platform_name], metadata=metadata,
            initial_states=initial_states, jobs=jobs, acceptance_plan_path=plan["acceptance_plan"],
            _verification=verification)
        staged, kernel = stage["input_root"], stage["kernel_dir"]
        release = check_release_inputs(spec, package,
            expected_package_identity=manifest["package_identity"], recipe_files=recipes,
            initial_states={job["arm_id"]: staged / job["initial_state"] for job in stage["launch"]["jobs"]},
            required_modules=sorted({module for j in validated_jobs for module in j["required_modules"]} |
                                    {"molgap.experiment_training_worker"}),
            input_root=staged, dataset_manifests=inputs["dataset_manifests"] or None,
            _verification=verification, _content_verification=_content_verification)
        _atomic(output / "release_report.json", release)
        record("check_release", release)
        if release["errors"]:
            report.update(status="BLOCKED", stage="check_release", prospective_published=False)
            return report
        report["prospective_published"] = "inspect_plan_result"
        report["publication_intent"] = True
        # Persist uncertainty BEFORE crossing the irreversible RML boundary.
        _atomic(output / "workflow_report.json", report)
        prospective, code = plan_prospective(spec, repo_root, execution_source_commit=manifest["source_commit"])
        report["prospective_published"] = True if not code else "inspect_plan_result"
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
        record("launch_receipt", build_launch_receipt(spec, package,
            expected_package_identity=manifest["package_identity"], _verification=verification))
        report.update(status="PREPARED_FOR_PLATFORM", prospective_published=True,
                      package_identity=manifest["package_identity"], source_commit=manifest["source_commit"],
                      release_report=str(output / "release_report.json"), kernel_dir=str(kernel),
                      source_dataset_dir=str(staged), accelerator=stage["accelerator"],
                      limitations=["Local preparation only; workload skill publishes and submits.",
                                   "Remote all-arm runtime qualification is still mandatory."])
        return report
    except Exception as exc:
        failure = exc
        status = "ERROR_BEFORE_PUBLICATION" if report["prospective_published"] is False else "ERROR_REQUIRES_RECONCILIATION"
        report.update(status=status, error=str(exc))
        raise
    except BaseException as exc:
        failure = exc
        raise
    finally:
        report["timings"]["total_local_preparation_seconds"] = perf_counter() - started
        try:
            # Only the pre-publication intent and terminal state are recovery
            # boundaries. Intermediate stages stay in this one final report.
            _atomic(output / "workflow_report.json", report)
        except OSError as report_error:
            if failure is None:
                raise
            # A secondary storage fault must not replace the causal failure.
            # The last durable intent (or missing report) still blocks retry.
            failure.workflow_report_error = report_error


def accept_workflow(spec, repo_root, supplied, *, receipt_path, package_dir,
                    expected_package_identity, locations=None, execute=False,
                    target_identities=None, execution_root=None, _verification=None,
                    _reports=None, _retention=None):
    """Inspect every arm independently of optional scientific closure inputs."""
    if execute and locations is None:
        raise ValueError("Terminal execution requires explicit reviewed terminal locations")
    from .experiment_family_workflow import inspect_terminal_output
    outputs = prepare_terminal_outputs(spec, repo_root, supplied, receipt_path=receipt_path,
        package_dir=package_dir, expected_package_identity=expected_package_identity,
        **({"_verification": _verification} if _verification is not None else {}))
    if target_identities is not None:
        if type(target_identities) is not dict or set(target_identities) - set(outputs):
            raise ValueError("Target identity bindings name unknown workflow arms")
        for arm_id, binding in target_identities.items():
            outputs[arm_id]["target_identity"] = binding
    retention = _retention
    if execution_root is not None and retention is None:
        from .experiment_retention import validate_execution_retention
        context = next(iter(outputs.values()))["context"]
        retention = validate_execution_retention(Path(execution_root), spec,
            package_identity=expected_package_identity, source_commit=context.source_commit,
            source_archive_sha256=context.source_archive_sha256)
    # The run adapter can share its inspection inside one synchronous closure.
    # Descriptor translation and finalization still verify the retained bytes.
    reports = (_reports if _reports is not None else
               {arm_id: inspect_terminal_output(item) for arm_id, item in outputs.items()})
    if any(r["status"] == "BLOCKED" for r in reports.values()):
        return {"status": "BLOCKED", "arms": reports, "executed": False}
    if locations is None:
        result = {"status": "MECHANICALLY_VERIFIED", "arms": reports, "executed": False,
                  "scientific_outcome": "NOT_EVALUATED"}
        if retention is not None:
            result["execution_retention"] = {"status": retention["status"], "manifest": retention["manifest"]}
        return result
    descriptor = build_verified_terminal_descriptor(repo_root, spec, outputs=outputs, locations=locations,
                                                     _reports=reports)
    result = close_verified_outputs(repo_root, spec, descriptor, outputs=outputs, execute=execute,
                                    _reports=reports)
    result = {**result, "descriptor": descriptor.to_dict()}
    if retention is not None:
        result["execution_retention"] = {"status": retention["status"], "manifest": retention["manifest"]}
    return result
