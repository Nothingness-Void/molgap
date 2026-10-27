"""Local Kaggle preparation adapter; remote operations remain in the workload skill."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil

from .experiment_family_workflow import _artifact_path
from .experiment_launch import canonical_json, publish_immutable_bytes
from .experiment_launch_config import validate_allocation_budget
from .experiment_package import read_packaged_text

def _metadata(kaggle: dict) -> dict:
    fields = {"account", "kernel", "title", "datasets", "source_dataset", "accelerator"}
    optional = {"allocation_wall_limit_seconds", "cleanup_reserve_seconds"}
    if type(kaggle) is not dict or not fields <= set(kaggle) or set(kaggle) - (fields | optional):
        raise ValueError("Expected explicit Kaggle account/kernel/title/datasets/source_dataset/accelerator")
    validate_allocation_budget(kaggle)
    if not re.fullmatch(r"[a-z0-9-]+", kaggle["account"]):
        raise ValueError("Invalid Kaggle account")
    prefix = kaggle["account"] + "/"
    slug = re.sub(r"[^a-z0-9]+", "-", kaggle["title"].lower()).strip("-")
    if not 5 <= len(kaggle["title"]) <= 50 or kaggle["kernel"] != prefix + slug:
        raise ValueError("Kernel title-generated slug must match the requested kernel")
    if (type(kaggle["datasets"]) is not list or not kaggle["datasets"] or
        len(set(kaggle["datasets"])) != len(kaggle["datasets"]) or
        any(not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", d) for d in kaggle["datasets"]) or
        not kaggle["source_dataset"].startswith(prefix) or
        kaggle["source_dataset"] not in kaggle["datasets"]):
        raise ValueError("Expected unique dataset mounts including the account-owned source dataset")
    if kaggle["accelerator"] != "NvidiaTeslaT4":
        raise ValueError("This platform adapter supports the explicit T4 allocation only")
    return {"id": kaggle["kernel"], "title": kaggle["title"], "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": kaggle["datasets"], "competition_sources": [], "kernel_sources": []}



def validate_plan(spec, plan):
    metadata = _metadata(plan)
    platform = spec.to_dict()["platform"]
    if platform["name"] != "kaggle" or platform["accelerator"] not in {"Tesla T4", "T4", "NvidiaTeslaT4"} or platform["device_count"] != 2:
        raise ValueError("Kaggle T4 requires its declared two-device allocation")
    return metadata


def stage_inputs(*, repo_root, output, package, manifest, spec, platform_plan,
                 metadata, initial_states, jobs, acceptance_plan_path=None, _verification=None):
    validate_allocation_budget(platform_plan)
    staged = output / "source_dataset"
    from .experiment_package import stage_source_inputs
    stage_source_inputs(package, staged, initial_states,
        initial_paths={job["arm_id"]: job["initial_state"] for job in jobs},
        **({"_verification": _verification} if _verification is not None else {}))
    acceptance_binding = None
    if acceptance_plan_path is not None:
        acceptance_binding = stage_acceptance_inputs(spec, repo_root, acceptance_plan_path, staged / "acceptance")
    publish_immutable_bytes(staged / "dataset-metadata.json", canonical_json({
        "id": platform_plan["source_dataset"], "title": platform_plan["source_dataset"].split("/")[1],
        "licenses": [{"name": "CC0-1.0"}]}).encode())
    kernel = output / "kernel"
    kernel.mkdir()
    marker = "EXPECTED_LAUNCH_SHA256 = None"
    entry_text = read_packaged_text(package, "platforms/kaggle/run_experiment.py")
    if entry_text.count(marker) != 1:
        raise ValueError("Expected one reviewed launch digest marker in the platform entry")
    publish_immutable_bytes(kernel / "kernel-metadata.json", canonical_json(metadata).encode())
    launch = {"format": "molgap-execution-launch-v1", "spec_identity": spec.identity,
        "expected_package_identity": manifest["package_identity"],
        "expected_source_archive_sha256": manifest["archive_sha256"],
        "account": platform_plan["account"], "run_reference": platform_plan["kernel"],
        "dataset_sources": metadata["dataset_sources"], "accelerator": platform_plan["accelerator"],
        "device_count": spec.to_dict()["platform"]["device_count"], "jobs": jobs}
    for field in ("allocation_wall_limit_seconds", "cleanup_reserve_seconds"):
        if field in platform_plan:
            launch[field] = platform_plan[field]
    if acceptance_binding is not None:
        launch["target_identity"] = acceptance_binding
    return {"input_root": staged, "kernel_dir": kernel, "entry_path": kernel / "run.py",
        "metadata_path": kernel / "kernel-metadata.json", "launch_path": staged / "experiment_launch.json",
        "entry_text": entry_text, "accelerator": platform_plan["accelerator"],
        "launch": launch}


def acceptance_input_pins(spec, repo_root, plan_path):
    """Return the validated allowlist shared by staging and publication."""
    from .experiment_family_workflow import _json, check_acceptance_plan, acceptance_dependency_pins
    from .research_memory.trace import file_digest
    root = Path(repo_root).absolute()
    path = _artifact_path(root, plan_path)
    plan = _json(path)
    report = check_acceptance_plan(spec, root, plan)
    if report["status"] != "ACCEPTANCE_INPUTS_AVAILABLE":
        raise ValueError("Producer acceptance inputs are blocked")
    def load_artifact(pointer):
        source = _artifact_path(root, pointer["path"])
        if file_digest(source) != pointer["sha256"]:
            raise ValueError("Acceptance artifact hash mismatch")
        return _json(source)
    pointers = acceptance_dependency_pins(plan, artifact_loader=load_artifact)
    digest = file_digest(path)
    if pointers.setdefault(plan_path, digest) != digest:
        raise ValueError("Conflicting acceptance artifact pins")
    return pointers


def stage_acceptance_inputs(spec, repo_root, plan_path, destination):
    from .research_memory.trace import file_digest
    root, destination = Path(repo_root).absolute(), Path(destination)
    pointers = acceptance_input_pins(spec, root, plan_path)
    for relative, digest in pointers.items():
        source = _artifact_path(root, relative)
        if file_digest(source) != digest:
            raise ValueError("Acceptance inputs changed during staging")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return {"plan_path": plan_path, "plan_sha256": pointers[plan_path]}


def freeze_inputs(stage, trajectories):
    config = stage["launch"]
    config["prospective_sha256"] = {}
    for arm_id, trajectory in trajectories.items():
        destination = stage["input_root"] / "prospective" / arm_id
        destination.mkdir(parents=True)
        # Publish and hash one observation; recovery supplies already captured bytes.
        raw = trajectory if type(trajectory) is bytes else Path(trajectory).read_bytes()
        publish_immutable_bytes(destination / "trajectory.json", raw)
        config["prospective_sha256"][arm_id] = hashlib.sha256(raw).hexdigest()
    publish_immutable_bytes(stage["launch_path"], canonical_json(config).encode())
    entry_text = stage["entry_text"].replace("EXPECTED_LAUNCH_SHA256 = None",
        "EXPECTED_LAUNCH_SHA256 = " + repr(hashlib.sha256(stage["launch_path"].read_bytes()).hexdigest()))
    publish_immutable_bytes(stage["entry_path"], entry_text.encode())


def bind_release(stage, spec, release):
    from .experiment_preflight import check_workflow_binding
    release["inputs"].update(entry_script=str(stage["entry_path"]),
        launch_config=str(stage["launch_path"]), kernel_metadata=str(stage["metadata_path"]))
    release["checks"]["entry_script:kernel"] = hashlib.sha256(stage["entry_path"].read_bytes()).hexdigest()
    release["checks"]["workflow:launch"] = check_workflow_binding(spec,
        launch_config=stage["launch_path"], kernel_metadata=stage["metadata_path"],
        entry_script=stage["entry_path"], package_dir=Path(release["inputs"]["package"]),
        expected_package_identity=release["package_identity"],
        recipe_files=release["inputs"]["recipe_files"],
        initial_states=release["inputs"]["initial_states"], input_root=stage["input_root"])


def recover_prepared_workflow(prepared, *, operations=None, _verification=None):
    """Finish local binding without republishing the frozen inputs."""
    import json
    from copy import deepcopy
    from .common.identity import validate_execution_identity
    from .experiment_family_workflow import _json, acceptance_dependency_pins
    from .experiment_launch import _safe_local, build_launch_receipt
    from .experiment_package import _PackageVerification, verify_experiment_source_package
    from .experiment_preflight import _atomic
    from .experiment_spec import ExperimentSpec
    from .research_memory.schemas import validate_trajectory
    from .research_memory.paths import repo_local_path

    prepared = Path(prepared).absolute()
    _safe_local(prepared)
    report_path = prepared / "workflow_report.json"
    report = _json(report_path)
    # Completed recovery is read-only, including when platform operations started.
    if report.get("status") == "PREPARED_FOR_PLATFORM":
        return report
    owner = prepared.with_name(prepared.name + ".submission_owner.json")
    journals = [prepared / "submission_state.json",
                prepared.parent / "operations" / "submission_state.json"]
    if operations is not None:
        journals.append(Path(operations).absolute() / "submission_state.json")
    for path in (owner, *journals):
        _safe_local(path)
        if path.exists():
            raise ValueError("Reconcile the existing submission owner/journal before preparation recovery")
    if (report.get("format") != "molgap-workflow-preparation-v1"
            or report.get("status") != "ERROR_REQUIRES_RECONCILIATION"
            or report.get("prospective_published") is not True
            or report.get("publication_intent") is not True
            or report.get("submitted") is not False):
        raise ValueError("Recovery requires a failed preparation after complete prospective publication")
    stages = {item["stage"]: item["result"] for item in report["stages"]}
    prospective = stages.get("prospective", {})
    if prospective.get("status") != "PROSPECTIVE_PLANNED_RML_PENDING":
        raise ValueError("Partial or uncertain prospective publication cannot be recovered")
    verification = _verification if _verification is not None else _PackageVerification()
    package = prepared / "package"
    manifest = verify_experiment_source_package(package, _verification=verification)
    spec = ExperimentSpec.from_json((package / "experiment_spec.json").read_text(encoding="utf-8"))
    if (report["spec_identity"] != spec.identity or stages.get("package") != manifest
            or spec.to_dict()["platform"]["name"] != "kaggle"):
        raise ValueError("Recovery differs from the original frozen package")
    staged, kernel = prepared / "source_dataset", prepared / "kernel"
    launch_path = staged / "experiment_launch.json"
    launch = _json(launch_path)
    if (launch.get("spec_identity") != spec.identity
            or launch.get("expected_package_identity") != manifest["package_identity"]
            or launch.get("expected_source_archive_sha256") != manifest["archive_sha256"]
            or "resume" in launch):
        raise ValueError("Recovery launch differs from the original frozen package")
    bindings = spec.to_dict()["prospective"]["arms"]
    expected_records = [(b["trajectory_id"], b["output"]) for b in bindings]
    records = prospective.get("completed_records", [])
    if (len(records) != len(bindings)
            or [(r.get("trajectory_id"), r.get("path")) for r in records] != expected_records
            or any(r.get("status") != "PLANNED" for r in records)
            or set(launch.get("prospective_sha256", {})) != {b["arm_id"] for b in bindings}):
        raise ValueError("Incomplete prospective publication bindings")
    for binding in bindings:
        path = staged / "prospective" / binding["arm_id"] / "trajectory.json"
        _safe_local(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != launch["prospective_sha256"][binding["arm_id"]]:
            raise ValueError("Staged prospective bytes changed")
        trajectory = validate_trajectory(_json(path))
        if (trajectory["record_mode"] != "prospective"
                or trajectory["trajectory_id"] != binding["trajectory_id"]):
            raise ValueError("Invalid staged prospective identity")
        validate_execution_identity(spec, binding["arm_id"], trajectory, manifest["source_commit"])
    original = deepcopy(stages.get("check_release", {}))
    if original.get("status") != "LOCAL_RELEASE_INPUTS_VERIFIED" or original.get("errors") != []:
        raise ValueError("Original release checks did not pass")
    if (original.get("spec_identity") != spec.identity
            or original.get("package_identity") != manifest["package_identity"]
            or original.get("source_commit") != manifest["source_commit"]
            or original.get("source_archive_sha256") != manifest["archive_sha256"]
            or Path(original["inputs"]["package"]).resolve() != package.resolve()):
        raise ValueError("Original release identity differs from the verified package")
    target = launch.get("target_identity", {})
    plan_name, plan_digest = target["plan_path"], target["plan_sha256"]
    plan_raw = read_packaged_text(package, plan_name, _verification=verification).encode("utf-8")
    if hashlib.sha256(plan_raw).hexdigest() != plan_digest:
        raise ValueError("Frozen acceptance plan differs from launch pin")
    def load_artifact(pointer):
        raw = read_packaged_text(package, pointer["path"], _verification=verification)
        if hashlib.sha256(raw.encode("utf-8")).hexdigest() != pointer["sha256"]:
            raise ValueError("Frozen package artifact differs from its plan pin")
        return json.loads(raw)
    pins = acceptance_dependency_pins(json.loads(plan_raw), artifact_loader=load_artifact)
    pins[plan_name] = plan_digest
    missing = {}
    for relative, digest in pins.items():
        destination = repo_local_path(staged / "acceptance", relative)
        _safe_local(destination)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise ValueError("Existing acceptance attachment bytes changed: " + relative)
        else:
            raw = read_packaged_text(package, relative, _verification=verification).encode("utf-8")
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Frozen package acceptance attachment differs: " + relative)
            missing[relative] = raw
    recovery = {"stage": "preparation_recovery", "result": {
        "status": "RECOVERING", "original_status": report["status"],
        "original_error": report.get("error"), "restored_acceptance_paths": sorted(missing)}}
    report["stages"].append(recovery)
    _atomic(report_path, report)
    try:
        for relative, raw in missing.items():
            publish_immutable_bytes(repo_local_path(staged / "acceptance", relative), raw)
        release = deepcopy(original)
        stage = {"entry_path": kernel / "run.py", "launch_path": launch_path,
                 "metadata_path": kernel / "kernel-metadata.json", "input_root": staged}
        bind_release(stage, spec, release)
        receipt = build_launch_receipt(spec, package,
            expected_package_identity=manifest["package_identity"], _verification=verification)
        _atomic(prepared / "release_report.json", release)
        report["stages"].extend([{"stage": "final_release_binding", "result": release},
                                 {"stage": "launch_receipt", "result": receipt}])
        recovery["result"]["status"] = "RECOVERED"
        report.pop("error", None)
        report.update(status="PREPARED_FOR_PLATFORM", package_identity=manifest["package_identity"],
            source_commit=manifest["source_commit"], release_report=str(prepared / "release_report.json"),
            kernel_dir=str(kernel), source_dataset_dir=str(staged), accelerator=launch["accelerator"],
            limitations=["Local preparation only; workload skill publishes and submits.",
                         "Remote all-arm runtime qualification is still mandatory."])
    except Exception as error:
        recovery["result"].update(status="RECOVERY_FAILED", error=str(error))
        raise
    finally:
        _atomic(report_path, report)
    # Reuse the existing retirement owner for any earlier unpublished attempts.
    # A successful recovery must not require another preparation invocation.
    if prepared.name == "prepared" and (prepared.parent / "local_preparation_failures").exists():
        from .local_storage import retire_failed_preparations
        try:
            report["retired_failed_preparations"] = retire_failed_preparations(
                prepared.parent, spec.identity, manifest["package_identity"])
        except (OSError, ValueError):
            report["retirement_status"] = "PENDING_INSPECTION"
        _atomic(report_path, report)
    return report
