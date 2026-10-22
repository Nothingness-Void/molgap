"""Local recovery preparation over an original immutable workflow release."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from .experiment_family_workflow import RunContext, _artifact_path, _check_context, _json
from .experiment_launch import _safe_local
from .experiment_launch_config import validate_launch_config
from .experiment_package import _PackageVerification, verify_experiment_source_package, read_packaged_text
from .experiment_preflight import check_release_inputs, _atomic
from .experiment_resume import RESUME_BUNDLE_FILE, create_resume_bundle, validate_resume_bundle
from .experiment_workflow import _platform_adapter, _path


RESUME_PLAN_FORMAT = "molgap-workflow-resume-v1"


def _rebase_paths(value, source, destination):
    if isinstance(value, dict):
        return {key: _rebase_paths(item, source, destination) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebase_paths(item, source, destination) for item in value]
    if isinstance(value, str) and (value == str(source) or value.startswith(str(source) + os.sep)):
        return str(destination) + value[len(str(source)):]
    return value


@contextmanager
def _original(spec, repo_root, prepared, package_dir, expected_package_identity, receipt_path, *, _verification):
    """Keep retired release copies in a private view only for this operation."""
    root, prepared, package = (Path(path).absolute() for path in (repo_root, prepared, package_dir))
    for path in (root, prepared, package, Path(receipt_path).absolute()):
        _safe_local(path)
    release = _json(prepared / "release_report.json")
    inputs = release["inputs"]
    staged = Path(inputs["input_root"])
    _safe_local(staged)
    missing = {"source_payload.bin"} if not (staged / "source_payload.bin").is_file() else set()
    missing.update(Path(path).relative_to(staged).as_posix() for path in inputs["initial_states"].values()
                   if not Path(path).is_file())
    if not missing:
        yield _check_original(spec, root, prepared, package, expected_package_identity, receipt_path,
                              _verification=_verification)
        return
    if staged != prepared / "source_dataset":
        raise ValueError("Retired recovery inputs must belong to the original source stage")
    # Retirement preserves the upload's exact bytes and their durable owners.
    # An arbitrary absent input is never permission to regenerate initialization.
    operations = Path(receipt_path).absolute().parent
    state = _json(operations / "submission_state.json")
    retirement = state.get("source_copy_retirement", {})
    manifest = verify_experiment_source_package(package, _verification=_verification)
    launch = _json(Path(inputs["launch_config"]))
    proof = _json(operations / "source_upload_verification.json")
    uploaded = retirement.get("uploaded", {})
    if (state.get("phase") != "BOUND" or state.get("binding", {}).get("package_identity") != expected_package_identity
            or manifest["package_identity"] != expected_package_identity or manifest["spec_identity"] != spec.identity
            or state["binding"].get("kernel") != launch["run_reference"]
            or proof.get("status") != "REMOTE_SOURCE_BYTES_VERIFIED" or proof.get("version") != 1
            or proof.get("dataset") != state["binding"].get("dataset")
            or retirement.get("status") not in {"PENDING", "BLOCKED", "REMOVED"}
            or not missing <= set(uploaded)
            or retirement.get("roundtrip") != {item["path"]: {key: item[key] for key in ("sha256", "bytes")}
                                               for item in proof["files"]}):
        raise ValueError("Missing recovery inputs require the original bound source retirement records")
    workflow = _json(prepared.parent / "inputs/workflow_plan.json")
    binding = _json(prepared.parent / "inputs/input_binding.json")
    from .screen_policy import canonical_fingerprint
    if (workflow.get("spec_identity") != spec.identity or binding.get("spec_identity") != spec.identity
            or canonical_fingerprint(workflow) != binding.get("workflow_identity")):
        raise ValueError("Recovery requires the original frozen workflow initialization pointers")
    declarations = {entry["arm_id"]: entry for entry in workflow["arms"]}
    sources = {"source_payload.bin": package / "source.tar.gz"}
    for arm_id, path in inputs["initial_states"].items():
        sources[Path(path).relative_to(staged).as_posix()] = _path(root, declarations[arm_id]["initial_state"])
    from .common.io import copy_immutable_file
    from .training_reproducibility import sha256_file
    with TemporaryDirectory(prefix=".resume-inputs-", dir=prepared.parent) as temporary:
        view = Path(temporary) / "source_dataset"
        # Retained metadata stays read-only. Hardlinks avoid copying unrelated
        # retained resume payloads; no producer path is modified by this view.
        for path in staged.rglob("*"):
            _safe_local(path)
        shutil.copytree(staged, view, copy_function=os.link)
        for relative in sorted(missing):
            record = uploaded[relative]
            if relative not in sources or record != retirement["roundtrip"].get(relative):
                raise ValueError("Retired recovery input differs from its bound upload")
            source = sources[relative]
            _safe_local(source)
            if not source.is_file() or source.stat().st_size != record["bytes"] or sha256_file(source) != record["sha256"]:
                raise ValueError("Original frozen recovery input is missing or changed: " + str(source))
            copy_immutable_file(source, view / relative, expected_sha256=record["sha256"])
        restored_release = _rebase_paths(release, staged, view)
        yield _check_original(spec, root, prepared, package, expected_package_identity, receipt_path,
                              release=restored_release, original_staged=staged, _verification=_verification)


def _check_original(spec, repo_root, prepared, package_dir, expected_package_identity, receipt_path,
                    *, release=None, original_staged=None, _verification):
    root, prepared, package = map(lambda p: Path(p).absolute(), (repo_root, prepared, package_dir))
    for path in (root, prepared, package):
        _safe_local(path)
    report = _json(prepared / "workflow_report.json")
    if (report.get("status") != "PREPARED_FOR_PLATFORM" or report.get("spec_identity") != spec.identity
            or report.get("package_identity") != expected_package_identity):
        raise ValueError("Recovery requires the original prepared workflow and package")
    manifest = verify_experiment_source_package(package, _verification=_verification)
    if manifest["package_identity"] != expected_package_identity or manifest["spec_identity"] != spec.identity:
        raise ValueError("Recovery package identity mismatch")
    if "src/molgap/experiment_resume.py" not in manifest["relative_allowlist"]:
        raise ValueError("Original frozen package predates portable recovery support")
    frozen_release = _json(prepared / "release_report.json")
    release = frozen_release if release is None else release
    inputs = release["inputs"]
    if inputs["pickle_inputs"]:
        raise ValueError("Recovery of extra pickled input staging requires the owning preparation adapter")
    if Path(inputs["package"]).resolve() != package.resolve():
        raise ValueError("Recovery release points to another package")
    observed = check_release_inputs(spec, package, expected_package_identity=expected_package_identity,
        recipe_files=inputs["recipe_files"], initial_states=inputs["initial_states"],
        required_modules=inputs["required_modules"], pickle_inputs=inputs["pickle_inputs"],
        entry_script=inputs["entry_script"], input_root=inputs["input_root"],
        launch_config=inputs["launch_config"], kernel_metadata=inputs["kernel_metadata"],
        dataset_manifests=inputs.get("dataset_manifests"), _verification=_verification)
    compared = (_rebase_paths(observed, Path(inputs["input_root"]), original_staged)
                if original_staged is not None else observed)
    if compared != frozen_release or observed["errors"] or observed["status"] != "LOCAL_RELEASE_INPUTS_VERIFIED":
        raise ValueError("Original release changed or did not pass verification")
    staged = Path(inputs["launch_config"]).parent
    validate_launch_config(spec, Path(inputs["launch_config"]), manifest=manifest,
                           package_dir=package, staged_root=staged)
    launch_raw = Path(inputs["launch_config"]).read_bytes()
    if hashlib.sha256(launch_raw).hexdigest() != observed["checks"]["workflow:launch"]["launch_sha256"]:
        raise ValueError("Original launch changed after verification")
    launch = json.loads(launch_raw)
    contexts, trajectories = {}, {}
    for arm in spec.to_dict()["arms"]:
        from importlib import import_module
        from .experiment_execution import training_adapter
        from .v4_runtime import normalized_source_sha256
        owner_name = training_adapter(arm).module
        owner = import_module(owner_name)
        packaged_owner = read_packaged_text(package, "src/" + owner_name.replace(".", "/") + ".py",
                                           _verification=_verification)
        if normalized_source_sha256(Path(owner.__file__)) != hashlib.sha256(packaged_owner.encode()).hexdigest():
            raise ValueError("Local resume validator differs from the frozen family owner")
        arm_id = arm["arm_id"]
        observed_context = RunContext.from_launch(spec, Path(receipt_path), package,
            expected_package_identity=expected_package_identity, arm_id=arm_id, _verification=_verification)
        producer = RunContext.for_training(spec, package, expected_package_identity=expected_package_identity,
            arm_id=arm_id, account=launch["account"], run_reference=launch["run_reference"], _verification=_verification)
        _check_context(producer.to_dict(), observed_context)
        binding = next(b for b in spec.to_dict()["prospective"]["arms"] if b["arm_id"] == arm_id)
        trajectory = _artifact_path(root, binding["output"] + "/trajectory.json")
        frozen = staged / "prospective" / arm_id / "trajectory.json"
        raw = trajectory.read_bytes()
        if (raw != frozen.read_bytes()
                or hashlib.sha256(raw).hexdigest() != launch["prospective_sha256"][arm_id]
                or json.loads(raw)["decision"]["outcome"] != "ACTIVE"):
            raise ValueError("Recovery requires the exact original ACTIVE canonical prospective trajectory")
        contexts[arm_id], trajectories[arm_id] = producer, raw
    return manifest, release, launch, contexts, trajectories


def _unchanged_prospectives(spec, repo_root, trajectories):
    for binding in spec.to_dict()["prospective"]["arms"]:
        path = _artifact_path(Path(repo_root).absolute(), binding["output"] + "/trajectory.json")
        if path.read_bytes() != trajectories[binding["arm_id"]]:
            raise ValueError("Canonical prospective trajectory changed during recovery preparation")


def build_workflow_resume(spec, repo_root, prepared, package_dir, expected_package_identity,
                          receipt_path, output, *, arm_id, source_output, _verification=None):
    """Export one incomplete owner-validated arm; never publish a new attempt."""
    verification = _verification or _PackageVerification()
    with _original(spec, repo_root, prepared, package_dir,
                   expected_package_identity, receipt_path, _verification=verification) as original:
        manifest, _, _, contexts, trajectories = original
        if arm_id not in contexts:
            raise ValueError("Unknown recovery arm")
        output = Path(output).absolute()
        _safe_local(output)
        if output.exists():
            raise ValueError("Resume export requires a fresh destination")
        bundle = create_resume_bundle(Path(source_output), contexts[arm_id], trajectories[arm_id], output)
        checked = validate_resume_bundle(output, spec, arm_id=arm_id,
                                        context=contexts[arm_id], trajectory=trajectories[arm_id])
        _unchanged_prospectives(spec, repo_root, trajectories)
        return {"status": checked["status"], "spec_identity": spec.identity,
                "package_identity": manifest["package_identity"], "arm_id": arm_id,
                "manifest": str(output / RESUME_BUNDLE_FILE),
                "manifest_sha256": hashlib.sha256((output / RESUME_BUNDLE_FILE).read_bytes()).hexdigest(),
                "cursor": bundle["cursor"], "prospective_published": False, "submitted": False}


def prepare_resumed_workflow(spec, repo_root, prepared, package_dir, expected_package_identity,
                             receipt_path, output, *, plan, _verification=None):
    """Stage all incomplete arms using the original source and prospective bytes."""
    verification = _verification or _PackageVerification()
    with _original(spec, repo_root, prepared, package_dir,
                   expected_package_identity, receipt_path, _verification=verification) as original:
        manifest, release, launch, contexts, trajectories = original
        if (type(plan) is not dict or set(plan) != {"format", "spec_identity", "arms"}
                or plan["format"] != RESUME_PLAN_FORMAT or plan["spec_identity"] != spec.identity
                or type(plan["arms"]) is not dict or set(plan["arms"]) != set(contexts)):
            raise ValueError("Resume plan must bind every Spec arm exactly once")
        checked = {}
        for arm_id, path in plan["arms"].items():
            checked[arm_id] = validate_resume_bundle(_path(Path(repo_root), path), spec,
                arm_id=arm_id, context=contexts[arm_id], trajectory=trajectories[arm_id])
        output = Path(output).absolute()
        _safe_local(output)
        if output.exists():
            raise ValueError("Resume preparation requires a fresh output directory")
        backend = _platform_adapter(spec.to_dict()["platform"]["name"])
        frozen_inputs = _json(Path(prepared) / "release_report.json")["inputs"]
        metadata = _json(Path(release["inputs"]["kernel_metadata"]))
        staged_original = Path(release["inputs"]["launch_config"]).parent
        dataset = _json(staged_original / "dataset-metadata.json")
        platform_plan = {"account": launch["account"], "kernel": launch["run_reference"],
            "title": metadata["title"], "datasets": launch["dataset_sources"],
            "source_dataset": dataset["id"], "accelerator": launch["accelerator"]}
        backend.validate_plan(spec, platform_plan)
        output.mkdir(parents=True)
        report = {"format": "molgap-workflow-recovery-preparation-v1", "status": "PREPARING",
                  "spec_identity": spec.identity, "package_identity": expected_package_identity,
                  "original_prepared": str(Path(prepared).absolute()),
                  "original_receipt": str(Path(receipt_path).absolute()),
                  "prospective_published": False, "submitted": False}
        try:
            stage = backend.stage_inputs(repo_root=Path(repo_root), output=output, package=Path(package_dir),
                manifest=manifest, spec=spec, platform_plan=platform_plan, metadata=metadata,
                initial_states={k: Path(v) for k, v in release["inputs"]["initial_states"].items()}, jobs=launch["jobs"],
                _verification=verification)
            if "target_identity" in launch:
                # Retain the original acceptance-input copy and binding on recovery.
                source = staged_original / "acceptance"
                if not source.is_dir():
                    raise ValueError("Original same-run acceptance inputs are missing")
                shutil.copytree(source, stage["input_root"] / "acceptance")
                stage["launch"]["target_identity"] = launch["target_identity"]
            stage["launch"]["resume"] = {}
            for arm_id, item in checked.items():
                destination = stage["input_root"] / "resume" / arm_id
                destination.mkdir(parents=True)
                pointers = [p for p in item["manifest"]["artifacts"].values() if p is not None]
                for group in ("runtime", "provenance", "context", "cost_segments"):
                    pointers.extend(item["manifest"][group])
                for relative in {RESUME_BUNDLE_FILE, *(p["path"] for p in pointers)}:
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(item["root"] / relative, target)
                stage["launch"]["resume"][arm_id] = {
                    "manifest": f"resume/{arm_id}/{RESUME_BUNDLE_FILE}",
                    "sha256": hashlib.sha256((destination / RESUME_BUNDLE_FILE).read_bytes()).hexdigest()}
            backend.freeze_inputs(stage, trajectories)
            _unchanged_prospectives(spec, repo_root, trajectories)
            result = check_release_inputs(spec, Path(package_dir), expected_package_identity=expected_package_identity,
                recipe_files=frozen_inputs["recipe_files"],
                initial_states={a: stage["input_root"] / "initial_states" / (a + ".pt") for a in contexts},
                required_modules=release["inputs"]["required_modules"], input_root=stage["input_root"],
                entry_script=stage["entry_path"], launch_config=stage["launch_path"],
                kernel_metadata=stage["metadata_path"],
                dataset_manifests=frozen_inputs.get("dataset_manifests"), _verification=verification)
            _atomic(output / "release_report.json", result)
            report.update(status="BLOCKED" if result["errors"] else "RESUME_INPUTS_PREPARED",
                source_commit=manifest["source_commit"], release_report=str(output / "release_report.json"),
                kernel_dir=str(stage["kernel_dir"]), source_dataset_dir=str(stage["input_root"]),
                accelerator=stage["accelerator"], errors=result["errors"],
                next_action="The owning continuation adapter must bind a physical successor identity; retain the original source, configuration and prospective records.",
                limitations=["All arms must have incomplete owner-validated checkpoints.",
                             "Workload skill must reconcile the prior launch before submitting a successor."])
            return report
        except Exception as exc:
            report.update(status="ERROR", error=str(exc))
            raise
        finally:
            _atomic(output / "workflow_report.json", report)
