"""Local recovery preparation over an original immutable workflow release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import ast
import subprocess
import re

from .experiment_family_workflow import RunContext, _artifact_path, _check_context, _json
from .experiment_launch import _safe_local
from .experiment_launch_config import validate_launch_config, RECOVERY_REPAIR_PATHS, validate_recovery_repair
from .experiment_package import verify_experiment_source_package, read_packaged_text
from .experiment_preflight import check_release_inputs, _atomic
from .experiment_resume import RESUME_BUNDLE_FILE, create_resume_bundle, validate_resume_bundle
from .experiment_workflow import _platform_adapter, _path


RESUME_PLAN_FORMAT = "molgap-workflow-resume-v1"


def _repair_payloads(spec, root, package, manifest, declaration):
    if declaration is None:
        return None, {}
    if (type(declaration) is not dict or set(declaration) != {"commit", "files"}
            or type(declaration["commit"]) is not str or re.fullmatch(r"[0-9a-f]{40}", declaration["commit"]) is None
            or type(declaration["files"]) is not dict or set(declaration["files"]) != RECOVERY_REPAIR_PATHS):
        raise ValueError("Expected a pinned three-file infrastructure repair")
    payloads = {}
    for name, binding in declaration["files"].items():
        original = read_packaged_text(package, name).encode()
        payload = subprocess.run(["git", "show", declaration["commit"] + ":" + name],
            cwd=root, check=True, capture_output=True).stdout
        from .v4_runtime import normalized_source_sha256
        if (type(binding) is not dict or set(binding) != {"original_sha256", "repaired_sha256"}
                or hashlib.sha256(original).hexdigest() != binding["original_sha256"]
                or hashlib.sha256(payload).hexdigest() != binding["repaired_sha256"]):
            raise ValueError("Committed recovery repair differs from its source pins")
        if normalized_source_sha256(Path(root) / name) != binding["repaired_sha256"]:
            raise ValueError("Local recovery validator differs from the committed repair")
        compile(payload, name, "exec")
        if name.endswith("family_training_runner.py"):
            def without_verifier(raw):
                tree = ast.parse(raw)
                nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_loaded_source"]
                if len(nodes) != 1:
                    raise ValueError("Expected exactly one frozen source verifier")
                nodes[0].body = [ast.Pass()]
                return ast.dump(tree, include_attributes=False)
            if without_verifier(original) != without_verifier(payload):
                raise ValueError("Runner repair may change only _loaded_source")
        payloads[name] = payload
    return {"format": "molgap-recovery-repair-v1", "spec_identity": spec.identity,
            "package_identity": manifest["package_identity"], "archive_sha256": manifest["archive_sha256"],
            **declaration}, payloads


def _original(spec, repo_root, prepared, package_dir, expected_package_identity, receipt_path, recovery_repair=None):
    root, prepared, package = map(lambda p: Path(p).absolute(), (repo_root, prepared, package_dir))
    for path in (root, prepared, package):
        _safe_local(path)
    report = _json(prepared / "workflow_report.json")
    if (report.get("status") != "PREPARED_FOR_PLATFORM" or report.get("spec_identity") != spec.identity
            or report.get("package_identity") != expected_package_identity):
        raise ValueError("Recovery requires the original prepared workflow and package")
    manifest = verify_experiment_source_package(package)
    if manifest["package_identity"] != expected_package_identity or manifest["spec_identity"] != spec.identity:
        raise ValueError("Recovery package identity mismatch")
    if "src/molgap/experiment_resume.py" not in manifest["relative_allowlist"]:
        raise ValueError("Original frozen package predates portable recovery support")
    _, repairs = _repair_payloads(spec, root, package, manifest, recovery_repair)
    release = _json(prepared / "release_report.json")
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
        dataset_manifests=inputs.get("dataset_manifests"))
    if observed != release or observed["errors"] or observed["status"] != "LOCAL_RELEASE_INPUTS_VERIFIED":
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
        packaged_owner = read_packaged_text(package, "src/" + owner_name.replace(".", "/") + ".py")
        expected_owner = repairs.get("src/" + owner_name.replace(".", "/") + ".py", packaged_owner.encode())
        if normalized_source_sha256(Path(owner.__file__)) != hashlib.sha256(expected_owner).hexdigest():
            raise ValueError("Local resume validator differs from the frozen family owner")
        arm_id = arm["arm_id"]
        observed_context = RunContext.from_launch(spec, Path(receipt_path), package,
            expected_package_identity=expected_package_identity, arm_id=arm_id)
        producer = RunContext.for_training(spec, package, expected_package_identity=expected_package_identity,
            arm_id=arm_id, account=launch["account"], run_reference=launch["run_reference"])
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
                          receipt_path, output, *, arm_id, source_output, recovery_repair=None):
    """Export one incomplete owner-validated arm; never publish a new attempt."""
    manifest, _, _, contexts, trajectories = _original(spec, repo_root, prepared, package_dir,
                                                        expected_package_identity, receipt_path, recovery_repair)
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
                             receipt_path, output, *, plan):
    """Stage all incomplete arms using the original source and prospective bytes."""
    manifest, release, launch, contexts, trajectories = _original(spec, repo_root, prepared, package_dir,
                                                                expected_package_identity, receipt_path,
                                                                plan.get("recovery_repair") if type(plan) is dict else None)
    if (type(plan) is not dict or set(plan) - {"recovery_repair", "allocation_wall_limit_seconds", "cleanup_reserve_seconds"} != {"format", "spec_identity", "arms"}
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
    metadata = _json(Path(release["inputs"]["kernel_metadata"]))
    staged_original = Path(release["inputs"]["launch_config"]).parent
    dataset = _json(staged_original / "dataset-metadata.json")
    platform_plan = {"account": launch["account"], "kernel": launch["run_reference"],
        "title": metadata["title"], "datasets": launch["dataset_sources"],
        "source_dataset": dataset["id"], "accelerator": launch["accelerator"]}
    for key in ("allocation_wall_limit_seconds", "cleanup_reserve_seconds"):
        if key in launch:
            platform_plan[key] = launch[key]
        if key in plan:
            value = plan[key]
            if type(value) is not int or value < (120 if key == "allocation_wall_limit_seconds" else 0):
                raise ValueError("Invalid recovery allocation budget")
            original_limit = launch.get(key)
            if key == "allocation_wall_limit_seconds" and original_limit is not None and value > original_limit:
                raise ValueError("Recovery allocation ceiling cannot increase")
            platform_plan[key] = value
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
            initial_states={k: Path(v) for k, v in release["inputs"]["initial_states"].items()}, jobs=launch["jobs"])
        repair, payloads = _repair_payloads(spec, repo_root, package_dir, manifest, plan.get("recovery_repair"))
        if repair is not None:
            if spec.to_dict()["platform"]["name"] != "kaggle":
                raise ValueError("Infrastructure recovery repair supports only the reviewed Kaggle bootstrap")
            entry = subprocess.run(["git", "show", repair["commit"] + ":platforms/kaggle/run_experiment.py"],
                cwd=repo_root, check=True, capture_output=True).stdout.decode()
            marker = "EXPECTED_RECOVERY_REPAIR_SHA256 = None"
            if entry.count(marker) != 1 or entry.count("EXPECTED_LAUNCH_SHA256 = None") != 1:
                raise ValueError("Repair commit lacks the reviewed recovery bootstrap")
            for name, payload in payloads.items():
                target = stage["input_root"] / "recovery_repair" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            _atomic(stage["input_root"] / "recovery_repair.json", repair)
            digest = hashlib.sha256((stage["input_root"] / "recovery_repair.json").read_bytes()).hexdigest()
            stage["entry_text"] = entry.replace(marker, "EXPECTED_RECOVERY_REPAIR_SHA256 = " + repr(digest))
            report["recovery_repair"] = {"manifest_sha256": digest, **repair}
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
        if repair is not None:
            validate_recovery_repair(stage["input_root"], expected_sha256=digest, manifest=manifest,
                                    package_dir=package_dir, entry_script=stage["entry_path"])
        _unchanged_prospectives(spec, repo_root, trajectories)
        result = check_release_inputs(spec, Path(package_dir), expected_package_identity=expected_package_identity,
            recipe_files=release["inputs"]["recipe_files"],
            initial_states={a: stage["input_root"] / "initial_states" / (a + ".pt") for a in contexts},
            required_modules=release["inputs"]["required_modules"], input_root=stage["input_root"],
            entry_script=stage["entry_path"], launch_config=stage["launch_path"],
            kernel_metadata=stage["metadata_path"],
            dataset_manifests=release["inputs"].get("dataset_manifests"))
        _atomic(output / "release_report.json", result)
        report.update(status="BLOCKED" if result["errors"] else "PREPARED_FOR_PLATFORM",
            source_commit=manifest["source_commit"], release_report=str(output / "release_report.json"),
            kernel_dir=str(stage["kernel_dir"]), source_dataset_dir=str(stage["input_root"]),
            accelerator=stage["accelerator"], errors=result["errors"],
            limitations=["All arms must have incomplete owner-validated checkpoints.",
                         "Workload skill must reconcile the prior launch before submitting a successor."])
        return report
    except Exception as exc:
        report.update(status="ERROR", error=str(exc))
        raise
    finally:
        _atomic(output / "workflow_report.json", report)
