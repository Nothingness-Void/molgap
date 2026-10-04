"""Standard Kaggle source bootstrap. The staged launch config owns the arms."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import types

import os
if os.environ.get("MOLGAP_RECOVERY_CHILD") != "1":
    child_env = dict(os.environ, MOLGAP_RECOVERY_CHILD="1", CUDA_VISIBLE_DEVICES="1", PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    subprocess.run([sys.executable, __file__], env=child_env, check=True, timeout=1900)
    raise SystemExit(0)

# Local preparation replaces this marker and binds the resulting entry bytes.
EXPECTED_LAUNCH_SHA256 = 'a17e49002a73c80acb6666f902925af2a8bc1d18820af56a681ba45de69e4120'


"""Restore one frozen K1 arm; the family owner implements every training step."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import os
import time


def resume_one_arm(*, source_root, package_dir, input_root, launch_path, output):
    import torch
    from molgap.experiment_execution import validate_execution_plan
    from molgap.experiment_family_workflow import RunContext, inspect_output, TargetIdentityBinding
    from molgap.experiment_spec import ExperimentSpec
    from molgap.training_reproducibility import atomic_json

    launch = json.loads(Path(launch_path).read_text())
    recovery = Path(launch_path).parent / "recovery"
    binding = json.loads((recovery / "resume_binding.json").read_text())
    spec = ExperimentSpec.from_json((Path(package_dir) / "experiment_spec.json").read_text())
    if binding["spec_identity"] != spec.identity or binding["package_identity"] != launch["expected_package_identity"]:
        raise ValueError("Recovery source/spec identity mismatch")
    jobs = validate_execution_plan(spec, launch["jobs"])
    job = next(j for j in jobs if j["arm_id"] == binding["arm_id"])
    recipe = json.loads((Path(source_root) / job["recipe"]).read_text())
    expected = recipe["acceptance_requirements"]
    start, end = binding["start_epoch"], expected["epochs"]
    if job["module"] != "molgap.k1_screen_training" or type(start) is not int or not 0 < start < end or binding["end_epoch"] != end:
        raise ValueError("Recovery requires an authorized incomplete K1 epoch cursor")
    context = RunContext.for_training(spec, Path(package_dir),
        expected_package_identity=launch["expected_package_identity"], arm_id=job["arm_id"],
        account=launch["account"], run_reference=launch["run_reference"])
    if binding["original_run"] != context.run_reference:
        raise ValueError("Recovery original run identity mismatch")
    target_pin = binding["target_identity"]
    if target_pin != launch.get("target_identity"):
        raise ValueError("Recovery target identity differs from launch binding")
    target_identity = TargetIdentityBinding.from_acceptance_plan(spec, Path(launch_path).parent / "acceptance",
        target_pin["plan_path"], plan_sha256=target_pin["plan_sha256"])
    if len(binding["files"]) != 5 or set(binding["files"]) != {
        "last_checkpoint.pt", "selected_model.pt", "development_predictions.pt",
        "canonical_trace.json", "canonical_trace.json.context.json"}:
        raise ValueError("Incomplete recovery retention")
    destination = Path(output) / binding["arm_id"]
    destination.mkdir(parents=True, exist_ok=False)
    for name, pin in binding["files"].items():
        path = recovery / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != pin:
            raise ValueError("Recovery artifact hash mismatch: " + name)
        shutil.copyfile(path, destination / name)
    saved = torch.load(destination / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    trace = json.loads((destination / "canonical_trace.json").read_text())
    rows = [r for r in trace["observations"] if r["event"] == "observation"]
    if (saved["context"] != context.to_dict() or saved["cursor"]["epoch"] != start or saved["cursor"]["next_batch"] != 0
        or saved["cursor"]["sampler_order_sha256"] != recipe["row_order_fingerprint"]
        or saved["optimizer_step"] != start * (expected["optimizer_steps"] // end)
        or saved["sample_presentations"] != start * (expected["sample_presentations"] // end)
        or len(rows) != start or [r["epoch_or_pass"] for r in rows] != list(range(1, start + 1))
        or rows[-1]["checkpoint_identity"] != "sha256:" + binding["files"]["last_checkpoint.pt"]):
        raise ValueError("Checkpoint and acknowledged trace disagree")
    predictions = torch.load(destination / "development_predictions.pt", map_location="cpu", weights_only=True)
    if target_identity.digest(predictions["target_eV"], context=context, expected=expected)[0] != expected["target_sha256"]:
        raise ValueError("Recovery original target identity mismatch")
    del predictions
    del saved
    # The original GPU assignment and full runtime identity must survive recovery.
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("Recovery requires the frozen Kaggle T4 allocation")
    # Qualification mutates process state. Reuse the standard isolated worker
    # for BOTH phases, exactly as the original pair runtime does.
    def worker(phase):
        command = [sys.executable, "-m", "molgap.experiment_training_worker",
            "--source-root", str(source_root), "--package-dir", str(package_dir),
            "--input-root", str(input_root), "--output", str(destination),
            "--launch", str(launch_path), "--arm", job["arm_id"], "--phase", phase]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job["device"]), PYTHONHASHSEED="42",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONPATH=str(Path(source_root) / "src"))
        with (destination / (phase + ".log")).open("w", encoding="utf-8") as log:
            return subprocess.run(command, cwd=source_root, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=900).returncode

    if worker("preflight") != 0:
        raise ValueError("Recovery runtime qualification blocked")
    runtime = json.loads((destination / "runtime_manifest.json").read_text())
    if runtime["runtime_fingerprint"] != binding["runtime_fingerprint"]:
        raise ValueError("Recovery runtime differs from the original segment")
    started = time.perf_counter()
    returncode = worker("train")
    default_result = inspect_output(destination, context=context, expected=expected)
    result = inspect_output(destination, context=context, expected=expected, target_identity=target_identity)
    atomic_json(destination / "original_completion_inspection.json", default_result)
    atomic_json(destination / "mechanical_inspection.json", result)
    manifest = json.loads((destination / "output_manifest.json").read_text())
    if manifest["progress"] != {key: expected[key] for key in ("epochs", "optimizer_steps", "sample_presentations")}:
        raise ValueError("Recovery did not complete the frozen exposure")
    if result["status"] != "MECHANICALLY_VERIFIED":
        raise ValueError("Unexpected recovered-output acceptance blocker: " + repr(result.get("blockers")))
    expected_returncode = 0 if default_result["status"] == "MECHANICALLY_VERIFIED" else 1
    if default_result["status"] != "MECHANICALLY_VERIFIED" and default_result.get("blockers") != ["Development target identity mismatch"]:
        raise ValueError("Unexpected original completion failure")
    if returncode != expected_returncode:
        raise ValueError("Unexpected training worker exit")
    atomic_json(Path(output) / "recovery_state.json", {
        "status": "TRAINING_COMPLETE", "acceptance_status": result["status"],
        "acceptance_blockers": result["blockers"], "arm_id": job["arm_id"],
        "original_run": binding["original_run"], "original_version": binding["original_version"],
        "progress": manifest["progress"], "additional_epochs": end - start,
        "assigned_device_seconds": time.perf_counter() - started,
        "device_cost_scope": "one assigned T4 recovery invocation including model/data loading and post-training CPU inspection; prior segment excluded; allocation time, not device busy time",
        "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_EVALUATED"})


def main():
    mounted = Path("/kaggle/input")
    configs = list(mounted.rglob("experiment_launch.json"))
    if len(configs) != 1:
        raise RuntimeError("Expected one frozen experiment_launch.json source mount")
    launch = configs[0]
    if EXPECTED_LAUNCH_SHA256 is None or hashlib.sha256(launch.read_bytes()).hexdigest() != EXPECTED_LAUNCH_SHA256:
        raise RuntimeError("Launch configuration differs from the prepared entrypoint")
    config = json.loads(launch.read_text(encoding="utf-8"))
    if hashlib.sha256((launch.parent / "recovery/resume_binding.json").read_bytes()).hexdigest() != config["recovery_binding_sha256"]:
        raise RuntimeError("Recovery binding changed after release")
    archive = launch.parent / "source_payload.bin"
    with archive.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != config["expected_source_archive_sha256"]:
        raise RuntimeError("Frozen source archive hash mismatch")
    # Dependencies are platform bootstrap, never imported from another experiment.
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy<2",
                    "torch-geometric==2.6.1", "ogb==1.3.6"], check=True)
    root = Path("/kaggle/temp/molgap-workflow")
    root.mkdir(parents=True, exist_ok=False)
    package, source = root / "package", root / "source"
    package.mkdir()
    source.mkdir()
    shutil.copyfile(archive, package / "source.tar.gz")
    for name in ("experiment_spec.json", "package_manifest.json", "SOURCE_COMMIT.txt",
                 "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
        shutil.copyfile(launch.parent / name, package / name)
    with tarfile.open(archive, "r:gz") as bundle:
        stream = bundle.extractfile("src/molgap/experiment_preflight.py")
        if stream is None:
            raise RuntimeError("Shared source extractor absent")
        with stream:
            code = stream.read()
    bootstrap = types.ModuleType("_frozen_source_bootstrap")
    bootstrap.__file__ = str(archive) + ":experiment_preflight.py"
    exec(compile(code, bootstrap.__file__, "exec"), bootstrap.__dict__)
    bootstrap._unpack(package, source)
    sys.path.insert(0, str(source / "src"))
    from molgap.experiment_package import verify_experiment_source_package
    from molgap.kaggle_pair_runtime import run_two_phase_pair
    manifest = verify_experiment_source_package(package)
    if manifest["package_identity"] != config["expected_package_identity"] or manifest["spec_identity"] != config["spec_identity"]:
        raise RuntimeError("Frozen launch/Spec/package binding mismatch")
    resume_one_arm(source_root=source, package_dir=package, input_root=mounted,
                       launch_path=launch, output=Path("/kaggle/working/experiment"))


if __name__ == "__main__":
    main()
