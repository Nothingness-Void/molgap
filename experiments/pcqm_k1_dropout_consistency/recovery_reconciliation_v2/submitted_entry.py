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
    child_env = dict(os.environ, MOLGAP_RECOVERY_CHILD="1", CUDA_VISIBLE_DEVICES="0", PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    subprocess.run([sys.executable, __file__], env=child_env, check=True)
    raise SystemExit(0)

# Local preparation replaces this marker and binds the resulting entry bytes.
EXPECTED_LAUNCH_SHA256 = 'eaa02224ee89156ab7b9642055b0cfa53d8724c2e58c48a4a5332104b27cd2a6'


"""Restore one frozen K1 arm; the family owner implements every training step."""
from pathlib import Path
import hashlib
import json
import shutil
import time


def resume_one_arm(*, source_root, package_dir, input_root, launch_path, output):
    import torch
    from molgap.experiment_execution import execute_training_phase, validate_execution_plan
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
    if binding["arm_id"] != "dropout_mean2" or binding["start_epoch"] != 39 or binding["end_epoch"] != 40:
        raise ValueError("Recovery is restricted to the authorized missing final epoch")
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
    if (saved["cursor"]["epoch"] != 39 or saved["cursor"]["next_batch"] != 0
        or saved["optimizer_step"] != 30459 or saved["sample_presentations"] != 3898752
        or len(rows) != 39 or rows[-1]["checkpoint_identity"] != "sha256:" + binding["files"]["last_checkpoint.pt"]):
        raise ValueError("Checkpoint and acknowledged trace disagree")
    del saved
    # The original GPU assignment and full runtime identity must survive recovery.
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("Recovery requires the frozen Kaggle T4 allocation")
    options = dict(spec=spec, job=job, source_root=source_root, package_dir=package_dir,
        expected_package_identity=launch["expected_package_identity"], input_root=input_root,
        output=destination, account=launch["account"], run_reference=launch["run_reference"],
        staged_root=Path(launch_path).parent, prospective_sha256=launch["prospective_sha256"][job["arm_id"]])
    preflight = execute_training_phase(phase="preflight", **options)
    if preflight.get("status") != "accepted" and preflight.get("accepted") is not True:
        raise ValueError("Recovery runtime qualification blocked")
    runtime = json.loads((destination / "runtime_manifest.json").read_text())
    if runtime["runtime_fingerprint"] != binding["runtime_fingerprint"]:
        raise ValueError("Recovery runtime differs from the original segment")
    started = time.perf_counter()
    result = execute_training_phase(phase="train", **options)
    atomic_json(destination / "mechanical_inspection.json", result)
    manifest = json.loads((destination / "output_manifest.json").read_text())
    if manifest["progress"] != {"epochs": 40, "optimizer_steps": 31240, "sample_presentations": 3998720}:
        raise ValueError("Recovery did not complete the frozen exposure")
    # Preserve the known validator failure as a failure, not a scientific acceptance.
    if result["status"] != "MECHANICALLY_VERIFIED" and result.get("blockers") != ["Development target identity mismatch"]:
        raise ValueError("Unexpected recovered-output acceptance blocker: " + repr(result.get("blockers")))
    atomic_json(Path(output) / "recovery_state.json", {
        "status": "TRAINING_COMPLETE", "acceptance_status": result["status"],
        "acceptance_blockers": result["blockers"], "arm_id": job["arm_id"],
        "original_run": binding["original_run"], "original_version": 1,
        "progress": manifest["progress"], "additional_epochs": 1,
        "assigned_device_seconds": time.perf_counter() - started,
        "device_cost_scope": "one assigned T4 recovery invocation including model/data loading; prior segment excluded",
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
