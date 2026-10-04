"""Pinned source bootstrap; disposable profiling only, never a training release."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import traceback
import types

SOURCE_SHA256 = "__PIN_SOURCE_ARCHIVE_SHA256__"


def main():
    began = time.time()
    archives = [p for p in Path("/kaggle/input").rglob("source_payload.bin")
                if hashlib.sha256(p.read_bytes()).hexdigest() == SOURCE_SHA256]
    if len(archives) != 1:
        raise ValueError("One hash-qualified profile source mount required")
    inputs = archives[0].parent
    with tarfile.open(archives[0], "r:gz") as bundle:
        extractor = bundle.extractfile("src/molgap/experiment_preflight.py").read()
    bootstrap = types.ModuleType("_frozen_profile_bootstrap")
    bootstrap.__file__ = "experiment_preflight.py"
    exec(compile(extractor, bootstrap.__file__, "exec"), bootstrap.__dict__)
    source = Path("/kaggle/working/profile_verified_source")
    package = Path("/kaggle/working/profile_verified_package")
    package.mkdir(exist_ok=False)
    import shutil
    shutil.copyfile(archives[0], package / "source.tar.gz")
    for name in ("experiment_spec.json", "package_manifest.json", "SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
        shutil.copyfile(inputs / name, package / name)
    bootstrap._unpack(package, source)
    sys.path.insert(0, str(source / "src"))
    from molgap.training_reproducibility import atomic_json
    from molgap.experiment_package import verify_experiment_source_package
    identity = verify_experiment_source_package(package)
    contract = json.loads((source / "experiments/pcqm_gptrans_scale_qualification/contract.json").read_text())
    output = Path("/kaggle/working/gptrans_scale_qualification")
    inventory = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip().splitlines()
    deadline = began + contract["maximum_wall_seconds"]
    atomic_json(output / "startup.json", {"package": identity, "allocation": inventory,
        "contract": contract, "started_unix": began, "deadline_unix": deadline,
        "single_arm_reason": "Two EMA filters share one live model during disposable qualification; no duplicate encoder is justified."})
    status = "ERROR"
    try:
        if not inventory or not all("T4" in device for device in inventory):
            raise RuntimeError("T4 allocation required")
        from molgap.kaggle_python_environment import prepare_python
        executable, environment = prepare_python(Path("/kaggle/working/profile_env"), deadline=deadline, source_root=source)
        atomic_json(output / "environment.json", environment)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONPATH="", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="42", OMP_NUM_THREADS="2")
        subprocess.run([str(executable), str(source / "experiments/pcqm_gptrans_scale_qualification/run.py"),
            "--inputs", str(inputs), "--output", str(output), "--source-package", str(package)],
            env=env, check=True, timeout=max(1, deadline - time.time() - 20))
        status = "COMPLETE"
    except BaseException:
        atomic_json(output / "failure.json", {"error": traceback.format_exc()})
        raise
    finally:
        elapsed = time.time() - began
        atomic_json(output / "cost.json", {"status": status, "allocated_devices": len(inventory),
            "used_devices": 1, "allocation_wall_seconds": elapsed,
            "allocated_device_hours": elapsed * len(inventory) / 3600,
            "scope": "kernel entry through worker; queue/teardown unavailable"})
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name != "output_manifest.json"}
        atomic_json(output / "output_manifest.json", {"status": status, "source_identity": identity,
            "files": files, "scientific_training_executed": False, "development_role_read": False,
            "official_validation_role_read": False, "test_dev_role_read": False, "test_challenge_role_read": False})


if __name__ == "__main__":
    main()
