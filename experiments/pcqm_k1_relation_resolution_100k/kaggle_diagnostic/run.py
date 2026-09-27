"""Budgeted parallel frozen inference; no model execution in the parent."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
OUT = Path("/kaggle/working/pcqm_k1_relation_diagnostic")


def one(name):
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name}, found {len(matches)}")
    return matches[0]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def unpack(blob, target):
    with tarfile.open(blob, "r:gz") as archive:
        archive.extractall(target, filter="data")


def worker(mode):
    spec = json.loads(one("DIAGNOSTIC_RELEASE.json").read_text())
    helper = one("diagnostic_impl.py")
    if sha(helper) != spec["helper_sha256"]:
        raise ValueError("Diagnostic helper changed")
    sys.path.insert(0, str(OUT / "source/src"))
    module_spec = importlib.util.spec_from_file_location("frozen_relation_diagnostic", helper)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    caches = {}
    for role, digest in spec["manifest_sha256"].items():
        found = [p.parent for p in Path("/kaggle/input").rglob("manifest.json") if sha(p) == digest]
        if len(found) != 1:
            raise ValueError("Fixed cache missing or ambiguous")
        caches[role] = str(found[0])
    module.run_worker(OUT / "inputs", caches, OUT / "workers" / mode, spec, mode)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
        return
    started = time.monotonic()
    spec = json.loads(one("DIAGNOSTIC_RELEASE.json").read_text())
    if spec["training_authorized"] is not False:
        raise ValueError("NO_TRAIN release required")
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()
    if not devices or len(devices) > 2:
        raise ValueError("Unsupported diagnostic allocation")
    OUT.mkdir(parents=True, exist_ok=False)
    result = {"run_id": spec["run_id"], "complete": False, "training_executed": False,
        "optimizer_steps": 0, "allocated_devices": devices, "used_device_count": len(devices),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "spec": spec}
    deadline = started + min(spec["max_wall_seconds"], spec["max_allocated_device_seconds"] / len(devices))
    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Diagnostic allocation exhausted")
        return value
    def queue(device, modes):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS="2")
        for mode in modes:
            subprocess.run([sys.executable, __file__, "--worker", mode], env=env,
                           check=True, timeout=remaining())
    try:
        source, inputs, helper = one("source_payload.bin"), one("diagnostic_inputs.bin"), one("diagnostic_impl.py")
        if (sha(source) != spec["source_archive_sha256"] or sha(inputs) != spec["input_archive_sha256"]
            or sha(helper) != spec["helper_sha256"] or one("SOURCE_COMMIT.txt").read_text().strip() != spec["model_source_commit"]):
            raise ValueError("Frozen model source/diagnostic inputs changed")
        unpack(source, OUT / "source")
        for item in json.loads(one("SOURCE_FILES.json").read_text())["files"]:
            if sha(OUT / "source" / item["path"]) != item["sha256"]:
                raise ValueError("Model source inventory changed")
        unpack(inputs, OUT / "inputs")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1+cu121",
            "--index-url", "https://download.pytorch.org/whl/cu121"], check=True, timeout=min(600,remaining()))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
            "torch-geometric==2.6.1", "ogb==1.3.6"], check=True, timeout=min(180,remaining()))
        modes = list(spec["variants"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(queue, device, modes[device::len(devices)]) for device in range(len(devices))]
            for future in futures:
                future.result()
        result["complete"] = True
    except Exception as error:
        result["failure"] = repr(error)
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["allocated_device_seconds"] = result["wall_seconds"] * len(devices)
        result["worker_terminal_sha256"] = {p.parent.name: sha(p) for p in (OUT / "workers").glob("*/terminal.json")}
        save(OUT / "execution.json", result)


if __name__ == "__main__":
    main()
