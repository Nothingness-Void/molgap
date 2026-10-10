"""Thin Kaggle adapter for the frozen shared single-forward/EMA500K runner."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time

PINNED_PAYLOAD_SHA256 = "__PAYLOAD_SHA256__"
STARTED = time.time()
OUTPUT = Path("/kaggle/working/k1_single_ema_500k")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=False)
    launch_files = list(Path("/kaggle/input").rglob("k1_t4_payload.json"))
    if len(launch_files) != 1 or digest(launch_files[0]) != PINNED_PAYLOAD_SHA256:
        raise ValueError("Expected one pinned T4 source payload")
    launch = launch_files[0]
    declaration = json.loads(launch.read_text())
    deadline = STARTED + declaration["wall_limit_seconds"]
    if declaration["wall_limit_seconds"] != 32400:
        raise ValueError("Only the frozen nine-hour control is released")
    for name, expected in declaration["files"].items():
        path = launch.parent / name
        if not path.resolve().is_relative_to(launch.parent.resolve()) or digest(path) != expected:
            raise ValueError("Payload binding changed: " + name)
    retained = OUTPUT / "frozen_payload"
    retained.mkdir()
    for name in (*declaration["files"], launch.name):
        shutil.copyfile(launch.parent / name, retained / name)
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()
    if not devices or any("T4" not in name for name in devices):
        raise ValueError("Expected physical T4 allocation")
    observation = {"physical_gpu_names": devices, "cuda_visible_devices": "0",
                   "allocation_started_unix": STARTED, "deadline_unix": deadline,
                   "provider_billing_units": None, "job_id_is_logical_locator": True}
    (OUTPUT / "allocation.json").write_text(json.dumps(observation, indent=2))
    print("PHYSICAL_T4_ALLOCATION", observation, flush=True)

    def bounded(command, maximum):
        remaining = deadline - time.time() - 180
        if remaining <= 0:
            raise TimeoutError("Setup exhausted original allocation budget")
        subprocess.run(command, check=True, timeout=min(maximum, remaining))

    bounded([sys.executable, "-m", "pip", "install", "uv==0.10.9"], 180)
    environment = Path("/kaggle/temp/k1-single-ema-python311")
    bounded(["uv", "venv", "--python", "3.11", str(environment)], 180)
    python = str(environment / "bin/python")
    bounded(["uv", "pip", "install", "--python", python, "torch==2.4.1",
             "--index-url", "https://download.pytorch.org/whl/cu121"], 600)
    bounded(["uv", "pip", "install", "--python", python, "numpy==1.26.4",
             "torch-geometric==2.6.1", "ogb==1.3.6"], 300)
    source = Path("/kaggle/temp/k1-single-ema-source")
    source.mkdir(exist_ok=False)
    archive = retained / declaration["source_archive_storage_name"]
    if archive.name != "source_payload.bin":
        raise ValueError("Source archive must use non-extracting Kaggle storage")
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            if not (member.isfile() or member.isdir()) or not (
                    source / member.name).resolve().is_relative_to(source.resolve()):
                raise ValueError("Unsafe source archive entry")
        bundle.extractall(source)
    inventory = json.loads((retained / "SOURCE_FILES.json").read_text())
    if inventory["source_commit"] != declaration["runconfig"]["source_commit"]:
        raise ValueError("Source commit mismatch")
    for entry in inventory["files"]:
        if digest(source / entry["path"]) != entry["sha256"]:
            raise ValueError("Source file mismatch: " + entry["path"])
    roots = [p.parent for p in Path("/kaggle/input").rglob("manifest.json")
             if declaration["graph_mount"] in p.parts and digest(p) == declaration["manifest_sha256"]]
    if len(roots) != 1:
        raise ValueError("Expected one accepted fixed500K data mount")
    config = dict(declaration["runconfig"], allocation_started_unix=STARTED)
    config_path = OUTPUT / "parent_runconfig.json"
    config_path.write_text(json.dumps(config, sort_keys=True, indent=2))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source / "src"), CUDA_VISIBLE_DEVICES="0",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="42")
    command = [python, "-u", "-m", "molgap.colab_k1_screen", "run",
               "--dataset-root", str(roots[0]), "--initial-path", str(retained / "initial_state.pt"),
               "--initial-sha256", declaration["files"]["initial_state.pt"],
               "--runconfig-path", str(config_path), "--runconfig-sha256", digest(config_path),
               "--source-archive", str(archive),
               "--output", str(OUTPUT / "worker"), "--deadline", str(deadline)]
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    def stream():
        with (OUTPUT / "worker.log").open("w", buffering=1) as log:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
    reader = threading.Thread(target=stream, daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=max(1, deadline - time.time() - 30))
    except subprocess.TimeoutExpired:
        process.kill()
        returncode = process.wait(timeout=10)
        observation["parent_forced_stop"] = True
    finally:
        if process.poll() is None:
            process.kill()
        reader.join(timeout=5)
    observation.update(returncode=returncode, observed_allocation_wall_seconds=time.time() - STARTED)
    (OUTPUT / "allocation.json").write_text(json.dumps(observation, indent=2))
    print("KAGGLE_BOUNDED_PAIR_FINISHED", observation, flush=True)
    if returncode:
        raise RuntimeError("Worker failed; retain output and reconcile, no retry")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUTPUT.mkdir(exist_ok=True)
        (OUTPUT / "parent_failure.json").write_text(json.dumps({
            "type": type(error).__name__, "error": str(error),
            "observed_wall_seconds": time.time() - STARTED, "complete": False}, indent=2))
        raise
