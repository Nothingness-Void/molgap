"""Kaggle NO_TRAIN bootstrap; two isolated frozen models and no model selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback


def one(name):
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected one mounted {name}: {matches}")
    return matches[0]


def source(inputs, destination):
    release = json.loads((inputs / "audit_release.json").read_text())
    for name, digest in release["files"].items():
        if hashlib.sha256((inputs / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Mounted input bytes differ: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((inputs / "SOURCE_FILES.json").read_text())
    entries = {row["path"]: row for row in inventory["files"]}
    with tarfile.open(inputs / "source_payload.bin") as archive:
        members = archive.getmembers()
        if len(members) != len(entries) or {m.name for m in members} != set(entries):
            raise ValueError("Mounted source inventory mismatch")
        for member in members:
            if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts or member.linkname:
                raise ValueError("Unsafe source archive")
            payload = archive.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != entries[member.name]["sha256"]:
                raise ValueError("Mounted source member changed")
        archive.extractall(destination)
    if inventory["source_commit"] != release["source_commit"]:
        raise ValueError("Source commit mismatch")
    sys.path.insert(0, str(destination / "src"))
    return release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("ema9999", "ema999"))
    args = parser.parse_args()
    inputs = one("audit_release.json").parent
    output = Path("/kaggle/working/gptrans_ema_portability")
    output.mkdir(parents=True, exist_ok=True)
    if args.arm:
        release = source(inputs, Path(f"/kaggle/working/audit_source_{args.arm}"))
        from molgap.gptrans_portability import worker
        from molgap.training_reproducibility import atomic_json
        allocation = json.loads((output / "allocation.json").read_text())
        try:
            worker(arm=args.arm, inputs=inputs,
                cache_100k=one("train_shard_0002.pt").parent.parent,
                cache_500k=one("train_shard_0010.pt").parent.parent,
                output=output, release=release, deadline=allocation["deadline_unix"])
        except BaseException:
            atomic_json(output / args.arm / "failure.json", dict(error=traceback.format_exc(), timestamp=time.time()))
            raise
        return
    began = time.time()
    # Allocation is captured without importing torch in the coordinator.
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,uuid", "--format=csv,noheader"], text=True).strip().splitlines()
    release = source(inputs, Path("/kaggle/working/audit_source"))
    from molgap.training_reproducibility import atomic_json
    deadline = began + 5400
    atomic_json(output / "allocation.json", dict(started_unix=began, deadline_unix=deadline,
        devices=devices, allocated_devices=len(devices), idle_devices=max(len(devices)-2, 0),
        precision="fp32", physical_batch=128, training_executed=False, release=release))
    if len(devices) != 2 or any("T4" not in row for row in devices):
        raise RuntimeError("Frozen dual audit requires actual T4x2 allocation")
    # Retain the tested training runtime. These warnings about unrelated image
    # packages do not change the scientific environment used by the encoder.
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy==1.26.4",
        "torch==2.4.1", "--extra-index-url", "https://download.pytorch.org/whl/cu121"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "torch-geometric==2.6.1", "ogb==1.3.6", "rdkit==2025.9.5"], check=True)
    children = []
    for device, arm in enumerate(("ema9999", "ema999")):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), PYTHONHASHSEED="42",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2")
        children.append(subprocess.Popen([sys.executable, __file__, "--arm", arm], env=environment))
    status = "RUNNING"
    try:
        while any(child.poll() is None for child in children):
            if time.time() >= deadline or any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("Audit deadline or isolated worker failure; stopping the peer")
            time.sleep(1)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("Frozen audit worker failed")
        status = "COMPLETE"
    except BaseException:
        status = "ERROR"
        atomic_json(output / "failure.json", dict(error=traceback.format_exc(), timestamp=time.time()))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        elapsed = time.time()-began
        atomic_json(output / "cost.json", dict(status=status, allocated_devices=len(devices),
            allocation_wall_seconds=elapsed, allocated_device_hours=elapsed*len(devices)/3600,
            scope="kernel-entry-through-workers; teardown/queue unavailable", queue_hours=None))
        files = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in output.rglob("*") if p.is_file() and p.name != "output_manifest.json"}
        atomic_json(output / "output_manifest.json", dict(status=status, files=files, identity=release,
            training_executed=False, official_validation_role_read=False, test_dev_role_read=False,
            test_challenge_role_read=False))


if __name__ == "__main__":
    main()
