"""Small NO_TRAIN coordinator reusing source integrity, environment and ledger IO."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

from .training_reproducibility import atomic_json, sha256_file


def extract_source(inputs: Path, destination: Path):
    release = json.loads((inputs/"audit_release.json").read_text())
    for name, digest in release["files"].items():
        if sha256_file(inputs/name) != digest:
            raise ValueError("Mounted frozen input differs: " + name)
    inventory = json.loads((inputs/"SOURCE_FILES.json").read_text())
    entries = {item["path"]: item for item in inventory["files"]}
    if inventory["source_commit"] != release["source_commit"]:
        raise ValueError("Source commit differs")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(inputs/"source_payload.bin") as archive:
        members = archive.getmembers()
        if len(members) != len(entries) or {m.name for m in members} != set(entries):
            raise ValueError("Source inventory differs")
        for member in members:
            if (not member.isfile() or Path(member.name).is_absolute()
                    or ".." in Path(member.name).parts or member.linkname):
                raise ValueError("Unsafe source member")
            payload = archive.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != entries[member.name]["sha256"]:
                raise ValueError("Source member differs")
        archive.extractall(destination)
    return release


def run_workers(*, inputs, output, source_root, entry_script, tasks, cap_seconds,
                allocation_started, started_unix):
    """Isolate workers; retain failures and all allocated time, never optimize."""
    from .experiment_allocation import AllocationLedger
    from .kaggle_python_environment import prepare_python
    output.mkdir(parents=True, exist_ok=True)
    release = json.loads((inputs/"audit_release.json").read_text())
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()
    if len(devices) != len(tasks) or any("T4" not in item for item in devices):
        raise ValueError("Frozen independent diagnostic tasks require actual T4x2")
    ledger = AllocationLedger(spec_identity=release["identity"], hardware=devices,
        assignments={task:i for i,task in enumerate(tasks)}, started=allocation_started)
    ledger.write(output, "running")
    deadline = started_unix + cap_seconds
    atomic_json(output/"allocation.json", dict(deadline_unix=deadline, started_unix=started_unix,
        release_identity=release["identity"], tasks=list(tasks), training_executed=False))
    children, status = [], "failed"
    try:
        executable, qualified = prepare_python(output.parent/"audit_environment", deadline=deadline, source_root=source_root)
        atomic_json(output/"environment.json", qualified)
        for index, task in enumerate(tasks):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index), PYTHONHASHSEED="42",
                PYTHONPATH=str(source_root/"src"), CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2")
            children.append(subprocess.Popen([str(executable), str(entry_script), "--task", task], env=env))
        while any(child.poll() is None for child in children):
            if time.time() >= deadline-25 or any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("Frozen audit worker failure/deadline")
            ledger.write(output, "running")
            time.sleep(2)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("Frozen audit worker failed")
        status = "complete"
    except BaseException:
        atomic_json(output/"failure.json", dict(error=traceback.format_exc(), timestamp=time.time()))
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
        ledger.write(output, status)
        files = {p.relative_to(output).as_posix():sha256_file(p) for p in output.rglob("*")
                 if p.is_file() and p.name != "output_manifest.json"}
        atomic_json(output/"output_manifest.json", dict(status=status, release=release, files=files,
            training_executed=False, optimizer_steps=0, official_validation_role_read=False,
            test_dev_role_read=False, test_challenge_role_read=False))
