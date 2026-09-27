"""Bounded Kaggle execution for the frozen relation-resolution study.

This is a platform workload recipe, not a submitter. The separate post-run
NO_TRAIN stage cannot be reached from this training entry point.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

SLOTS = {
    "dual": ("neural_atom_k1_receiver_pair", "neural_atom_k1_triplet_aggregate"),
    "rrwp": ("neural_atom_k1_rrwp_pair",),
}
RUNS = {
    "dual": "kaseichou/molgap-k1-receiver-triplet-s42:v1",
    "rrwp": "kaseichou/molgap-k1-rrwp-pair-s42:v1",
}
TRAJECTORIES = {
    mode: f"TC-k1-{mode.removeprefix('neural_atom_k1_').replace('_', '-')}-100k-s42"
    for modes in SLOTS.values() for mode in modes
}
ROOT = Path("/kaggle/working/pcqm_k1_relation_resolution")
LIMIT_SECONDS = 6 * 3600


def bootstrap(source, archive, slot):
    """Verify packaged source and the exact slot plan before consuming compute."""
    source = Path(source)
    archive = Path(archive)
    payload = archive.parent
    commit = (payload / "SOURCE_COMMIT.txt").read_text().strip()
    digest = (payload / "SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    release = json.loads((payload / "STUDY_RELEASE.json").read_text())
    if len(commit) != 40 or hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Source archive identity failed")
    if release["source_commit"] != commit or release["slot_runs"][slot] != RUNS[slot]:
        raise RuntimeError("Prospective source/run release differs")
    for row in json.loads((payload / "SOURCE_FILES.json").read_text())["files"]:
        path = (source / row["path"]).resolve()
        if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"Source inventory differs: {row['path']}")
    for mode in SLOTS[slot]:
        row = release["arms"][mode]
        if row["prelaunch_ready"] is not True or row["trajectory_id"] != TRAJECTORIES[mode]:
            raise RuntimeError("Arm is not prospectively released")
    run({"python_root": str(source / "src"), "source_commit": commit,
         "source_archive_sha256": digest}, slot)


def child(context, mode):
    from .training_reproducibility import atomic_json, sha256_file
    output = ROOT / mode
    if output.exists():
        raise RuntimeError("Existing arm output cannot be silently overwritten or restarted")
    output.mkdir(parents=True)
    started = time.monotonic()
    succeeded = False
    gpu = None
    try:
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Each model must see exactly one accelerator")
        gpu = torch.cuda.get_device_name(0)
        from .pcqm_k1_variants_runner import train_arm
        train_arm(mode, output, source_commit=context["source_commit"],
                  source_archive_sha256=context["source_archive_sha256"],
                  trajectory_id=TRAJECTORIES[mode], physical_run_id=RUNS[context["slot"]])
        succeeded = True
    except BaseException:
        atomic_json(output / "failure.json", {"mode": mode, "traceback": traceback.format_exc(),
            "run_id": RUNS[context["slot"]], "training_completed": False})
        raise
    finally:
        elapsed = time.monotonic() - started
        atomic_json(output / "native_cost.json", {
            "format": "molgap-relation-study-native-cost-v1", "mode": mode,
            "trajectory_id": TRAJECTORIES[mode], "run_id": RUNS[context["slot"]],
            "hardware": gpu, "device_count": 1, "wall_seconds": elapsed,
            "allocated_device_seconds": elapsed, "measurement": "monotonic-worker-wall-times-one-device",
            "scope": "role-loading-runtime-qualification-preflight-and-training",
            "training_completed": succeeded, "inference_executed_locally": False,
        })
        if succeeded:
            manifest_path = output / "completion_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifact_sha256"]["native_cost.json"] = sha256_file(output / "native_cost.json")
            atomic_json(manifest_path, manifest)


def worker(context, mode, device, deadline, *, output_root=None,
           child_module="molgap.k1_relation_study_runtime", run_id=None):
    from .training_reproducibility import atomic_json
    root = Path(output_root) if output_root is not None else ROOT
    run_id = run_id or RUNS[context["slot"]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), CUBLAS_WORKSPACE_CONFIG=":4096:8",
               PYTHONHASHSEED="42", PYTHONUNBUFFERED="1", PYTHONPATH=context["python_root"],
               MOLGAP_PLATFORM_ID="kaggle2",
               MOLGAP_FIXED_DATASET="kaseichou/pcqm4mv2-ogb-fixed-100k-v1",
               MOLGAP_RELATION_CONTEXT=json.dumps(context), MOLGAP_RELATION_MODE=mode)
    log_path = root / f"{mode}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", child_module],
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        last = None
        timed_out = False
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            try:
                process.wait(timeout=min(45, remaining))
            except subprocess.TimeoutExpired:
                pass
            tail = log_path.read_text(errors="replace")[-12000:]
            lines = tail.splitlines()
            if lines and lines[-1] != last:
                last = lines[-1]
                print(f"{mode}: {last}", flush=True)
    elapsed = time.monotonic() - started
    result = {"mode": mode, "device": device, "exit_code": process.returncode,
        "timed_out": timed_out, "worker_wall_seconds": elapsed,
        "allocated_device_seconds": elapsed, "run_id": run_id,
        "log": log_path.name, "complete": process.returncode == 0 and not timed_out}
    atomic_json(root / f"{mode}_execution.json", result)
    if not result["complete"]:
        print(log_path.read_text(errors="replace")[-14000:], flush=True)
    return result


def run(context, slot):
    if slot not in SLOTS:
        raise ValueError(slot)
    started = time.monotonic()
    from .k1_edge_kaggle_runtime import _pin_runtime
    _pin_runtime(required_devices=len(SLOTS[slot]), required_name="T4" if slot == "dual" else None)
    import torch
    from .training_reproducibility import atomic_json
    context = dict(context, slot=slot)
    ROOT.mkdir(parents=True, exist_ok=True)
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    atomic_json(ROOT / "launch_identity.json", {**context, "allocated_devices": names,
        "modes": SLOTS[slot], "run_id": RUNS[slot], "audit_in_this_process": False})
    print(json.dumps({"run_id": RUNS[slot], "gpu_names": names, "modes": SLOTS[slot]}), flush=True)
    deadline = time.monotonic() + LIMIT_SECONDS
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SLOTS[slot])) as executor:
        futures = [executor.submit(worker, context, mode, gpu, deadline)
                   for gpu, mode in enumerate(SLOTS[slot])]
        results = [future.result() for future in futures]
    atomic_json(ROOT / "execution_summary.json", {
        "format": "molgap-relation-study-execution-v1", "run_id": RUNS[slot],
        "workers": results, "complete": all(row["complete"] for row in results),
        "allocated_device_names": names,
        "used_device_count": len(SLOTS[slot]),
        "allocated_device_count": len(names),
        "total_job_wall_seconds": time.monotonic() - started,
        "total_allocated_device_seconds": (time.monotonic() - started) * len(names),
        "cost_scope": "source-verified-bootstrap-through-workers-including-runtime-install",
        "audit_submitted": False, "automatic_successor_submitted": False})
    if not all(row["complete"] for row in results):
        raise RuntimeError("Retained terminal worker failure; no automatic restart")


if __name__ == "__main__":
    child(json.loads(os.environ["MOLGAP_RELATION_CONTEXT"]), os.environ["MOLGAP_RELATION_MODE"])
