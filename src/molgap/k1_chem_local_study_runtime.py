"""Kaggle2 T4x2 execution for the frozen local-color comparison.

Each arm is an independent FP32 K1 variant with one visible GPU. Submission,
acceptance, and any later NO_TRAIN audit remain separate controller actions.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from .k1_chem_local import MODES


RUN_ID = "kaseichou/molgap-k1-chem-local-s42:v1"
TRAJECTORIES = {
    mode: f"TC-k1-{mode.removeprefix('neural_atom_k1_').replace('_', '-')}-100k-s42"
    for mode in MODES
}
ROOT = Path("/kaggle/working/pcqm_k1_chem_local")
MAX_SECONDS = 6 * 3600


def bootstrap(source, archive):
    """Refuse uncommitted or mismatched code before installing training imports."""
    source, archive = Path(source), Path(archive)
    payload = archive.parent
    commit = (payload / "SOURCE_COMMIT.txt").read_text().strip()
    digest = (payload / "SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    release = json.loads((payload / "STUDY_RELEASE.json").read_text())
    if len(commit) != 40 or hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Immutable source archive identity failed")
    if release["source_commit"] != commit or release["run_id"] != RUN_ID:
        raise RuntimeError("Prospective study release differs from source")
    for row in json.loads((payload / "SOURCE_FILES.json").read_text())["files"]:
        path = (source / row["path"]).resolve()
        if (not path.is_relative_to(source.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise RuntimeError(f"Source inventory differs: {row['path']}")
    if set(release["arms"]) != set(MODES):
        raise RuntimeError("Dual-arm release differs")
    for mode in MODES:
        if (release["arms"][mode]["trajectory_id"] != TRAJECTORIES[mode]
                or release["arms"][mode]["prelaunch_ready"] is not True):
            raise RuntimeError(f"Arm lacks prospective release: {mode}")
    run({"python_root": str(source / "src"), "source_commit": commit,
         "source_archive_sha256": digest})


def child(context, mode):
    from .training_reproducibility import atomic_json, sha256_file

    if mode not in MODES:
        raise ValueError(mode)
    output = ROOT / mode
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    succeeded, gpu = False, None
    try:
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Each independent arm must see one GPU")
        gpu = torch.cuda.get_device_name(0)
        from .pcqm_k1_variants_runner import train_arm
        train_arm(mode, output, source_commit=context["source_commit"],
                  source_archive_sha256=context["source_archive_sha256"],
                  trajectory_id=TRAJECTORIES[mode], physical_run_id=RUN_ID)
        succeeded = True
    except BaseException:
        atomic_json(output / "failure.json", {"mode": mode, "run_id": RUN_ID,
            "traceback": traceback.format_exc(), "automatic_retry": False})
        raise
    finally:
        elapsed = time.monotonic() - started
        atomic_json(output / "native_cost.json", {
            "format": "molgap-k1-chem-local-native-cost-v1", "mode": mode,
            "trajectory_id": TRAJECTORIES[mode], "run_id": RUN_ID,
            "hardware": gpu, "device_count": 1, "wall_seconds": elapsed,
            "allocated_device_seconds": elapsed,
            "cpu_process_seconds": time.process_time() - cpu_started,
            "cpu_scope": "worker-process-only", "training_completed": succeeded,
            "inference_executed_locally": False,
        })
        manifest_path = output / "completion_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["artifact_sha256"]["native_cost.json"] = sha256_file(output / "native_cost.json")
            atomic_json(manifest_path, manifest)


def run(context):
    from .k1_edge_kaggle_runtime import _pin_runtime
    from .k1_relation_study_runtime import worker
    from .training_reproducibility import atomic_json

    ROOT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _pin_runtime(required_devices=2, required_name="T4")
    import torch
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    atomic_json(ROOT / "launch_identity.json", {**context, "run_id": RUN_ID,
        "modes": MODES, "allocated_devices": names, "used_device_count": 2})
    deadline = time.monotonic() + MAX_SECONDS
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, context, mode, device, deadline, output_root=ROOT,
                               child_module="molgap.k1_chem_local_study_runtime", run_id=RUN_ID)
                   for device, mode in enumerate(MODES)]
        results = [future.result() for future in futures]
    elapsed = time.monotonic() - started
    complete = all(row["complete"] for row in results)
    atomic_json(ROOT / "execution_summary.json", {
        "format": "molgap-k1-chem-local-execution-v1", "run_id": RUN_ID,
        "workers": results, "complete": complete, "allocated_device_names": names,
        "used_device_count": 2, "allocated_device_count": len(names),
        "total_job_wall_seconds": elapsed,
        "total_allocated_device_seconds": elapsed * len(names),
        "automatic_successor_submitted": False,
    })
    if not complete:
        raise RuntimeError("Retained worker failure; no automatic restart")


if __name__ == "__main__":
    child(json.loads(os.environ["MOLGAP_RELATION_CONTEXT"]), os.environ["MOLGAP_RELATION_MODE"])
