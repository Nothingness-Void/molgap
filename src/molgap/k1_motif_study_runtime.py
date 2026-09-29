"""One isolated Kaggle2 T4 arm; existing K1 runner owns all training state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .k1_motif_hierarchy import MODE


RUN_ID = "kaseichou/molgap-k1-motif-hierarchy-s42:v3"
TRAJECTORY_ID = "TC-k1-motif-hierarchy-100k-s42-v3"
ROOT = Path("/kaggle/working/pcqm_k1_motif_hierarchy")


def select_t4_device(device_names: list[str]) -> int:
    """Reserve one T4 without treating a Kaggle T4x2 allocation as two arms."""
    if (len(device_names) not in (1, 2)
            or any(not isinstance(name, str) or "T4" not in name for name in device_names)):
        raise RuntimeError(f"Expected one or two T4 devices, got {device_names!r}")
    return 0


def bootstrap(source: Path, archive: Path) -> None:
    from .training_reproducibility import sha256_file

    source, archive = Path(source), Path(archive)
    payload = archive.parent
    commit = (payload / "SOURCE_COMMIT.txt").read_text().strip()
    digest = (payload / "SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    release = json.loads((payload / "STUDY_RELEASE.json").read_text())
    if (len(commit) != 40 or sha256_file(archive) != digest
            or release.get("source_commit") != commit
            or release.get("run_id") != RUN_ID
            or release.get("mode") != MODE
            or release.get("trajectory_id") != TRAJECTORY_ID
            or release.get("prelaunch_ready") is not True):
        raise RuntimeError("Motif GPU source or scientific release identity changed")
    inventory = json.loads((payload / "SOURCE_FILES.json").read_text())["files"]
    for row in inventory:
        path = (source / row["path"]).resolve()
        if (not path.is_relative_to(source.resolve())
                or sha256_file(path) != row["sha256"]):
            raise RuntimeError(f"Source inventory differs: {row['path']}")
    run(source_commit=commit, source_archive_sha256=digest)


def run(*, source_commit: str, source_archive_sha256: str) -> None:
    from .k1_edge_kaggle_runtime import _pin_runtime
    from .training_reproducibility import atomic_json, sha256_file

    if ROOT.exists():
        raise RuntimeError("Refusing unreviewed same-worker restart")
    ROOT.mkdir(parents=True)
    try:
        _pin_runtime(required_devices=1, required_name="T4")
        probe = subprocess.run(
            [sys.executable, "-c", (
                "import json,importlib.metadata as m,torch; "
                "print(json.dumps({'torch':torch.__version__,"
                "'cuda':torch.version.cuda,"
                "'pyg':m.version('torch-geometric'),'ogb':m.version('ogb'),"
                "'device_names':[torch.cuda.get_device_name(i) "
                "for i in range(torch.cuda.device_count())]}))"
            )], capture_output=True, text=True, check=True,
        )
        allocation = json.loads(probe.stdout)
        allocated_names = allocation["device_names"]
        selected_device = select_t4_device(allocated_names)
        # torch has only been imported in subprocesses: mask before this worker
        # imports CUDA so physical batch 128 remains on exactly one device.
        if "torch" in sys.modules:
            raise RuntimeError("CUDA was imported before T4 isolation")
        print("MOLGAP_GPU_ALLOCATION " + json.dumps(allocation), flush=True)
        atomic_json(ROOT / "runtime_probe.json", {
            "run_id": RUN_ID, "torch": allocation["torch"],
            "cuda": allocation["cuda"], "allocated_device_names": allocated_names,
            "allocated_device_count": len(allocated_names),
            "selected_device_index": selected_device, "accepted_gpu_family": "T4",
        })
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_device)
    except BaseException:
        probe = subprocess.run(
            [sys.executable, "-c", (
                "import json,torch; print(json.dumps({'device_names':"
                "[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"
            )], capture_output=True, text=True, check=False,
        )
        atomic_json(ROOT / "runtime_preflight_failure.json", {
            "run_id": RUN_ID, "failure": traceback.format_exc(),
            "probe_returncode": probe.returncode,
            "probe_stdout": probe.stdout[-4000:], "probe_stderr": probe.stderr[-4000:],
            "scientific_epochs_started": False,
        })
        raise
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.device_count() == 1 else None
    if gpu is None or "T4" not in gpu:
        raise RuntimeError("One isolated T4 is required for this candidate")
    output = ROOT / MODE
    output.mkdir()
    atomic_json(ROOT / "launch_identity.json", {
        "run_id": RUN_ID, "trajectory_id": TRAJECTORY_ID, "mode": MODE,
        "source_commit": source_commit, "source_archive_sha256": source_archive_sha256,
        "allocated_device_names": allocated_names, "used_device_count": 1,
        "motif_sidecar_aggregate_sha256":
            "5466ccd1f498619b045eb73d82f958949c1474ff0d303b99d6fd226737a0b8ae",
    })
    started, cpu_started = time.monotonic(), time.process_time()
    complete = False
    try:
        from .pcqm_k1_variants_runner import train_arm
        train_arm(MODE, output, source_commit=source_commit,
                  source_archive_sha256=source_archive_sha256,
                  trajectory_id=TRAJECTORY_ID, physical_run_id=RUN_ID)
        complete = True
    except BaseException:
        atomic_json(output / "failure.json", {
            "run_id": RUN_ID, "mode": MODE, "traceback": traceback.format_exc(),
            "automatic_retry": False,
        })
        raise
    finally:
        elapsed = time.monotonic() - started
        atomic_json(output / "native_cost.json", {
            "format": "molgap-k1-motif-native-cost-v1", "run_id": RUN_ID,
            "trajectory_id": TRAJECTORY_ID, "mode": MODE, "hardware": gpu,
            "allocated_device_count": len(allocated_names), "used_device_count": 1,
            "wall_seconds": elapsed,
            "allocated_device_seconds": elapsed * len(allocated_names),
            "cpu_process_seconds": time.process_time() - cpu_started,
            "training_completed": complete, "account_billing_inferred": False,
        })
        manifest = output / "completion_manifest.json"
        if manifest.is_file():
            record = json.loads(manifest.read_text())
            record["artifact_sha256"]["native_cost.json"] = sha256_file(output / "native_cost.json")
            atomic_json(manifest, record)
        atomic_json(ROOT / "execution_summary.json", {
            "format": "molgap-k1-motif-execution-v1", "run_id": RUN_ID,
            "complete": complete, "allocated_device_names": allocated_names,
            "used_device_count": 1, "total_job_wall_seconds": elapsed,
            "automatic_successor_submitted": False,
        })
