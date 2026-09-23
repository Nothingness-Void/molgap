"""Kaggle T4x2 independent candidate processes; no baseline retraining."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


MODES = (
    "neural_atom_k1_conjugated_oneshot",
    "neural_atom_k1_conjugated_persistent",
)
OUT = Path("/kaggle/working/pcqm_k1_conjugated_dual")


def one(pattern):
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}: {matches}")
    return matches[0]


def source_root():
    archive = one("source_payload.bin")
    digest = one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    commit = one("SOURCE_COMMIT.txt").read_text().strip()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest or len(commit) != 40:
        raise RuntimeError("Frozen source identity changed")
    root = OUT / "source"
    shutil.unpack_archive(archive, root, format="gztar")
    for item in json.loads(one("SOURCE_FILES.json").read_text())["files"]:
        path = root / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Mounted source differs: {item['path']}")
    return root / "src", commit, digest


def worker(mode, root, commit, digest):
    sys.path.insert(0, str(root))
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each arm must see exactly one CUDA accelerator")
    os.environ["MOLGAP_PLATFORM_ID"] = "kaggle2"
    from molgap.pcqm_k1_variants_runner import train_arm
    train_arm(mode, OUT / mode, source_commit=commit, source_archive_sha256=digest)


def main():
    if len(sys.argv) == 6 and sys.argv[1] == "--worker":
        worker(sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5])
        return
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch-geometric==2.6.1", "ogb==1.3.6",
    ])
    root, commit, digest = source_root()
    import torch
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != 2 or any("T4" not in name for name in names):
        raise RuntimeError(f"Dual arm requires T4x2: {names}")
    print(json.dumps({"gpu_names": names, "source_commit": commit}), flush=True)
    jobs = []
    for gpu, mode in enumerate(MODES):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = str(root)
        jobs.append((mode, subprocess.Popen(
            [sys.executable, __file__, "--worker", mode, str(root), commit, digest],
            env=env,
        )))
    statuses = {mode: process.wait() for mode, process in jobs}
    print(json.dumps({"worker_exit_codes": statuses}), flush=True)
    if any(code != 0 for code in statuses.values()):
        raise RuntimeError("At least one arm failed; retain both output directories")


if __name__ == "__main__":
    main()
