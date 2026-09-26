"""One single-device 2D screen, then a separately frozen NO_TRAIN audit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

# A child's environment changes never propagate back to the launcher. The audit
# is a fresh sibling of training, so both must inherit the CuBLAS contract here.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


MODES = ("neural_atom_k1_linear_attention",)
OUT = Path("/kaggle/working/pcqm_k1_linear_attention")
SOURCE_REF = "nothingnessvoid/molgap-k1-linear-attention-source"
FIXED_100K = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
FIXED_500K = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"


def one(pattern):
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {len(matches)}")
    return matches[0]


def source_root():
    archive = one("source_payload.bin")
    digest = one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    commit = one("SOURCE_COMMIT.txt").read_text().strip()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest or len(commit) != 40:
        raise RuntimeError("Frozen source archive changed")
    root = OUT / "source"
    shutil.unpack_archive(archive, root, format="gztar")
    for item in json.loads(one("SOURCE_FILES.json").read_text())["files"]:
        path = root / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Mounted source differs: {item['path']}")
    return root / "src", commit, digest


def fixed_root(digest):
    matches = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        if hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one accepted fixed graph cache {digest}, found {len(matches)}")
    return matches[0]


def worker(mode, root, commit, digest):
    sys.path.insert(0, str(root))
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each T4 training arm must see one GPU")
    os.environ["MOLGAP_PLATFORM_ID"] = "kaggle1"
    os.environ["MOLGAP_FIXED_DATASET"] = "nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1"
    from molgap.pcqm_k1_variants_runner import train_arm
    train_arm(mode, OUT / mode, source_commit=commit, source_archive_sha256=digest)


def audit(root, commit, digest):
    sys.path.insert(0, str(root))
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("NO_TRAIN audit must see one GPU")
    from molgap.k1_portability_audit import run
    run(
        cache_100k=fixed_root(FIXED_100K),
        cache_500k=fixed_root(FIXED_500K),
        reference_model=one("K1_REFERENCE_BEST_MODEL.pt"),
        reference_payload=one("K1_REFERENCE_DEVELOPMENT.pt"),
        transform_asset=one("K1_REFERENCE_TARGET_TRANSFORM.json"),
        candidate_root=OUT,
        output=OUT / "post100k_audit",
        source_commit=commit,
        source_archive_sha256=digest,
        modes=MODES,
    )


def main():
    if len(sys.argv) == 6 and sys.argv[1] == "--worker":
        worker(sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5])
        return
    if len(sys.argv) == 5 and sys.argv[1] == "--audit":
        audit(Path(sys.argv[2]), sys.argv[3], sys.argv[4])
        return
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1+cu121",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch-geometric==2.6.1", "ogb==1.3.6",
    ])
    root, commit, digest = source_root()
    import torch
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not names:
        raise RuntimeError(f"Single-device screen requires CUDA, allocated {names}")
    fixed_root(FIXED_100K)
    fixed_root(FIXED_500K)  # identity only; role shard is not opened before training.
    print(json.dumps({"gpu_names": names, "source_commit": commit}), flush=True)
    jobs = []
    for gpu, mode in enumerate(MODES):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = str(root)
        jobs.append((mode, subprocess.Popen(
            [sys.executable, __file__, "--worker", mode, str(root), commit, digest], env=env,
        )))
    statuses = {}
    for mode, process in jobs:
        try:
            statuses[mode] = process.wait(timeout=6 * 3600)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=30)
            raise RuntimeError("STOP_FOR_COST: training exceeded six-hour safety budget; retain checkpoints")
    print(json.dumps({"worker_exit_codes": statuses}), flush=True)
    if any(status != 0 for status in statuses.values()):
        raise RuntimeError("Training arm failure; retain all independent checkpoints")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = str(root)
    subprocess.check_call(
        [sys.executable, __file__, "--audit", str(root), commit, digest], env=env, timeout=5400
    )


if __name__ == "__main__":
    main()
