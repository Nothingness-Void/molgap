"""Kaggle1 entry point for the single K1-MoSE V5 candidate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


MODE = "neural_atom_k1_mose"


def find_one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {matches}")
    return matches[0]


def install_dependencies() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "torch-geometric==2.6.1", "ogb==1.3.6"]
    )


def source_root() -> Path:
    archive = find_one("source_payload.bin")
    root = Path("/kaggle/working/_molgap_mose_source")
    if not root.exists():
        shutil.unpack_archive(archive, root, format="gztar")
    return root


def verify_source(root: Path) -> tuple[str, str]:
    archive = find_one("source_payload.bin")
    digest = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest or len(commit) != 40:
        raise RuntimeError("Source identity mismatch")
    inventory = json.loads(find_one("SOURCE_FILES.json").read_text())["files"]
    for item in inventory:
        path = root / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Source file hash mismatch: {item['path']}")
    return commit, digest


def main() -> None:
    os.environ["MOLGAP_PLATFORM_ID"] = "kaggle1"
    install_dependencies()
    root = source_root()
    commit, digest = verify_source(root)
    sys.path.insert(0, str(root / "src"))
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("K1-MoSE requires exactly one visible CUDA device")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must be disabled before training")
    from molgap.pcqm_k1_variants_runner import train_arm

    train_arm(
        MODE,
        Path("/kaggle/working/pcqm_k1_mose_100k/neural_atom_k1_mose"),
        source_commit=commit,
        source_archive_sha256=digest,
    )


if __name__ == "__main__":
    main()

