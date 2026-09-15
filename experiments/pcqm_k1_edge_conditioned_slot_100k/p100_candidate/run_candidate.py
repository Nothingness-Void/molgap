"""Kaggle2 P100 entry point for the single K1 edge-conditioned candidate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


MODE = "neural_atom_k1_edge_conditioned_slot"


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def ensure_p100_torch() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; raise SystemExit(0 if 'sm_60' in torch.cuda.get_arch_list() else 3)"],
        check=False,
    )
    if probe.returncode == 0:
        return
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "--force-reinstall",
            "torch==2.4.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]
    )
    verified = subprocess.run(
        [sys.executable, "-c", "import torch; raise SystemExit(0 if 'sm_60' in torch.cuda.get_arch_list() else 4)"],
        check=False,
    )
    if verified.returncode:
        raise RuntimeError("Installed PyTorch still lacks P100 sm_60 support")


def install_dependencies() -> None:
    ensure_p100_torch()
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "torch-geometric==2.6.1", "ogb==1.3.6"]
    )


def source_root() -> Path:
    modules = list(Path("/kaggle/input").rglob("src/molgap/pcqm_k1_variants.py"))
    if len(modules) == 1:
        return modules[0].parents[1]
    archive = find_one("source_payload.bin")
    root = Path("/kaggle/working/_molgap_source")
    if not root.exists():
        shutil.unpack_archive(archive, root, format="gztar")
    modules = list(root.rglob("src/molgap/pcqm_k1_variants.py"))
    if len(modules) != 1:
        raise FileNotFoundError(f"Unexpected source layout: {modules}")
    return modules[0].parents[1]


def verify_source(root: Path) -> tuple[str, str]:
    archive = find_one("source_payload.bin")
    digest = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
    commit = find_one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest or len(commit) != 40:
        raise RuntimeError("Source identity mismatch")
    inventory = json.loads(find_one("SOURCE_FILES.json").read_text(encoding="utf-8"))["files"]
    for item in inventory:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "src":
            raise RuntimeError("Invalid source inventory path")
        mounted = root.joinpath(*relative.parts[1:])
        if hashlib.sha256(mounted.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Mounted source differs: {relative}")
    return commit, digest


def main() -> None:
    install_dependencies()
    root = source_root()
    commit, digest = verify_source(root)
    sys.path.insert(0, str(root))
    import torch

    if torch.cuda.device_count() != 1 or "P100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Candidate requires one P100, got {torch.cuda.get_device_name(0)}")
    from molgap.pcqm_k1_variants_runner import train_arm

    train_arm(
        MODE,
        Path(f"/kaggle/working/pcqm_k1_edge_conditioned_slot/{MODE}"),
        source_commit=commit,
        source_archive_sha256=digest,
    )


if __name__ == "__main__":
    main()
