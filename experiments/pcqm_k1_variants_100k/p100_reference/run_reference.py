"""Kaggle2 P100 entry point for the one-time K1-v4 reference."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def ensure_p100_torch() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.cuda.get_arch_list()); "
            "raise SystemExit(0 if 'sm_60' in torch.cuda.get_arch_list() else 3)",
        ],
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
        [
            sys.executable,
            "-c",
            "import torch; print(torch.__version__, torch.cuda.get_arch_list()); "
            "raise SystemExit(0 if 'sm_60' in torch.cuda.get_arch_list() else 4)",
        ],
        check=False,
    )
    if verified.returncode:
        raise RuntimeError("Installed PyTorch still lacks P100 sm_60 support")


def install_dependencies() -> None:
    ensure_p100_torch()
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "torch-geometric==2.6.1",
            "ogb==1.3.6",
        ]
    )


def source_root() -> Path:
    archive = find_one("src.zip")
    root = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, root)
    return root / "src"


def main() -> None:
    install_dependencies()
    sys.path.insert(0, str(source_root()))
    import torch

    if torch.cuda.device_count() != 1 or "P100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Reference requires one P100, got {torch.cuda.get_device_name(0)}")
    from molgap.pcqm_k1_variants_runner import train_arm

    source_commit = find_one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    archive_sha = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
    train_arm(
        "neural_atom_k1_v4",
        Path("/kaggle/working/pcqm_k1_v4_reference/neural_atom_k1_v4"),
        source_commit=source_commit,
        source_archive_sha256=archive_sha,
    )


if __name__ == "__main__":
    main()

