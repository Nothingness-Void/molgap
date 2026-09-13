"""Kaggle1 P100 entry point for the frozen K1 combined simplification."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


MODE = "neural_atom_k1_no_attention_uniform_return"


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_root() -> Path:
    archive = find_one("src.zip")
    root = Path("/kaggle/working/_molgap_source")
    if not root.exists():
        shutil.unpack_archive(archive, root)
    modules = list(root.rglob("molgap/pcqm_k1_variants.py"))
    if len(modules) != 1:
        raise FileNotFoundError(f"Unexpected source layout: {modules}")
    return modules[0].parents[1]


def ensure_p100_torch() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; raise SystemExit(0 if 'sm_60' in torch.cuda.get_arch_list() else 3)"],
        check=False,
    )
    if probe.returncode != 0:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--upgrade",
            "--force-reinstall", "torch==2.4.1", "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ])


def main() -> None:
    ensure_p100_torch()
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch-geometric==2.6.1", "ogb==1.3.6",
    ])
    import torch
    if torch.cuda.device_count() != 1 or "P100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Candidate requires one P100, found {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    os.environ["MOLGAP_PLATFORM_ID"] = "kaggle1"
    os.environ["MOLGAP_FIXED_DATASET"] = "nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1"
    sys.path.insert(0, str(source_root()))
    from molgap.pcqm_k1_variants_runner import train_arm
    train_arm(
        MODE,
        Path("/kaggle/working/pcqm_k1_combined_simplification") / MODE,
        source_commit=find_one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip(),
        source_archive_sha256=find_one("SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip(),
    )


if __name__ == "__main__":
    main()
