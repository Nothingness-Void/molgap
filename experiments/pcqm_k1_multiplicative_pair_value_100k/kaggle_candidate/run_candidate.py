"""Kaggle entry point for one frozen candidate and one visible GPU."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


MODE = "neural_atom_k1_multiplicative_pair_value"


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_root() -> Path:
    modules = list(
        Path("/kaggle/input").rglob("src/molgap/k1_multiplicative_pair_value.py")
    )
    if len(modules) == 1:
        return modules[0].parents[1]
    archive = find_one("source_payload.bin")
    root = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, root, format="gztar")
    return next(root.rglob("src/molgap/k1_multiplicative_pair_value.py")).parents[1]


def main() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",", maxsplit=1)[0] or "0"
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "torch-geometric==2.6.1", "ogb==1.3.6"]
    )
    root = source_root()
    archive = find_one("source_payload.bin")
    digest = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest or len(commit) != 40:
        raise RuntimeError("Source identity mismatch")
    inventory = json.loads(find_one("SOURCE_FILES.json").read_text())["files"]
    for item in inventory:
        relative = Path(item["path"])
        mounted = root.joinpath(*relative.parts[1:])
        if hashlib.sha256(mounted.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Mounted source differs: {relative}")
    sys.path.insert(0, str(root))
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA accelerator is required")
    print(json.dumps({"device": torch.cuda.get_device_name(0)}), flush=True)
    os.environ["MOLGAP_PLATFORM_ID"] = "kaggle3"
    from molgap.pcqm_k1_variants_runner import train_arm

    train_arm(
        MODE,
        Path("/kaggle/working/pcqm_k1_multiplicative_pairvalue") / MODE,
        source_commit=commit,
        source_archive_sha256=digest,
    )


if __name__ == "__main__":
    main()
