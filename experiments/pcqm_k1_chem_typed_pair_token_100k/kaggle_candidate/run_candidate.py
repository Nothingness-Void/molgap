"""Kaggle entry point for one frozen chemistry-conditioned PairToken screen."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


MODE = "neural_atom_k1_chem_typed_pair_token"


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_root() -> Path:
    modules = list(
        Path("/kaggle/input").rglob("src/molgap/k1_chem_typed_pair_token.py")
    )
    if len(modules) == 1:
        return modules[0].parents[2]
    archive = find_one("source_payload.bin")
    root = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, root, format="gztar")
    return next(root.rglob("src/molgap/k1_chem_typed_pair_token.py")).parents[2]


def find_manifest(format_name: str) -> Path:
    matches = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("format") == format_name:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {format_name}, found {matches}")
    return matches[0].parent


def main() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",", maxsplit=1)[0] or "0"
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
    root = source_root()
    archive = find_one("source_payload.bin")
    digest = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest or len(commit) != 40:
        raise RuntimeError("Source identity mismatch")
    inventory = json.loads(find_one("SOURCE_FILES.json").read_text())["files"]
    for item in inventory:
        mounted = root / item["path"]
        if hashlib.sha256(mounted.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Mounted source differs: {item['path']}")
    sys.path.insert(0, str(root / "src"))
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA accelerator is required")
    os.environ["MOLGAP_PLATFORM_ID"] = "kaggle2"
    os.environ["MOLGAP_FIXED_CACHE_ROOT"] = str(
        find_manifest("molgap-pcqm4mv2-kaggle-fixed-subset-v1")
    )
    os.environ["MOLGAP_FUNCTIONAL_GROUP_SIDECAR_ROOT"] = str(
        find_manifest("molgap-pcqm-fixed100k-functional-group-sidecar-v1")
    )
    from molgap.pcqm_k1_variants_runner import train_arm

    train_arm(
        MODE,
        Path("/kaggle/working/pcqm_k1_chem_typed_pair_token") / MODE,
        source_commit=commit,
        source_archive_sha256=digest,
    )


if __name__ == "__main__":
    main()
