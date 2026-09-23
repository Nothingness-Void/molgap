"""Kaggle CPU-only fixed-graph conjugated membership preparation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def one(pattern):
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}: {matches}")
    return matches[0]


def main():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch-geometric==2.6.1", "ogb==1.3.6",
    ])
    archive = one("source_payload.bin")
    digest = one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    commit = one("SOURCE_COMMIT.txt").read_text().strip()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest or len(commit) != 40:
        raise RuntimeError("Frozen source identity changed")
    root = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, root, format="gztar")
    for item in json.loads(one("SOURCE_FILES.json").read_text())["files"]:
        mounted = root / item["path"]
        if hashlib.sha256(mounted.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Mounted source differs: {item['path']}")
    sys.path.insert(0, str(root / "src"))
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("CPU sidecar must not expose CUDA")
    from molgap.k1_conjugated_sidecar import (
        accept_sidecar, build_sidecar,
    )
    from molgap.training_reproducibility import atomic_json
    output = Path("/kaggle/working/molgap-k1-conjugated-sidecar-v1")
    manifest = build_sidecar(output, source_commit=commit)
    acceptance = accept_sidecar(output, expected_source_commit=commit)
    atomic_json(output / "acceptance.json", acceptance)
    print(json.dumps({
        "status": manifest["status"],
        "train_rows": manifest["train_rows"],
        "development_rows": manifest["development_rows"],
        "aggregate_sha256": manifest["aggregate_sha256"],
        "component_counts": manifest["component_counts"],
        "accepted": acceptance["accepted"],
    }), flush=True)


if __name__ == "__main__":
    main()
