"""Kaggle CPU entry point for immutable functional-group incidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_root() -> Path:
    modules = list(
        Path("/kaggle/input").rglob("src/molgap/pcqm_functional_group_sidecar.py")
    )
    if len(modules) == 1:
        return modules[0].parents[2]
    archive = find_one("source_payload.bin")
    root = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, root, format="gztar")
    return next(root.rglob("src/molgap/pcqm_functional_group_sidecar.py")).parents[2]


def fixed_root() -> Path:
    manifests = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            payload.get("format") == "molgap-pcqm4mv2-kaggle-fixed-subset-v1"
            and payload.get("identity", {}).get("name") == "ogb-train-100k"
        ):
            manifests.append(path)
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one fixed cache, found {manifests}")
    return manifests[0].parent


def main() -> None:
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

    if torch.cuda.is_available():
        raise RuntimeError("CPU sidecar job must not expose a GPU")
    from molgap.pcqm_functional_group_sidecar import (
        accept_functional_group_sidecar,
        build_functional_group_sidecar,
    )

    output = Path("/kaggle/working/molgap-functional-group-sidecar-v1")
    manifest = build_functional_group_sidecar(
        fixed_root(), output, source_commit=commit
    )
    acceptance = accept_functional_group_sidecar(
        output, expected_source_commit=commit
    )
    print(
        json.dumps(
            {
                "complete": manifest["complete"],
                "accepted": acceptance["accepted"],
                "aggregate_sha256": manifest["aggregate_sha256"],
                "covered_atom_fraction": manifest["covered_atom_fraction"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
