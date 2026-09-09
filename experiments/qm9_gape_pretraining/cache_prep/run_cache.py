"""Kaggle CPU entry point for the immutable transferable QM9 cache."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_root() -> Path:
    modules = list(Path("/kaggle/input").rglob("src/molgap/qm9_gape.py"))
    if len(modules) == 1:
        return modules[0].parents[1]
    archive = find_one("src.zip")
    root = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, root)
    modules = list(root.rglob("molgap/qm9_gape.py"))
    if len(modules) != 1:
        raise FileNotFoundError(f"Unexpected source layout: {modules}")
    return modules[0].parents[1]


def main() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "rdkit==2023.9.6",
            "torch-geometric==2.6.1",
            "ogb==1.3.6",
        ]
    )
    sys.path.insert(0, str(source_root()))
    from molgap.qm9_local_hierarchy import build_cache

    source_commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
    build_cache(
        Path("/kaggle/working/molgap-qm9-gape-cache-v1"),
        source_commit=source_commit,
    )


if __name__ == "__main__":
    main()

