"""Kaggle T4x2 entry point for paired EdgeState local-hierarchy training."""
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


def source_root() -> Path:
    modules = list(Path("/kaggle/input").rglob("molgap/pcqm_local_hierarchy.py"))
    if len(modules) == 1:
        return modules[0].parents[1]
    archive = find_one("src.zip")
    root = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, root)
    modules = list(root.rglob("molgap/pcqm_local_hierarchy.py"))
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
            "--no-deps",
            "torch-geometric==2.6.1",
            "ogb==1.3.6",
        ]
    )
    sys.path.insert(0, str(source_root()))
    from molgap.pcqm_local_hierarchy import run_paired_screen, run_worker

    output = Path(
        os.environ.get(
            "MOLGAP_LOCAL_HIERARCHY_OUTPUT",
            "/kaggle/working/pcqm_gap100k_edgestate_local_hierarchy_s42",
        )
    )
    source_commit = os.environ.get("MOLGAP_SOURCE_COMMIT") or find_one(
        "PCQM_GAP100K_SOURCE_COMMIT.txt"
    ).read_text().strip()
    label_sha256 = os.environ.get("MOLGAP_LOCAL_LABEL_SHA256") or find_one(
        "EXPECTED_LOCAL_LABEL_SHA256.txt"
    ).read_text().strip()
    role = os.environ.get("MOLGAP_LOCAL_HIERARCHY_WORKER")
    if role:
        run_worker(
            role,
            output,
            label_sha256=label_sha256,
            source_commit=source_commit,
        )
    else:
        run_paired_screen(
            output,
            label_sha256=label_sha256,
            source_commit=source_commit,
        )


if __name__ == "__main__":
    main()
