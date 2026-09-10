"""Kaggle entry point for the frozen one-time K1 shadow audit."""
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
    modules = list(Path("/kaggle/input").rglob("src/molgap/pcqm_k1_shadow_audit.py"))
    if len(modules) == 1:
        return modules[0].parents[1]
    archive = find_one("src.zip")
    root = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, root)
    modules = list(root.rglob("molgap/pcqm_k1_shadow_audit.py"))
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
    from molgap.pcqm_k1_shadow_audit import run_audit

    output = Path("/kaggle/working/pcqm_k1_shadow_audit")
    output.mkdir(parents=True, exist_ok=True)
    source_commit = find_one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    run_audit(find_one("data.csv"), output, source_commit=source_commit)


if __name__ == "__main__":
    main()
