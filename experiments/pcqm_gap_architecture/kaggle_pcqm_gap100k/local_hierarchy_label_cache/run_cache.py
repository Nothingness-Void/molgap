"""Kaggle CPU entry point for the PCQM-100K local hierarchy label sidecar."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_gap100k_local_hierarchy_labels")


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


def find_parent_cache() -> Path:
    matches = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("format") == "molgap-pcqm-gap100k-etkdg-geometry-cache-v1":
            matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one parent geometry cache, found {matches}")
    return matches[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torch-geometric==2.6.1",
                "rdkit==2025.3.5",
            ]
        )
        sys.path.insert(0, str(source_root()))
        from molgap.pcqm_local_hierarchy import (
            accept_local_label_cache,
            build_local_label_cache,
        )

        source_commit = find_one("PCQM_GAP100K_SOURCE_COMMIT.txt").read_text().strip()
        manifest = build_local_label_cache(
            find_one("data.csv"),
            find_parent_cache(),
            OUT,
            source_commit=source_commit,
        )
        acceptance = accept_local_label_cache(
            OUT, expected_source_commit=source_commit
        )
        print(json.dumps({"manifest": manifest, "acceptance": acceptance}, indent=2))
    except Exception as error:
        (OUT / "failure.json").write_text(
            json.dumps(
                {"type": type(error).__name__, "message": str(error)}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
