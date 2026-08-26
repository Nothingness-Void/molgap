"""Kaggle CPU preparation of the official-train-derived PCQM Gap100K cache."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_gap100k_cache")
SOURCE_DATASET = "piero0/pcqm4mv2"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_python_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("src/molgap/pcqm_gap_data.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archive = find_one("src.zip")
    extracted = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, extracted)
    modules = list(extracted.rglob("molgap/pcqm_gap_data.py"))
    if len(modules) != 1:
        raise FileNotFoundError(f"Unexpected source archive layout: {modules}")
    return modules[0].parents[1]


def install_dependencies() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "ogb==1.3.6",
            "torch-geometric==2.6.1",
        ]
    )


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        install_dependencies()
        sys.path.insert(0, str(source_python_root()))
        source_commit = find_one("PCQM_GAP100K_SOURCE_COMMIT.txt").read_text().strip()
        source_csv = find_one("data.csv")
        from molgap.pcqm_gap_data import build_pcqm_gap_screen_cache

        manifest = build_pcqm_gap_screen_cache(
            source_csv,
            OUT,
            source_dataset=SOURCE_DATASET,
            source_commit=source_commit,
        )
        summary = {
            "format": "molgap-pcqm-gap100k-prep-run-v1",
            "complete": True,
            "source_commit": source_commit,
            "aggregate_sha256": manifest["aggregate_sha256"],
            "elapsed_s": time.perf_counter() - started,
            "gpu_used": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        }
        atomic_json(OUT / "run_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()

