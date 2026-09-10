"""Kaggle CPU entry point for the label-sealed K1 shadow cache."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_k1_shadow_cache")


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_root() -> Path:
    modules = list(Path("/kaggle/input").rglob("src/molgap/pcqm_shadow.py"))
    if len(modules) == 1:
        return modules[0].parents[1]
    root = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(find_one("src.zip"), root)
    modules = list(root.rglob("molgap/pcqm_shadow.py"))
    if len(modules) != 1:
        raise FileNotFoundError(f"Unexpected source layout: {modules}")
    return modules[0].parents[1]


def main() -> None:
    started = time.perf_counter()
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
    from molgap.pcqm_shadow import build_shadow_cache

    OUT.mkdir(parents=True, exist_ok=True)
    source_commit = find_one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    manifest = build_shadow_cache(
        find_one("data.csv"),
        OUT,
        source_dataset="piero0/pcqm4mv2",
        source_commit=source_commit,
    )
    summary = {
        "format": "molgap-pcqm-k1-shadow-cache-run-v1",
        "complete": True,
        "source_commit": source_commit,
        "aggregate_sha256": manifest["aggregate_sha256"],
        "elapsed_s": time.perf_counter() - started,
        "shadow_labels_read": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "gpu_used": False,
    }
    temporary = OUT / ".run_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT / "run_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
