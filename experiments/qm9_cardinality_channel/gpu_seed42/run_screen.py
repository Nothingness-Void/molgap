"""Kaggle2 T4x2 entry point for the frozen cardinality-channel screen."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


CACHE_SHA256 = "80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340"


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_root() -> Path:
    modules = list(Path("/kaggle/input").rglob("src/molgap/qm9_cardinality.py"))
    if len(modules) == 1:
        return modules[0].parents[1]
    archive = find_one("src.zip")
    root = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, root)
    modules = list(root.rglob("molgap/qm9_cardinality.py"))
    if len(modules) != 1:
        raise FileNotFoundError(f"Unexpected source layout: {modules}")
    return modules[0].parents[1]


def install_dependencies() -> None:
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


def launch_worker(
    role: str, output: Path, source_commit: str, cache_root: Path
):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0" if role == "baseline" else "1"
    environment["MOLGAP_CARDINALITY_WORKER"] = role
    environment["MOLGAP_CARDINALITY_OUTPUT"] = str(output)
    environment["MOLGAP_SOURCE_COMMIT"] = source_commit
    environment["MOLGAP_CACHE_ROOT"] = str(cache_root)
    return subprocess.Popen([sys.executable, __file__], env=environment)


def main() -> None:
    role = os.environ.get("MOLGAP_CARDINALITY_WORKER")
    if not role:
        install_dependencies()
    sys.path.insert(0, str(source_root()))
    from molgap.qm9_cardinality import aggregate, run_worker

    output = Path(
        os.environ.get(
            "MOLGAP_CARDINALITY_OUTPUT",
            "/kaggle/working/qm9_cardinality_channel_s42",
        )
    )
    source_commit = os.environ.get("MOLGAP_SOURCE_COMMIT") or find_one(
        "SOURCE_COMMIT.txt"
    ).read_text(encoding="utf-8").strip()
    cache_root = Path(
        os.environ.get("MOLGAP_CACHE_ROOT") or find_one("manifest.json").parent
    )
    cache_manifest = json.loads(
        (cache_root / "manifest.json").read_text(encoding="utf-8")
    )
    if cache_manifest.get("aggregate_sha256") != CACHE_SHA256:
        raise RuntimeError("Frozen pure-2D cache identity changed")
    if role:
        run_worker(
            role,
            cache_root,
            output,
            source_commit=source_commit,
            cache_sha256=CACHE_SHA256,
        )
        return
    import torch

    if torch.cuda.device_count() != 2:
        raise RuntimeError("Frozen screen requires Kaggle T4x2")
    output.mkdir(parents=True, exist_ok=True)
    processes = [
        launch_worker("baseline", output, source_commit, cache_root),
        launch_worker("candidates", output, source_commit, cache_root),
    ]
    codes = [process.wait() for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"Cardinality worker failure: {codes}")
    aggregate(
        output,
        source_commit=source_commit,
        cache_sha256=CACHE_SHA256,
    )


if __name__ == "__main__":
    main()

