"""Kaggle2 T4x2 entry point for the frozen PCQM Fourier-Edge transfer."""
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
    modules = list(Path("/kaggle/input").rglob("src/molgap/pcqm_fourier_edge.py"))
    if len(modules) == 1:
        return modules[0].parents[1]
    archive = find_one("src.zip")
    root = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, root)
    modules = list(root.rglob("molgap/pcqm_fourier_edge.py"))
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


def launch_worker(role: str, output: Path, source_commit: str):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0" if role == "anchor" else "1"
    environment["MOLGAP_PCQM_FOURIER_WORKER"] = role
    environment["MOLGAP_PCQM_FOURIER_OUTPUT"] = str(output)
    environment["MOLGAP_SOURCE_COMMIT"] = source_commit
    return subprocess.Popen([sys.executable, __file__], env=environment)


def main() -> None:
    role = os.environ.get("MOLGAP_PCQM_FOURIER_WORKER")
    if not role:
        install_dependencies()
    sys.path.insert(0, str(source_root()))
    from molgap.pcqm_fourier_edge import aggregate, run_worker

    output = Path(
        os.environ.get(
            "MOLGAP_PCQM_FOURIER_OUTPUT",
            "/kaggle/working/pcqm_gap100k_fourier_edge_s42",
        )
    )
    source_commit = os.environ.get("MOLGAP_SOURCE_COMMIT") or find_one(
        "SOURCE_COMMIT.txt"
    ).read_text(encoding="utf-8").strip()
    if role:
        run_worker(role, output, source_commit=source_commit)
        return

    import torch

    if torch.cuda.device_count() != 2:
        raise RuntimeError("Frozen transfer requires Kaggle T4x2")
    output.mkdir(parents=True, exist_ok=True)
    processes = [
        launch_worker("anchor", output, source_commit),
        launch_worker("edge", output, source_commit),
    ]
    codes = [process.wait() for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"PCQM Fourier-Edge worker failure: {codes}")
    aggregate(output, source_commit=source_commit)


if __name__ == "__main__":
    main()
