"""Kaggle T4x2 entry point for the frozen K1 500K paired bridge."""
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
    modules = list(Path("/kaggle/input").rglob("src/molgap/pcqm_k1_scale_runner.py"))
    if len(modules) == 1:
        return modules[0].parents[1]
    archive = find_one("src.zip")
    root = Path("/kaggle/working/_source")
    shutil.unpack_archive(archive, root)
    modules = list(root.rglob("molgap/pcqm_k1_scale_runner.py"))
    if len(modules) != 1:
        raise FileNotFoundError(f"Unexpected source layout: {modules}")
    return modules[0].parents[1]


def launch(arm: str, output: Path, commit: str, cache_sha: str):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0" if arm == "full_gps" else "1"
    environment["MOLGAP_SCALE_ARM"] = arm
    environment["MOLGAP_SCALE_OUTPUT"] = str(output)
    environment["MOLGAP_SOURCE_COMMIT"] = commit
    environment["MOLGAP_CACHE_SHA256"] = cache_sha
    return subprocess.Popen([sys.executable, __file__], env=environment)


def main() -> None:
    arm = os.environ.get("MOLGAP_SCALE_ARM")
    if not arm:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "torch-geometric==2.6.1", "ogb==1.3.6"])
    sys.path.insert(0, str(source_root()))
    from molgap.pcqm_k1_scale_runner import aggregate, run_worker

    output = Path(os.environ.get("MOLGAP_SCALE_OUTPUT", "/kaggle/working/pcqm_k1_scale500k_s42"))
    commit = os.environ.get("MOLGAP_SOURCE_COMMIT") or find_one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    cache_sha = os.environ.get("MOLGAP_CACHE_SHA256") or find_one("CACHE_AGGREGATE_SHA256.txt").read_text(encoding="utf-8").strip()
    if arm:
        run_worker(arm, output, source_commit=commit, cache_sha256=cache_sha)
        return
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Frozen scale bridge requires Kaggle T4x2")
    output.mkdir(parents=True, exist_ok=True)
    workers = [launch(name, output, commit, cache_sha) for name in ("full_gps", "neural_atom_k1")]
    codes = [worker.wait() for worker in workers]
    if codes != [0, 0]:
        raise RuntimeError(f"Scale worker failure: {codes}")
    aggregate(output, source_commit=commit, cache_sha256=cache_sha)


if __name__ == "__main__":
    main()
