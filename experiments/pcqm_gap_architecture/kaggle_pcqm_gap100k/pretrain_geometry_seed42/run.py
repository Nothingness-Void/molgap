"""Kaggle2 entry point for train-only ETKDG geometry denoising."""
import os
import runpy
import shutil
import sys
from pathlib import Path


def source_root() -> Path:
    modules = list(Path("/kaggle/input").rglob("molgap/pcqm_pretraining_runner.py"))
    if len(modules) == 1:
        return modules[0].parents[1]
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one source archive, found {modules}/{archives}")
    root = Path("/kaggle/working/_source")
    shutil.unpack_archive(archives[0], root)
    modules = list(root.rglob("molgap/pcqm_pretraining_runner.py"))
    if len(modules) != 1:
        raise RuntimeError(f"Unexpected source layout: {modules}")
    return modules[0].parents[1]


os.environ["MOLGAP_PRETRAIN_MODE"] = "etkdg_geometry_denoising"
sys.path.insert(0, str(source_root()))
runpy.run_module("molgap.pcqm_pretraining_runner", run_name="__main__")

