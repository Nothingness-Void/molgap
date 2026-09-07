"""Kaggle1 entry point for the hop-path GraphState seed-42 screen."""
from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path


EXPECTED_SOURCE_COMMIT = "0253aaa2061d0e7ac655fc8b6d9effd283722096"
EXPECTED_HOP_PATH_SHA256 = (
    "6d3a7df67cfadc26db2a38c006fd20e0d4294c16968fa92b01739daf8527993e"
)


def source_python_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("molgap/pcqm_local_global_runner.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one runner source, found {matches}/{archives}")
    extracted = Path("/kaggle/working/_molgap_hop_path_gpu_source")
    shutil.unpack_archive(archives[0], extracted)
    modules = list(extracted.rglob("molgap/pcqm_local_global_runner.py"))
    if len(modules) != 1:
        raise RuntimeError(f"Unexpected source archive layout: {modules}")
    return modules[0].parents[1]


os.environ["MOLGAP_LOCAL_GLOBAL_RUN_MODE"] = "hop_path_graphstate"
os.environ["MOLGAP_LOCAL_GLOBAL_SEED"] = "42"
os.environ["MOLGAP_LOCAL_GLOBAL_OUTPUT"] = (
    "/kaggle/working/pcqm_gap100k_hop_path_graphstate_seed42"
)
os.environ["MOLGAP_EXPECTED_MODEL_SOURCE_COMMIT"] = EXPECTED_SOURCE_COMMIT
os.environ["MOLGAP_EXPECTED_HOP_PATH_SOURCE_COMMIT"] = EXPECTED_SOURCE_COMMIT
os.environ["MOLGAP_EXPECTED_HOP_PATH_CACHE_SHA256"] = EXPECTED_HOP_PATH_SHA256
sys.path.insert(0, str(source_python_root()))
runpy.run_module("molgap.pcqm_local_global_runner", run_name="__main__")
