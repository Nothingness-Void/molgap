"""Thin Kaggle entry point for the paired ContactState seed-42 screen."""
from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path


EXPECTED_SOURCE_COMMIT = "a28862c3d91590b0827c3cfd6d7d2586a9c6ab47"


def source_python_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("molgap/pcqm_local_global_runner.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one runner source, found {matches}/{archives}")
    extracted = Path("/kaggle/working/_molgap_contact_graphstate_source")
    shutil.unpack_archive(archives[0], extracted)
    modules = list(extracted.rglob("molgap/pcqm_local_global_runner.py"))
    if len(modules) != 1:
        raise RuntimeError(f"Unexpected source archive layout: {modules}")
    return modules[0].parents[1]


os.environ["MOLGAP_LOCAL_GLOBAL_RUN_MODE"] = "contact_graphstate"
os.environ["MOLGAP_LOCAL_GLOBAL_SEED"] = "42"
os.environ["MOLGAP_LOCAL_GLOBAL_OUTPUT"] = (
    "/kaggle/working/pcqm_gap100k_contact_graphstate_seed42"
)
os.environ["MOLGAP_EXPECTED_MODEL_SOURCE_COMMIT"] = EXPECTED_SOURCE_COMMIT
sys.path.insert(0, str(source_python_root()))
runpy.run_module("molgap.pcqm_local_global_runner", run_name="__main__")
