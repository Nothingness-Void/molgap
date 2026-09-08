"""Single-GPU Kaggle entry point for the GraphState64 control."""
from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path


EXPECTED_SOURCE_COMMIT = "c413fc13b8b15e659e90853c429119189577c379"


def source_python_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("src/molgap/pcqm_graph_state_width_single_runner.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one runner source, found {matches}/{archives}")
    extracted = Path("/kaggle/working/_molgap_graph_state_width_source")
    shutil.unpack_archive(archives[0], extracted)
    modules = list(extracted.rglob("molgap/pcqm_graph_state_width_single_runner.py"))
    if len(modules) != 1:
        raise RuntimeError(f"Unexpected source archive layout: {modules}")
    return modules[0].parents[1]


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MOLGAP_LOCAL_GLOBAL_RUN_MODE"] = "graph_state_width"
os.environ["MOLGAP_LOCAL_GLOBAL_SEED"] = "42"
os.environ["MOLGAP_EXPECTED_GPU_TOKEN"] = "Tesla"
os.environ["MOLGAP_SINGLE_CANDIDATE"] = (
    "ogb_distance_angle_triangle_edge_state_graph_state9"
)
os.environ["MOLGAP_LOCAL_GLOBAL_OUTPUT"] = (
    "/kaggle/working/pcqm_gap100k_graph_state_width_baseline_seed42"
)
os.environ["MOLGAP_EXPECTED_MODEL_SOURCE_COMMIT"] = EXPECTED_SOURCE_COMMIT
sys.path.insert(0, str(source_python_root()))
runpy.run_module("molgap.pcqm_graph_state_width_single_runner", run_name="__main__")
