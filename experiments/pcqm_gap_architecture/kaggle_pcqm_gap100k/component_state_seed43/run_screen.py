"""Kaggle1 T4x2 entry point for ComponentState seed-43 confirmation."""
from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path


EXPECTED_SOURCE_COMMIT = "c3c54e09c783d8718cb3129b08e33fd8de3525cd"
EXPECTED_CACHE_SHA256 = "434dc60b40064eb30eab279e85dcc7f43ef7763cda9e9e42d7a8630f35de57fc"


def source_python_root() -> Path:
    modules = list(Path("/kaggle/input").rglob("molgap/pcqm_local_global_runner.py"))
    if len(modules) == 1:
        return modules[0].parents[1]
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(
            f"Expected one source tree/archive, found {modules}/{archives}"
        )
    extracted = Path("/kaggle/working/_molgap_componentstate_confirm_source")
    shutil.unpack_archive(archives[0], extracted)
    extracted_modules = list(extracted.rglob("molgap/pcqm_local_global_runner.py"))
    if len(extracted_modules) != 1:
        raise RuntimeError(f"Unexpected source archive layout: {extracted_modules}")
    return extracted_modules[0].parents[1]


os.environ["MOLGAP_LOCAL_GLOBAL_RUN_MODE"] = "conjugated_component_confirmation"
os.environ["MOLGAP_LOCAL_GLOBAL_SEED"] = "43"
os.environ["MOLGAP_LOCAL_GLOBAL_OUTPUT"] = (
    "/kaggle/working/pcqm_gap100k_componentstate_confirmation_seed43"
)
os.environ["MOLGAP_EXPECTED_MODEL_SOURCE_COMMIT"] = EXPECTED_SOURCE_COMMIT
os.environ["MOLGAP_EXPECTED_COMPONENT_SOURCE_COMMIT"] = EXPECTED_SOURCE_COMMIT
os.environ["MOLGAP_EXPECTED_COMPONENT_CACHE_SHA256"] = EXPECTED_CACHE_SHA256
sys.path.insert(0, str(source_python_root()))
runpy.run_module("molgap.pcqm_local_global_runner", run_name="__main__")
