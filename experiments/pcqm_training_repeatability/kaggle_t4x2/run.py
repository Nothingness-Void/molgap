"""Thin Kaggle entry point for the packaged repeatability runner."""
import runpy
import shutil
import sys
from pathlib import Path


modules = list(Path("/kaggle/input").rglob("molgap/pcqm_repeatability_runner.py"))
if len(modules) == 1:
    source = modules[0].parents[1]
else:
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(
            f"Expected one source tree or archive, received {modules}/{archives}"
        )
    root = Path("/kaggle/working/_repeatability_source")
    shutil.unpack_archive(archives[0], root)
    source = root / "source" / "src"
sys.path.insert(0, str(source))
runpy.run_module("molgap.pcqm_repeatability_runner", run_name="__main__")
