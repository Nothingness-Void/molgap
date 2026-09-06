"""Kaggle CPU entry point for the deterministic conjugated-component cache."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


OUT = Path("/kaggle/working/pcqm_gap100k_componentstate_cache_s43")
GEOMETRY_FORMAT = "molgap-pcqm-gap100k-etkdg-geometry-cache-v1"


def source_python_root() -> Path:
    archives = list(Path("/kaggle/input").rglob("src.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one source archive, found {archives}")
    extracted = Path("/kaggle/working/_molgap_componentstate_source")
    shutil.unpack_archive(archives[0], extracted)
    modules = list(extracted.rglob("molgap/pcqm_conjugated_cache.py"))
    if len(modules) != 1:
        raise RuntimeError(f"Unexpected source archive layout: {modules}")
    return modules[0].parents[1]


def source_commit() -> str:
    markers = list(Path("/kaggle/input").rglob("PCQM_GAP100K_SOURCE_COMMIT.txt"))
    if len(markers) != 1:
        raise RuntimeError(f"Expected one source marker, found {markers}")
    return markers[0].read_text(encoding="utf-8").strip()


def geometry_root() -> Path:
    matches = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("format") == GEOMETRY_FORMAT:
            matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one geometry cache, found {matches}")
    return matches[0]


def builder_module():
    matches = list(Path("/kaggle/input").rglob("build_kunshan_conjugated_cache.py"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one component cache builder, found {matches}")
    spec = importlib.util.spec_from_file_location("component_cache_builder", matches[0])
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "torch-geometric==2.6.1"]
    )
    sys.path.insert(0, str(source_python_root()))
    result = builder_module().build(geometry_root(), OUT, source_commit())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
