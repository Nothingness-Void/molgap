"""Kaggle2 CPU stage: build the R8 train/validation multihop cache."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working/qm9_multihop_r8_cache")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_python_root() -> Path:
    matches = list(Path("/kaggle/input").rglob("src/molgap/qm9_screen.py"))
    if len(matches) == 1:
        return matches[0].parents[1]
    archive = find_one("src.zip")
    extracted = Path("/kaggle/working/_molgap_source")
    shutil.unpack_archive(archive, extracted)
    modules = list(extracted.rglob("molgap/qm9_screen.py"))
    if len(modules) != 1:
        raise FileNotFoundError(f"Unexpected source archive layout: {modules}")
    return modules[0].parents[1]


def install_dependencies() -> None:
    import torch

    version = torch.__version__.split("+")[0]
    cuda = (torch.version.cuda or "").replace(".", "")
    wheel_version = "2.7.0" if version.startswith("2.7.") else version
    suffix = f"+cu{cuda}" if cuda else "+cpu"
    index = f"https://data.pyg.org/whl/torch-{wheel_version}{suffix}.html"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch-geometric==2.6.1",
            "torch_cluster",
            "torch_scatter",
            "-f",
            index,
        ]
    )


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        python_root = source_python_root()
        sys.path.insert(0, str(python_root))
        install_dependencies()
        from molgap.qm9_screen import build_qm9_multihop_screen_cache

        source_commit = find_one("R8_SOURCE_COMMIT.txt").read_text().strip()
        rwse_acceptance_path = find_one("acceptance.json")
        source_cache = rwse_acceptance_path.parents[2]
        acceptance = build_qm9_multihop_screen_cache(
            train_size=30_000,
            validation_size=3_000,
            test_size=3_000,
            split_seed=42,
            max_distance=4,
            shard_size=2_000,
            cache_dir=OUT,
            source_cache_dir=source_cache,
        )
        if acceptance.get("test_role_read") is not False:
            raise RuntimeError("R8 cache reports a test-role read")
        result = {
            "format": "molgap-multihop-edgestate-r8-prep-result-v1",
            "experiment": "multihop_edge_state_r8_qm9_prep",
            "source_commit": source_commit,
            "source_rwse_acceptance": str(rwse_acceptance_path),
            "gpu_used": False,
            "test_role_read": False,
            "acceptance": acceptance,
            "elapsed_s": time.perf_counter() - started,
        }
        atomic_json(Path("/kaggle/working/prep_result.json"), result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            Path("/kaggle/working/prep_failure.json"),
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
