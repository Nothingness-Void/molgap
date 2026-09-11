"""Kaggle CPU entry point for the frozen K1 500K cache."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def main() -> None:
    started = time.perf_counter()
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
            "rdkit==2026.3.6",
        ]
    )
    modules = list(Path("/kaggle/input").rglob("src/molgap/pcqm_k1_scale_cache.py"))
    if len(modules) == 1:
        python_root = modules[0].parents[1]
    else:
        archive = find_one("src.zip")
        source = Path("/kaggle/working/_source")
        shutil.unpack_archive(archive, source)
        modules = list(source.rglob("molgap/pcqm_k1_scale_cache.py"))
        if len(modules) != 1:
            raise FileNotFoundError(f"Unexpected source layout: {modules}")
        python_root = modules[0].parents[1]
    sys.path.insert(0, str(python_root))
    commit = find_one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    from rdkit import Chem
    from ogb.utils.mol import smiles2graph

    if Chem.MolFromSmiles("CC") is None or smiles2graph("CC")["num_nodes"] != 2:
        raise RuntimeError("RDKit/OGB molecular parsing preflight failed")
    from molgap.pcqm_k1_scale_cache import build_scale_cache

    output = Path("/kaggle/working/pcqm_k1_scale500k_cache")
    manifest = build_scale_cache(find_one("data.csv"), output, source_commit=commit)
    summary = {
        "format": "molgap-pcqm-k1-scale500k-cache-run-v2",
        "complete": True,
        "source_commit": commit,
        "aggregate_sha256": manifest["aggregate_sha256"],
        "parser_fallback_count": manifest["parser_fallback_count"],
        "parser_fallback_index_sha256": manifest["parser_fallback_index_sha256"],
        "elapsed_s": time.perf_counter() - started,
        "gpu_used": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
