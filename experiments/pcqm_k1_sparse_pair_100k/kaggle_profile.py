"""Kaggle T4x2 wrapper that profiles one isolated T4 worker."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def _fixed_dataset() -> tuple[Path, Path]:
    matches = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("format")
            == "molgap-pcqm4mv2-kaggle-fixed-subset-v1"
            and payload.get("identity", {}).get("name") == "ogb-train-100k"
        ):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one fixed 100K manifest, found {matches}")
    return matches[0].parent, matches[0]


def _source_bundle() -> tuple[Path, str, str]:
    archives = [
        path
        for name in ("source_bundle.bin", "source.tar.gz")
        for path in Path("/kaggle/input").rglob(name)
    ]
    if len(archives) != 1:
        raise RuntimeError(f"Expected one source bundle, found {archives}")
    archive = archives[0]
    source_commit = (archive.parent / "SOURCE_COMMIT.txt").read_text(
        encoding="utf-8"
    ).strip()
    expected_sha = (archive.parent / "SOURCE_ARCHIVE_SHA256.txt").read_text(
        encoding="utf-8"
    ).strip()
    observed_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if observed_sha != expected_sha:
        raise RuntimeError("Source bundle SHA-256 changed")
    return archive, source_commit, expected_sha


def main() -> None:
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    print("GPU allocation:", names, flush=True)
    if len(names) != 2 or any("T4" not in name for name in names):
        raise RuntimeError(f"Expected Kaggle T4x2, received {names}")
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
        ]
    )
    dataset_root, manifest = _fixed_dataset()
    archive, source_commit, archive_sha = _source_bundle()
    runtime = Path("/kaggle/working/k1_sparse_pair_source")
    runtime.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(runtime)

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(runtime / "src"),
            "PYTHONHASHSEED": "42",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
    )
    command = [
        sys.executable,
        "-u",
        str(runtime / "experiments/pcqm_k1_sparse_pair_100k/run.py"),
        "profile",
        "--dataset-root",
        str(dataset_root),
        "--manifest",
        str(manifest),
        "--source-archive",
        str(archive),
        "--source-archive-sha256",
        archive_sha,
        "--source-commit",
        source_commit,
        "--output",
        "/kaggle/working/k1_sparse_pair_profile",
        "--platform-id",
        "kaggle3-t4",
    ]
    result = subprocess.run(command, env=environment, check=False)
    if result.returncode:
        raise RuntimeError(f"Profile worker failed with code {result.returncode}")


if __name__ == "__main__":
    main()
