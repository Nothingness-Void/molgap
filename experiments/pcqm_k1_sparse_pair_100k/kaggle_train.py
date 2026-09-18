"""Kaggle T4x2 wrapper for one isolated K1 sparse-pair training worker."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
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
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha:
        raise RuntimeError("Source bundle SHA-256 changed")
    return archive, source_commit, expected_sha


def _accepted_profile(source_commit: str, source_sha: str) -> Path:
    matches = []
    for path in Path("/kaggle/input").rglob("profile.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("format")
            == "molgap-k1-sparse-pair-100k-profile-v1"
            and payload.get("status") == "accepted"
            and payload.get("source_commit") == source_commit
            and payload.get("source_archive_sha256") == source_sha
            and payload.get("runtime_certificate", {}).get("platform_id")
            == "kaggle3-t4"
        ):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one matching accepted profile, found {matches}")
    return matches[0]


def _safe_extract(archive: Path, output: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise RuntimeError(f"Unsafe source archive member: {member.name}")
        handle.extractall(output)


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
    profile = _accepted_profile(source_commit, archive_sha)
    runtime = Path("/kaggle/working/k1_sparse_pair_source")
    runtime.mkdir(parents=True, exist_ok=True)
    _safe_extract(archive, runtime)

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
        "train",
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
        "--profile",
        str(profile),
        "--output",
        "/kaggle/working/k1_sparse_pair_training",
        "--platform-id",
        "kaggle3-t4",
    ]
    result = subprocess.run(command, env=environment, check=False)
    if result.returncode:
        raise RuntimeError(f"Training worker failed with code {result.returncode}")


if __name__ == "__main__":
    main()
