"""Freeze the TRAIN-only engineering benchmark, without publishing RML."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.v4_bundle import build_v4_source_bundle
from molgap.v4_runtime import validate_standard_source_bundle


def prepare(root, staging, input_root, initial_path):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    names = ["__init__", "constants", "k1_local_speed", "k1_screen_training",
        "k1_execution_layout", "k1_loader_reuse", "packed_graph_dataset",
        "qm9_neural_atom", "pcqm_gap_architecture", "gps", "edge_state_gps",
        "screen_policy", "training_reproducibility", "v4_runtime", "pcqm_wedge"]
    source = build_v4_source_bundle(repo_root=root,
        relative_paths=[f"src/molgap/{name}.py" for name in names],
        output_dir=staging / "package", source_commit=commit)
    frozen = staging / "frozen"
    frozen.mkdir()
    validate_standard_source_bundle(Path(source["archive"]), source["archive_sha256"], commit)
    with tarfile.open(source["archive"], "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                raise ValueError("Source inventory must contain regular files only")
            target = frozen / member.name
            target.resolve().relative_to(frozen.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as handle, target.open("wb") as output:
                shutil.copyfileobj(handle, output)
    for name in ("SOURCE_COMMIT.txt", "SOURCE_FILES.json", "SOURCE_ARCHIVE_SHA256.txt"):
        shutil.copyfile(staging / "package" / name, frozen / name)
    config = {"format": "molgap-k1-local-epoch-speed-v2", "epochs": 10,
        "allocation_seconds": 3600, "schedule_epochs": 40, "batch_size": 128,
        "precision": "fp32-tf32-off", "development_access": False, "rml_recording": False,
        "source_commit": commit, "source_archive": source["archive"],
        "source_package_sha256": source["archive_sha256"],
        "source_inventory_sha256": sha256_file(frozen / "SOURCE_FILES.json"),
        "input_root": str(input_root), "initial_path": str(initial_path),
        "initial_format": "molgap-k1-single-ema-initial-v1",
        "initial_file_sha256": "0ce2cb3206a7a59a059b5efb09922a5d6e4478058ce2dbfec1434f3086fb5a80",
        "initial_tensor_sha256": "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd"}
    atomic_json(staging / "config.json", config)
    print(f"Frozen TRAIN-only speed benchmark: {staging}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--initial-path", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo_root.resolve(), args.staging.resolve(), args.input_root, args.initial_path)
