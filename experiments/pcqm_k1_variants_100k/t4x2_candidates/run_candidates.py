"""Kaggle2 T4x2 entry point for independent K1-G and K1-R arms."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


MODES = ("neural_atom_k1_g", "neural_atom_k1_r")


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_root() -> Path:
    archive = find_one("src.zip")
    root = Path("/kaggle/working/_molgap_source")
    if not root.exists():
        shutil.unpack_archive(archive, root)
    return root / "src"


def install_dependencies() -> None:
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


def launch(mode: str, device: int, output: Path, commit: str, archive_sha: str):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device)
    environment["MOLGAP_K1_VARIANT_MODE"] = mode
    environment["MOLGAP_K1_VARIANT_OUTPUT"] = str(output)
    environment["MOLGAP_SOURCE_COMMIT"] = commit
    environment["MOLGAP_SOURCE_ARCHIVE_SHA256"] = archive_sha
    return subprocess.Popen([sys.executable, __file__], env=environment)


def main() -> None:
    mode = os.environ.get("MOLGAP_K1_VARIANT_MODE")
    if not mode:
        install_dependencies()
    sys.path.insert(0, str(source_root()))
    from molgap.pcqm_k1_variants_runner import train_arm

    output = Path(
        os.environ.get(
            "MOLGAP_K1_VARIANT_OUTPUT",
            "/kaggle/working/pcqm_k1_v4_candidates",
        )
    )
    commit = os.environ.get("MOLGAP_SOURCE_COMMIT") or find_one(
        "SOURCE_COMMIT.txt"
    ).read_text(encoding="utf-8").strip()
    archive_sha = os.environ.get("MOLGAP_SOURCE_ARCHIVE_SHA256") or find_one(
        "SOURCE_ARCHIVE_SHA256.txt"
    ).read_text(encoding="utf-8").strip()
    if mode:
        train_arm(
            mode,
            output,
            source_commit=commit,
            source_archive_sha256=archive_sha,
        )
        return

    import torch

    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(index) for index in range(2)
    ):
        raise RuntimeError("Candidate screen requires Kaggle T4x2")
    output.mkdir(parents=True, exist_ok=True)
    processes = [
        launch(mode_name, device, output / mode_name, commit, archive_sha)
        for device, mode_name in enumerate(MODES)
    ]
    codes = [process.wait() for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"K1 candidate worker failure: {codes}")


if __name__ == "__main__":
    main()
