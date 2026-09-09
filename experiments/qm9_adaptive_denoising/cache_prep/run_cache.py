"""Kaggle CPU entry point for the immutable QM9 ETKDG geometry cache."""
from __future__ import annotations

import shutil
import hashlib
import subprocess
import sys
from pathlib import Path


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_root() -> Path:
    modules = list(
        Path("/kaggle/input").rglob("src/molgap/qm9_adaptive_denoising.py")
    )
    if len(modules) == 1:
        return modules[0].parents[1]
    archive = find_one("src.zip")
    root = Path("/kaggle/working/_molgap_adaptive_denoising_source")
    shutil.unpack_archive(archive, root)
    modules = list(root.rglob("molgap/qm9_adaptive_denoising.py"))
    if len(modules) != 1:
        raise FileNotFoundError(f"Unexpected source layout: {modules}")
    return modules[0].parents[1]


def verify_source_tree(root: Path) -> None:
    digest = hashlib.sha256()
    package = root / "molgap"
    files = sorted(
        (
            (path.relative_to(package).as_posix(), path)
            for path in package.rglob("*")
            if path.is_file()
        ),
        key=lambda item: item[0],
    )
    for relative, path in files:
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    expected = find_one("SOURCE_TREE_SHA256.txt").read_text().strip()
    if digest.hexdigest() != expected:
        raise RuntimeError("Packaged source tree hash changed")


def main() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            # RDKit 2023.09 wheels use the NumPy 1.x C ABI.  Kaggle's
            # Python 3.12 CPU image currently starts with NumPy 2.x.
            "numpy==1.26.4",
            "rdkit==2023.9.6",
            "torch-geometric==2.6.1",
            "ogb==1.3.6",
        ]
    )
    root = source_root()
    verify_source_tree(root)
    sys.path.insert(0, str(root))
    from molgap.qm9_adaptive_denoising import build_cache

    source_commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
    build_cache(
        Path("/kaggle/input"),
        Path("/kaggle/working/molgap-qm9-adaptive-denoising-cache-v1"),
        source_commit=source_commit,
    )


if __name__ == "__main__":
    main()
