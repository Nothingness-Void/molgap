"""Kaggle2 T4x2 entry point for the frozen three-arm denoising screen."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Frozen only after independent CPU-cache acceptance on 2026-09-10.
EXPECTED_CACHE_AGGREGATE_SHA256 = (
    "42bf7d73eb4450235ee4901e021aac267127448505a33096d3d7ea44e08935af"
)
EXPECTED_CACHE_MANIFEST_SHA256 = (
    "f72142c3d33707c821a7fec6f497396eedb264fc311a26797c4cde16a10b3013"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def launch_worker(
    arm: str,
    output: Path,
    source_commit: str,
    cache_root: Path,
    cache_sha: str,
):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0" if arm == "scratch40" else "1"
    environment["MOLGAP_ADAPTIVE_ARM"] = arm
    environment["MOLGAP_ADAPTIVE_OUTPUT"] = str(output)
    environment["MOLGAP_SOURCE_COMMIT"] = source_commit
    environment["MOLGAP_CACHE_ROOT"] = str(cache_root)
    environment["MOLGAP_CACHE_SHA256"] = cache_sha
    return subprocess.Popen([sys.executable, __file__], env=environment)


def worker_main(arm: str) -> None:
    root = source_root()
    verify_source_tree(root)
    sys.path.insert(0, str(root))
    from molgap.qm9_adaptive_denoising import run_worker

    output = Path(os.environ["MOLGAP_ADAPTIVE_OUTPUT"])
    run_worker(
        arm,
        Path(os.environ["MOLGAP_CACHE_ROOT"]),
        output,
        source_commit=os.environ["MOLGAP_SOURCE_COMMIT"],
        cache_sha256=os.environ["MOLGAP_CACHE_SHA256"],
    )


def main() -> None:
    arm = os.environ.get("MOLGAP_ADAPTIVE_ARM")
    if arm:
        worker_main(arm)
        return
    install_dependencies()
    root = source_root()
    verify_source_tree(root)
    sys.path.insert(0, str(root))
    import torch
    from molgap.qm9_adaptive_denoising import aggregate

    if torch.cuda.device_count() != 2:
        raise RuntimeError("Frozen screen requires Kaggle T4x2")
    if EXPECTED_CACHE_AGGREGATE_SHA256.startswith("PENDING"):
        raise RuntimeError("CPU cache acceptance has not been frozen")
    source_commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
    cache_manifests = [
        path
        for path in Path("/kaggle/input").rglob("manifest.json")
        if json.loads(path.read_text()).get("format")
        == "molgap-qm9-adaptive-denoising-cache-v1"
    ]
    if len(cache_manifests) != 1:
        raise RuntimeError(f"Expected one accepted geometry cache, found {cache_manifests}")
    cache_manifest_path = cache_manifests[0]
    cache_root = cache_manifest_path.parent
    cache_manifest = json.loads(cache_manifest_path.read_text())
    cache_sha = cache_manifest["aggregate_sha256"]
    if (
        cache_sha != EXPECTED_CACHE_AGGREGATE_SHA256
        or sha256_file(cache_manifest_path) != EXPECTED_CACHE_MANIFEST_SHA256
    ):
        raise RuntimeError("Geometry cache is not the independently frozen cache")
    acceptance = json.loads((cache_root / "acceptance.json").read_text())
    if (
        acceptance.get("accepted") is not True
        or acceptance.get("cache_aggregate_sha256") != EXPECTED_CACHE_AGGREGATE_SHA256
        or acceptance.get("manifest_sha256") != EXPECTED_CACHE_MANIFEST_SHA256
        or acceptance.get("model_inference_executed") is not False
        or acceptance.get("test_role_read") is not False
    ):
        raise RuntimeError("Independent cache acceptance changed")
    output = Path("/kaggle/working/qm9_adaptive_denoising_s42_v1")
    output.mkdir(parents=True, exist_ok=True)

    scratch = launch_worker(
        "scratch40", output, source_commit, cache_root, cache_sha
    )
    # GPU1 serializes the two denoising arms; every process sees one GPU only.
    fixed = launch_worker(
        "fixed10_gap30", output, source_commit, cache_root, cache_sha
    )
    process = fixed
    fixed_code = process.wait()
    adaptive = None
    adaptive_code = -1
    if fixed_code == 0:
        adaptive = launch_worker(
            "adaptive10_gap30", output, source_commit, cache_root, cache_sha
        )
        process = adaptive
        adaptive_code = process.wait()
    process = scratch
    scratch_code = process.wait()
    if [scratch_code, fixed_code, adaptive_code] != [0, 0, 0]:
        raise RuntimeError(
            "Adaptive denoising worker failure: "
            f"{[scratch_code, fixed_code, adaptive_code]}"
        )
    aggregate(output, source_commit=source_commit, cache_sha256=cache_sha)


if __name__ == "__main__":
    main()
