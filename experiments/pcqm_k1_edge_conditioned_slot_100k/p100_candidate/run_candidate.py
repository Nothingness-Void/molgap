"""Kaggle2 entry point for the single K1 edge-conditioned candidate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


MODE = "neural_atom_k1_edge_conditioned_slot"


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def pin_one_visible_gpu() -> None:
    """Expose exactly one assigned accelerator to preserve the v4 contract."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    first = visible.split(",", maxsplit=1)[0].strip() or "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = first


def ensure_compatible_torch() -> None:
    probe_code = (
        "import torch; "
        "available=torch.cuda.is_available(); "
        "cap=torch.cuda.get_device_capability(0) if available else (-1,-1); "
        "arch=f'sm_{cap[0]}{cap[1]}'; "
        "raise SystemExit(0 if available and arch in torch.cuda.get_arch_list() else 3)"
    )
    probe = subprocess.run(
        [sys.executable, "-c", probe_code],
        check=False,
    )
    if probe.returncode == 0:
        return
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "--force-reinstall",
            "torch==2.4.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]
    )
    verified = subprocess.run(
        [sys.executable, "-c", probe_code],
        check=False,
    )
    if verified.returncode:
        raise RuntimeError("Installed PyTorch does not support the assigned CUDA accelerator")


def install_dependencies() -> None:
    ensure_compatible_torch()
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "torch-geometric==2.6.1", "ogb==1.3.6"]
    )


def source_root() -> Path:
    modules = list(Path("/kaggle/input").rglob("src/molgap/pcqm_k1_variants.py"))
    if len(modules) == 1:
        return modules[0].parents[1]
    archive = find_one("source_payload.bin")
    root = Path("/kaggle/working/_molgap_source")
    if not root.exists():
        shutil.unpack_archive(archive, root, format="gztar")
    modules = list(root.rglob("src/molgap/pcqm_k1_variants.py"))
    if len(modules) != 1:
        raise FileNotFoundError(f"Unexpected source layout: {modules}")
    return modules[0].parents[1]


def verify_source(root: Path) -> tuple[str, str]:
    archive = find_one("source_payload.bin")
    digest = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
    commit = find_one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest or len(commit) != 40:
        raise RuntimeError("Source identity mismatch")
    inventory = json.loads(find_one("SOURCE_FILES.json").read_text(encoding="utf-8"))["files"]
    for item in inventory:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "src":
            raise RuntimeError("Invalid source inventory path")
        mounted = root.joinpath(*relative.parts[1:])
        if hashlib.sha256(mounted.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Mounted source differs: {relative}")
    return commit, digest


def main() -> None:
    pin_one_visible_gpu()
    install_dependencies()
    root = source_root()
    commit, digest = verify_source(root)
    sys.path.insert(0, str(root))
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Candidate requires exactly one visible CUDA accelerator, got {torch.cuda.device_count()}"
        )
    capability = torch.cuda.get_device_capability(0)
    architecture = f"sm_{capability[0]}{capability[1]}"
    if architecture not in torch.cuda.get_arch_list():
        raise RuntimeError(
            f"PyTorch lacks {architecture} support for {torch.cuda.get_device_name(0)}"
        )
    print(
        json.dumps(
            {
                "event": "gpu_runtime",
                "device": torch.cuda.get_device_name(0),
                "capability": list(capability),
                "visible_device_count": torch.cuda.device_count(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    from molgap.pcqm_k1_variants_runner import train_arm

    train_arm(
        MODE,
        Path(f"/kaggle/working/pcqm_k1_edge_conditioned_slot/{MODE}"),
        source_commit=commit,
        source_archive_sha256=digest,
    )


if __name__ == "__main__":
    main()
