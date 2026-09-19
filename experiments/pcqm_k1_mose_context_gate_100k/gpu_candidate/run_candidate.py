"""Kaggle1 entry point for the molecule-context MoSE gate V5 screen."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


MODE = "neural_atom_k1_rwse_mose_context_gate"


def find_one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {matches}")
    return matches[0]


def pin_one_visible_gpu() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",", maxsplit=1)[0].strip() or "0"


def ensure_compatible_torch() -> None:
    probe = "import torch; a=torch.cuda.is_available(); c=torch.cuda.get_device_capability(0) if a else (-1,-1); s=f'sm_{c[0]}{c[1]}'; raise SystemExit(0 if a and s in torch.cuda.get_arch_list() else 3)"
    if subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0:
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu121"])
    if subprocess.run([sys.executable, "-c", probe], check=False).returncode:
        raise RuntimeError("Installed PyTorch cannot execute the assigned accelerator")


def install_dependencies() -> None:
    ensure_compatible_torch()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "torch-geometric==2.6.1", "ogb==1.3.6"])


def source_root() -> Path:
    archive = find_one("source_payload.bin")
    root = Path("/kaggle/working/_molgap_mose_context_source")
    if not root.exists():
        shutil.unpack_archive(archive, root, format="gztar")
    return root


def verify_source(root: Path) -> tuple[str, str]:
    archive = find_one("source_payload.bin")
    digest = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest or len(commit) != 40:
        raise RuntimeError("Source identity mismatch")
    for item in json.loads(find_one("SOURCE_FILES.json").read_text())["files"]:
        if hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Source file hash mismatch: {item['path']}")
    return commit, digest


def main() -> None:
    os.environ["MOLGAP_PLATFORM_ID"] = "kaggle1"
    pin_one_visible_gpu()
    install_dependencies()
    root = source_root()
    commit, digest = verify_source(root)
    sys.path.insert(0, str(root / "src"))
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Candidate requires one compatible GPU, found {names}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from molgap.pcqm_k1_variants_runner import train_arm

    train_arm(MODE, Path("/kaggle/working/pcqm_k1_mose_context_gate_100k") / MODE, source_commit=commit, source_archive_sha256=digest)


if __name__ == "__main__":
    main()

