"""Kaggle2 CPU wrapper for real-input GPTrans initialization-scale diagnosis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


SOURCE_SHA256 = "bcbfe2122d8c388f39c3728def4aa6cd1919e530c97578e10b41d5fe4e244411"
SOURCE_COMMIT = "de62aadeae02bc0fea9f8e4bc215d8fdf2f75c20"
INITIAL_SHA256 = "9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c"
MANIFEST_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mounted(name: str, expected_sha256: str) -> Path:
    candidates = list(Path("/kaggle/input").rglob(name))
    matches = [path for path in candidates if sha256_file(path) == expected_sha256]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one SHA-matched {name}; found {len(matches)} among {len(candidates)} candidates")
    return matches[0]


def main() -> None:
    started = time.monotonic()
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch-geometric==2.6.1", "ogb==1.3.6",
    ])
    archive = mounted("source_payload.bin", SOURCE_SHA256)
    commit = (archive.parent / "SOURCE_COMMIT.txt").read_text(encoding="ascii").strip()
    archive_sha = (archive.parent / "SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="ascii").strip()
    if commit != SOURCE_COMMIT or archive_sha != SOURCE_SHA256:
        raise RuntimeError("Source archive identity changed")
    inventory = json.loads((archive.parent / "SOURCE_FILES.json").read_text(encoding="utf-8"))
    paths = [entry["path"] for entry in inventory["files"]]
    if ("src/molgap/gptrans_initial_scale_preflight.py" not in paths
            or "src/molgap/__init__.py" not in paths or len(paths) != len(set(paths))):
        raise RuntimeError("Required source files are absent or duplicated")
    root = Path("/kaggle/working/verified_gptrans_initial_source")
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if {item.name for item in members} != set(paths) or not all(item.isfile() for item in members):
            raise RuntimeError("Unexpected source archive members")
        bundle.extractall(root, filter="data")
    for entry in inventory["files"]:
        if hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Mounted source changed: {entry['path']}")
    sys.path.insert(0, str(root / "src"))
    from molgap import gptrans_initial_scale_preflight as module
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("Initialization input diagnostic must run without a GPU")
    output = Path("/kaggle/working/gptrans_initial_scale_preflight_v1")
    manifest = mounted("manifest.json", MANIFEST_SHA256)
    initial = mounted("initial_state.pt", INITIAL_SHA256)
    result = module.run(manifest.parent, initial, output)
    module.atomic_json(output / "native_cost.json", {
        "format": "molgap-gptrans-initial-scale-preflight-cost-v1",
        "wall_seconds": time.monotonic() - started,
        "gpu_used": False, "sampled_train_rows": 512,
        "source_commit": commit, "source_archive_sha256": archive_sha,
    })
    print(json.dumps({"status": result["status"], "source_commit": commit,
                      "degree_fraction_without_cross": result["input_energy"]["degree_fraction_without_cross"]}),
          flush=True)


if __name__ == "__main__":
    main()
