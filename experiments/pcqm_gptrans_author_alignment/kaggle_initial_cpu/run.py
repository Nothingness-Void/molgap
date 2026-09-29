"""Kaggle2 CPU wrapper for real-input GPTrans initialization-scale diagnosis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def mounted(dataset: str, name: str) -> Path:
    root = Path("/kaggle/input") / dataset
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {dataset}/{name}; found {len(matches)}")
    return matches[0]


def main() -> None:
    started = time.monotonic()
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch-geometric==2.6.1", "ogb==1.3.6",
    ])
    source_dataset = "molgap-gptrans-initial-scale-source-v1"
    archive = mounted(source_dataset, "source_payload.bin")
    commit = mounted(source_dataset, "SOURCE_COMMIT.txt").read_text(encoding="ascii").strip()
    archive_sha = mounted(source_dataset, "SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="ascii").strip()
    if len(commit) != 40 or hashlib.sha256(archive.read_bytes()).hexdigest() != archive_sha:
        raise RuntimeError("Source archive identity changed")
    inventory = json.loads(mounted(source_dataset, "SOURCE_FILES.json").read_text(encoding="utf-8"))
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
    manifest = mounted("pcqm4mv2-ogb-fixed-100k-v1", "manifest.json")
    initial = mounted("molgap-gptrans-t-v4-source", "initial_state.pt")
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
