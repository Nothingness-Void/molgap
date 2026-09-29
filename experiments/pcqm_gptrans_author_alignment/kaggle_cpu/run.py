"""CPU-only accepted-graph preflight; scientific logic is in molgap source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name}; found {len(matches)}")
    return matches[0]


def main() -> None:
    started = time.monotonic()
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch-geometric==2.6.1", "ogb==1.3.6",
    ])
    archive = one("source_payload.bin")
    commit = one("SOURCE_COMMIT.txt").read_text(encoding="ascii").strip()
    archive_sha = one("SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="ascii").strip()
    if len(commit) != 40 or hashlib.sha256(archive.read_bytes()).hexdigest() != archive_sha:
        raise RuntimeError("Source archive identity changed")
    inventory = json.loads(one("SOURCE_FILES.json").read_text(encoding="utf-8"))
    expected = "src/molgap/gptrans_path_real_preflight.py"
    paths = [entry["path"] for entry in inventory["files"]]
    if expected not in paths or "src/molgap/__init__.py" not in paths or len(paths) != len(set(paths)):
        raise RuntimeError("Required source files are absent or duplicated")
    root = Path("/kaggle/working/verified_gptrans_path_source")
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if {item.name for item in members} != set(paths) or not all(item.isfile() for item in members):
            raise RuntimeError("Unexpected source archive members")
        bundle.extractall(root, filter="data")
    for entry in inventory["files"]:
        source = root / entry["path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Mounted source changed: {entry['path']}")
    sys.path.insert(0, str(root / "src"))
    from molgap import gptrans_path_real_preflight as module
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("Path construction preflight must run without a GPU")
    manifest = one("manifest.json")
    output = Path("/kaggle/working/gptrans_real_path_preflight_v1")
    result = module.run(manifest.parent, output)
    module.atomic_json(output / "native_cost.json", {
        "format": "molgap-gptrans-real-path-preflight-cost-v1",
        "wall_seconds": time.monotonic() - started,
        "gpu_used": False,
        "sampled_train_rows": sum(item["sampled_rows"] for item in result["shards"]),
        "source_commit": commit,
        "source_archive_sha256": archive_sha,
    })
    print(json.dumps({"status": result["status"], "source_commit": commit,
                      "sampled_train_rows": 512}), flush=True)


if __name__ == "__main__":
    main()
