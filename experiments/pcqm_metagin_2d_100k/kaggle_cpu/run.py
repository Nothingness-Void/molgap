"""CPU-only derivation and full acceptance of MetaGIN topology."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def _one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {len(matches)}")
    return matches[0]


def main() -> None:
    started = time.monotonic()
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch-geometric==2.6.1", "ogb==1.3.6",
    ])
    archive = _one("source_payload.bin")
    commit = _one("SOURCE_COMMIT.txt").read_text().strip()
    digest = _one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    if len(commit) != 40 or hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise RuntimeError("MetaGIN source archive identity changed")
    root = Path("/kaggle/working/verified_metagin_source")
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(root, filter="data")
    for item in json.loads(_one("SOURCE_FILES.json").read_text())["files"]:
        path = root / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"MetaGIN source inventory changed: {item['path']}")
    sys.path.insert(0, str(root / "src"))
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CPU topology task must not consume a GPU")
    from molgap.pcqm_k1_variants_runner import find_fixed_cache
    from molgap.pcqm_metagin_sidecar import accept_sidecar, build_sidecar
    from molgap.training_reproducibility import atomic_json

    fixed_root, _ = find_fixed_cache()
    output = Path("/kaggle/working/molgap-metagin-2d-hop-cache-v1")
    built = build_sidecar(output, source_commit=commit)
    accepted = accept_sidecar(
        output, fixed_root=fixed_root, expected_source_commit=commit,
    )
    atomic_json(output / "native_cost.json", {
        "format": "molgap-metagin-cpu-sidecar-cost-v1",
        "wall_seconds": time.monotonic() - started,
        "gpu_used": False, "rows_verified": accepted["rows_recomputed"],
    })
    print(json.dumps({
        "accepted": accepted["accepted"], "role_counts": built["role_counts"],
        "aggregate_sha256": accepted["aggregate_sha256"],
    }), flush=True)


if __name__ == "__main__":
    main()
