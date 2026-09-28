"""CPU-only, independent fixed-cache motif sidecar construction/acceptance."""
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
    commit = _one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    digest = _one("SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
    if len(commit) != 40 or hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Motif source archive identity changed")
    root = Path("/kaggle/working/verified_motif_source")
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(root, filter="data")
    for item in json.loads(_one("SOURCE_FILES.json").read_text(encoding="utf-8"))["files"]:
        path = root / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Motif source changed: {item['path']}")
    sys.path.insert(0, str(root / "src"))
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("Motif construction must use CPU resources only")
    from molgap.pcqm_k1_variants_runner import find_fixed_cache
    from molgap.pcqm_motif_sidecar import accept_sidecar, build_sidecar
    from molgap.training_reproducibility import atomic_json

    fixed_root, _ = find_fixed_cache()
    if not fixed_root.is_dir():
        raise RuntimeError("Accepted fixed cache is not mounted")
    output = Path("/kaggle/working/molgap-motif-partition-cache-v1")
    built = build_sidecar(output, source_commit=commit)
    accepted = accept_sidecar(output, expected_source_commit=commit)
    train = [item for item in built["shards"] if item["role"] == "train"]
    rows = sum(item["statistics"]["rows"] for item in train)
    multi = sum(item["statistics"]["multi_motif_rows"] for item in train)
    three = sum(item["statistics"]["three_plus_motif_rows"] for item in train)
    elapsed = time.monotonic() - started
    feasible = (
        accepted["accepted"] is True and rows == 100_000
        and multi >= 25_000 and three >= 10_000 and elapsed <= 7_200
    )
    atomic_json(output / "feasibility.json", {
        "format": "molgap-motif-hierarchy-feasibility-v1",
        "accepted_partition": accepted["accepted"],
        "gpu_training_released": False,
        "feasible_for_later_gpu_preflight": feasible,
        "train_rows": rows, "train_multi_motif_rows": multi,
        "train_three_plus_motif_rows": three,
        "train_multi_motif_fraction": multi / rows,
        "train_three_plus_motif_fraction": three / rows,
        "cpu_wall_seconds": elapsed,
        "gap_labels_read": False, "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False, "test_challenge_role_read": False,
    })
    atomic_json(output / "native_cost.json", {
        "format": "molgap-motif-hierarchy-cpu-cost-v1",
        "wall_seconds": elapsed, "gpu_used": False,
        "rows_recomputed": accepted["rows_recomputed"],
    })
    print(json.dumps({
        "accepted": accepted["accepted"], "feasible_for_later_gpu_preflight": feasible,
        "multi_motif_fraction": multi / rows,
        "three_plus_motif_fraction": three / rows,
        "aggregate_sha256": accepted["aggregate_sha256"],
    }), flush=True)


if __name__ == "__main__":
    main()
