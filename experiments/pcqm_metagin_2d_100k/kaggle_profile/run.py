"""Thin Kaggle entry for a short train-role-only steady-state profile."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
import traceback


def _one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {len(matches)}")
    return matches[0]


def main() -> None:
    started = time.monotonic()
    archive = _one("source_payload.bin")
    commit = _one("SOURCE_COMMIT.txt").read_text().strip()
    digest = _one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    if len(commit) != 40 or hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise RuntimeError("MetaGIN profiling source identity changed")
    root = Path("/kaggle/working/verified_metagin_profile_source")
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(root, filter="data")
    for item in json.loads(_one("SOURCE_FILES.json").read_text())["files"]:
        path = root / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"MetaGIN source inventory changed: {item['path']}")
    sys.path.insert(0, str(root / "src"))
    from molgap.k1_edge_kaggle_runtime import _pin_runtime

    _pin_runtime(required_devices=1, required_name=None)
    from molgap.pcqm_metagin_profile import profile_runtime
    from molgap.pcqm_metagin_sidecar import FORMAT
    from molgap.training_reproducibility import atomic_json

    sidecars = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("format") == FORMAT:
            sidecars.append(path.parent)
    if len(sidecars) != 1:
        raise RuntimeError("Expected one accepted MetaGIN topology sidecar")
    output = Path("/kaggle/working/metagin-runtime-profile")
    try:
        result = profile_runtime(
            output, source_commit=commit, sidecar_root=sidecars[0],
            transform_path=_one("target_transform.json"),
        )
        print(json.dumps({key: result[key] for key in (
            "hardware", "allocated_device_count", "steady_train_step_mean_seconds",
            "projected_with_20_percent_reserve_seconds", "profile_wall_seconds",
        )}), flush=True)
    except BaseException:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "failure.json", {
            "format": "molgap-metagin-runtime-profile-failure-v1",
            "source_commit": commit, "traceback": traceback.format_exc(),
            "complete": False,
        })
        raise
    finally:
        atomic_json(output / "native_cost.json", {
            "format": "molgap-metagin-runtime-profile-cost-v1",
            "source_commit": commit,
            "wall_seconds": time.monotonic() - started,
            "training_screen_executed": False,
        })


if __name__ == "__main__":
    main()
