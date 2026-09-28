"""One-GPU frozen MetaGIN2D V5 seed-42 entry; no successor submission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
import traceback


RUN_ID = "kaseichou/molgap-metagin-2d-s42:v2"
TRAJECTORY_ID = "TC-metagin-2d-3hop-100k-s42-v2"
OUTPUT = Path("/kaggle/working/molgap-metagin-2d-s42-v2")


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
    from molgap.k1_edge_kaggle_runtime import _pin_runtime

    _pin_runtime(required_devices=1, required_name=None)
    import torch
    allocated_devices = torch.cuda.device_count()
    if allocated_devices < 1:
        raise RuntimeError("MetaGIN screen requires one visible accelerator")
    from molgap.pcqm_metagin_screen import MODEL_ID, train_screen
    from molgap.pcqm_metagin_sidecar import FORMAT
    from molgap.training_reproducibility import atomic_json, sha256_file

    sidecars = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("format") == FORMAT:
            sidecars.append(path.parent)
    if len(sidecars) != 1:
        raise RuntimeError("Expected exactly one accepted MetaGIN hop sidecar")
    finished = False
    try:
        train_screen(
            OUTPUT, source_commit=commit, source_archive_sha256=digest,
            sidecar_root=sidecars[0], transform_path=_one("target_transform.json"),
            run_id=RUN_ID, trajectory_id=TRAJECTORY_ID,
        )
        finished = True
    except BaseException:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUTPUT / "failure.json", {
            "format": "molgap-metagin-2d-failure-v1", "model_id": MODEL_ID,
            "run_id": RUN_ID, "source_commit": commit,
            "traceback": traceback.format_exc(), "complete": False,
        })
        raise
    finally:
        atomic_json(OUTPUT / "native_cost.json", {
            "format": "molgap-metagin-2d-native-cost-v1",
            "run_id": RUN_ID, "model_id": MODEL_ID,
            "hardware": torch.cuda.get_device_name(0),
            "allocated_device_count": allocated_devices,
            "wall_seconds": time.monotonic() - started,
            "allocated_device_seconds": (time.monotonic() - started) * allocated_devices,
            "measurement": "monotonic-full-kernel-lifetime-times-visible-allocated-devices",
            "training_completed": finished,
        })
        if finished:
            manifest_path = OUTPUT / "completion_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifact_sha256"]["native_cost.json"] = sha256_file(
                OUTPUT / "native_cost.json"
            )
            atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
