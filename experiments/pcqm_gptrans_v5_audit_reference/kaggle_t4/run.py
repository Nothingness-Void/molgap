"""Kaggle2 T4 adapter for one 10-epoch GPTrans audit-reference segment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


MANIFEST_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
INITIAL_SHA256 = "9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c"
SOURCE_SHA256 = "af94a63c3c4626ceec8af4106aad0ea97d398e40aefb04c51b15356b994137a3"
SOURCE_COMMIT = "21f70ab93c3b0ddc4745316daf8459df40efdb48"
OUTPUT = Path("/kaggle/working/gptrans_v5_audit_reference")
TRAIN_FILES = (
    "last_checkpoint.pt", "trace.json", "canonical_trace.json",
    "best_model.pt", "development_predictions.pt",
)
EXPECTED_PREVIOUS_EPOCHS = None


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def mounted(name: str, expected: str) -> Path:
    matches = [p for p in Path("/kaggle/input").rglob(name) if digest(p) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one verified {name}, found {len(matches)}")
    return matches[0]


def source_archive() -> tuple[Path, str, str]:
    matches = []
    for archive in Path("/kaggle/input").rglob("source_payload.bin"):
        sidecar = archive.parent / "SOURCE_ARCHIVE_SHA256.txt"
        commit = archive.parent / "SOURCE_COMMIT.txt"
        inventory = archive.parent / "SOURCE_FILES.json"
        if not all(p.is_file() for p in (sidecar, commit, inventory)):
            continue
        expected = sidecar.read_text(encoding="ascii").strip()
        if digest(archive) != expected or expected != SOURCE_SHA256:
            continue
        if commit.read_text(encoding="ascii").strip() != SOURCE_COMMIT:
            continue
        files = {item["path"] for item in json.loads(inventory.read_text(encoding="utf-8"))["files"]}
        if "src/molgap/pcqm_gptrans_v4.py" in files and "src/molgap/research_memory/trace.py" in files:
            with tarfile.open(archive, "r:gz") as bundle:
                member = bundle.extractfile("src/molgap/pcqm_gptrans_v4.py")
                if member is not None and b"def _v5_audit_recorder" in member.read():
                    matches.append((archive, expected, commit.read_text(encoding="ascii").strip()))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one verified audit source archive, found {len(matches)}")
    return matches[0]


def prior_segment() -> Path | None:
    candidates = list(Path("/kaggle/input").rglob("partial_manifest.json"))
    if not candidates:
        if EXPECTED_PREVIOUS_EPOCHS is not None:
            raise RuntimeError("Required previous segment is not mounted")
        return None
    if EXPECTED_PREVIOUS_EPOCHS is None:
        raise RuntimeError("Unexpected previous segment mounted for first job")
    if len(candidates) != 1:
        raise RuntimeError("Ambiguous previous audit segment")
    prior = candidates[0].parent
    record = json.loads(candidates[0].read_text(encoding="utf-8"))
    if record.get("v5_audit") is not True or record.get("complete") is not False:
        raise RuntimeError("Previous segment is not a partial V5 audit")
    if record["completed_epochs"] != EXPECTED_PREVIOUS_EPOCHS:
        raise RuntimeError("Unexpected previous epoch boundary")
    if record.get("source_archive_sha256") != SOURCE_SHA256 or record.get("manifest_sha256") != MANIFEST_SHA256:
        raise RuntimeError("Previous segment scientific identity changed")
    expected = record.get("resume_file_sha256", {})
    if set(expected) != set(TRAIN_FILES):
        raise RuntimeError("Previous segment lacks resume-file checksums")
    for name in TRAIN_FILES:
        if digest(prior / name) != expected[name]:
            raise RuntimeError(f"Previous segment changed: {name}")
    return prior


def main() -> None:
    started = time.monotonic()
    allocation = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout.strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "torch-geometric==2.6.1", "ogb==1.3.6"])
    import torch

    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Expected one visible T4, found {torch.cuda.device_count()} devices")
    archive, archive_sha, commit = source_archive()
    manifest = mounted("manifest.json", MANIFEST_SHA256)
    initial = mounted("initial_state.pt", INITIAL_SHA256)
    source_root = OUTPUT / "verified_source"
    source_root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if not all(member.isfile() and member.name.startswith("src/molgap/") and ".." not in Path(member.name).parts for member in members):
            raise RuntimeError("Untrusted source archive path")
        bundle.extractall(source_root, filter="data")
    sys.path.insert(0, str(source_root / "src"))
    from molgap.pcqm_gptrans_v4 import run_preflight, run_training
    from molgap.training_reproducibility import atomic_json

    prior = prior_segment()
    training = OUTPUT / "training"
    training.mkdir(parents=True, exist_ok=True)
    if prior is not None:
        for name in TRAIN_FILES:
            shutil.copyfile(prior / name, training / name)
    common = {
        "dataset_root": manifest.parent,
        "manifest_path": manifest,
        "source_archive": archive,
        "source_archive_sha256": archive_sha,
        "source_commit": commit,
        "platform_id": "kaggle2-t4-v5-audit",
        "initial_state_path": initial,
    }
    preflight = OUTPUT / "preflight"
    run_preflight(output=preflight, **common)
    result = run_training(
        preflight_path=preflight / "preflight.json", output=training,
        v5_audit=True, max_epochs_this_job=10, **common,
    )
    if not result.get("complete"):
        result["resume_file_sha256"] = {name: digest(training / name) for name in TRAIN_FILES}
        atomic_json(training / "partial_manifest.json", result)
    atomic_json(OUTPUT / "native_cost.json", {
        "format": "molgap-gptrans-v5-audit-native-cost-v1",
        "allocated_gpu_inventory": allocation.splitlines(),
        "visible_gpu": torch.cuda.get_device_name(0),
        "allocated_gpu_count": len(allocation.splitlines()),
        "used_gpu_count": 1,
        "wall_seconds": time.monotonic() - started,
        "completed_epochs": result.get("completed_epochs", 60),
        "source_commit": commit,
        "source_archive_sha256": archive_sha,
    })
    print(json.dumps({"complete": result["complete"], "epochs": result.get("completed_epochs", 60), "allocated_gpus": len(allocation.splitlines())}), flush=True)


if __name__ == "__main__":
    main()
