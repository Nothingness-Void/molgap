"""Kaggle2 P100 entry point for the pure-2D GPTrans-T V4 reference."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def find_fixed_cache() -> tuple[Path, Path]:
    matches = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("format") == "molgap-pcqm4mv2-kaggle-fixed-subset-v1" and payload.get("identity", {}).get("name") == "ogb-train-100k":
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one fixed cache manifest, found {matches}")
    return matches[0].parent, matches[0]


def ensure_p100_torch() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; raise SystemExit(0 if 'sm_60' in torch.cuda.get_arch_list() else 3)"],
        check=False,
    )
    if probe.returncode != 0:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--upgrade",
            "--force-reinstall", "torch==2.4.1", "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ])


def main() -> None:
    ensure_p100_torch()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "torch-geometric==2.6.1", "ogb==1.3.6"])
    import torch
    if torch.cuda.device_count() != 1 or "P100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"GPTrans V4 requires one P100, found {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    archive = find_one("source_payload.bin")
    modules = list(Path("/kaggle/input").rglob("src/molgap/pcqm_gptrans_v4.py"))
    if len(modules) == 1:
        python_root = modules[0].parents[1]
    else:
        source_root = Path("/kaggle/working/_gptrans_source")
        if not source_root.exists():
            shutil.unpack_archive(archive, source_root)
        python_root = source_root / "src"
    sys.path.insert(0, str(python_root))
    from molgap.pcqm_gptrans_v4 import run_preflight, run_training
    root, manifest = find_fixed_cache()
    source_commit = find_one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    archive_sha = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
    initial_state = find_one("initial_state.pt")
    output = Path("/kaggle/working/pcqm_gptrans_t_100k_v4")
    preflight_root = output / "preflight"
    training_root = output / "training"
    run_preflight(dataset_root=root, manifest_path=manifest, source_archive=archive, source_archive_sha256=archive_sha, source_commit=source_commit, output=preflight_root, platform_id="kaggle2-p100", initial_state_path=initial_state)
    run_training(dataset_root=root, manifest_path=manifest, preflight_path=preflight_root / "preflight.json", source_archive=archive, source_archive_sha256=archive_sha, source_commit=source_commit, output=training_root, platform_id="kaggle2-p100", initial_state_path=initial_state)


if __name__ == "__main__":
    main()
