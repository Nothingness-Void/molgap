"""Kaggle2 P100 entry point for the pure-2D GPTrans-T V4 reference."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


STAGE_ENV = "MOLGAP_GPTRANS_STAGE"
CONTEXT_ENV_PREFIX = "MOLGAP_GPTRANS_"


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


def validate_accelerator() -> None:
    import torch
    if torch.cuda.device_count() != 1 or "P100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"GPTrans V4 requires one P100, found {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")


def discover_context() -> dict[str, Path | str]:
    archive = find_one("source_payload.bin")
    modules = list(Path("/kaggle/input").rglob("src/molgap/pcqm_gptrans_v4.py"))
    if len(modules) == 1:
        python_root = modules[0].parents[1]
    else:
        source_root = Path("/kaggle/working/_gptrans_source")
        if not source_root.exists():
            shutil.unpack_archive(archive, source_root)
        python_root = source_root / "src"
    root, manifest = find_fixed_cache()
    source_commit = find_one("SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    archive_sha = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text(encoding="utf-8").strip()
    initial_state = find_one("initial_state.pt")
    output = Path("/kaggle/working/pcqm_gptrans_t_100k_v4")
    preflight_root = output / "preflight"
    training_root = output / "training"
    return {
        "PYTHON_ROOT": python_root,
        "DATASET_ROOT": root,
        "MANIFEST": manifest,
        "SOURCE_ARCHIVE": archive,
        "SOURCE_ARCHIVE_SHA256": archive_sha,
        "SOURCE_COMMIT": source_commit,
        "INITIAL_STATE": initial_state,
        "PREFLIGHT_ROOT": preflight_root,
        "TRAINING_ROOT": training_root,
    }


def run_stage(stage: str, context: dict[str, Path | str]) -> None:
    environment = os.environ.copy()
    environment[STAGE_ENV] = stage
    for key, value in context.items():
        environment[f"{CONTEXT_ENV_PREFIX}{key}"] = str(value)
    subprocess.check_call([sys.executable, str(Path(__file__).resolve())], env=environment)


def child_main(stage: str) -> None:
    python_root = Path(os.environ[f"{CONTEXT_ENV_PREFIX}PYTHON_ROOT"])
    sys.path.insert(0, str(python_root))
    from molgap.pcqm_gptrans_v4 import run_preflight, run_training

    def context_path(name: str) -> Path:
        return Path(os.environ[f"{CONTEXT_ENV_PREFIX}{name}"])

    common = {
        "dataset_root": context_path("DATASET_ROOT"),
        "manifest_path": context_path("MANIFEST"),
        "source_archive": context_path("SOURCE_ARCHIVE"),
        "source_archive_sha256": os.environ[f"{CONTEXT_ENV_PREFIX}SOURCE_ARCHIVE_SHA256"],
        "source_commit": os.environ[f"{CONTEXT_ENV_PREFIX}SOURCE_COMMIT"],
        "platform_id": "kaggle2-p100",
        "initial_state_path": context_path("INITIAL_STATE"),
    }
    if stage == "preflight":
        run_preflight(output=context_path("PREFLIGHT_ROOT"), **common)
    elif stage == "training":
        run_training(
            preflight_path=context_path("PREFLIGHT_ROOT") / "preflight.json",
            output=context_path("TRAINING_ROOT"),
            **common,
        )
    else:
        raise ValueError(f"Unknown GPTrans stage: {stage}")


def main() -> None:
    stage = os.environ.get(STAGE_ENV)
    if stage:
        validate_accelerator()
        child_main(stage)
        return

    ensure_p100_torch()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "torch-geometric==2.6.1", "ogb==1.3.6"])
    validate_accelerator()
    context = discover_context()
    # Fresh processes make preflight and training observe the same frozen runtime,
    # matching the two-job production contract instead of inheriting import state.
    run_stage("preflight", context)
    run_stage("training", context)


if __name__ == "__main__":
    main()
