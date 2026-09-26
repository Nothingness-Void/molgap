"""Kaggle T4x2 entry point for two frozen GPTrans V4 screen profiles."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROFILES = {
    "pair-norm-kaggle3-v1": (("pair_update_norm", "pair_post_norm"), "kaggle3-t4x2"),
    "reference-centered-logits-kaggle1-v1": (("reference", "centered_logits"), "kaggle1-t4x2"),
    "reference-memory-value-kaggle1-v1": (("reference", "memory_value"), "kaggle1-t4x2"),
    "memory-value-message-kaggle1-v1": (("memory_value", "memory_message"), "kaggle1-t4x2"),
    "rwse16-local-edge-kaggle1-v1": (("rwse16", "rwse16_local_edge"), "kaggle1-t4x2"),
}
PROFILE = os.environ.get("MOLGAP_PAIR_PROFILE", "pair-norm-kaggle3-v1")
if PROFILE not in PROFILES:
    raise RuntimeError(f"Unauthorized pair profile: {PROFILE}")
MODES, PLATFORM_ID = PROFILES[PROFILE]


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def source_root() -> Path:
    module = find_one("src/molgap/gptrans_variants.py")
    return module.parents[2]


def install_dependencies() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "torch-geometric==2.6.1",
            "ogb==1.3.6",
        ]
    )


def launch(mode: str, device: int, root: Path, phase: str):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device)
    environment["MOLGAP_VARIANT"] = mode
    environment["MOLGAP_OUTPUT"] = str(Path("/kaggle/working") / mode)
    environment["MOLGAP_SOURCE_ROOT"] = str(root)
    environment["MOLGAP_PHASE"] = phase
    return subprocess.Popen([sys.executable, __file__], env=environment)


def worker(mode: str, root: Path) -> None:
    sys.path.insert(0, str(root / "src"))
    from molgap.pcqm_gptrans_v4 import run_preflight, run_training

    dataset_manifest = find_one("manifest.json")
    dataset_root = dataset_manifest.parent
    source_archive = find_one("source_payload.bin")
    source_sha = find_one("SOURCE_ARCHIVE_SHA256.txt").read_text().strip()
    source_commit = find_one("SOURCE_COMMIT.txt").read_text().strip()
    initial_state = find_one("initial_state.pt")
    output = Path(os.environ["MOLGAP_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    preflight_path = output / "preflight.json"
    phase = os.environ["MOLGAP_PHASE"]
    if phase == "preflight":
        result = run_preflight(
            dataset_root=dataset_root,
            manifest_path=dataset_manifest,
            source_archive=source_archive,
            source_archive_sha256=source_sha,
            source_commit=source_commit,
            output=output,
            platform_id=PLATFORM_ID,
            initial_state_path=initial_state,
            variant=mode,
        )
        if result.get("accepted") is not True:
            raise RuntimeError(f"Preflight rejected {mode}: {result}")
        return
    if phase != "train":
        raise RuntimeError(f"Unauthorized pair phase: {phase}")
    run_training(
        dataset_root=dataset_root,
        manifest_path=dataset_manifest,
        preflight_path=preflight_path,
        source_archive=source_archive,
        source_archive_sha256=source_sha,
        source_commit=source_commit,
        output=output,
        platform_id=PLATFORM_ID,
        initial_state_path=initial_state,
        variant=mode,
    )


def main() -> None:
    mode = os.environ.get("MOLGAP_VARIANT")
    root = (Path(os.environ["MOLGAP_SOURCE_ROOT"])
            if "MOLGAP_SOURCE_ROOT" in os.environ else source_root())
    if mode:
        if mode not in MODES:
            raise RuntimeError(f"Unauthorized mode: {mode}")
        worker(mode, root)
        return

    install_dependencies()
    import torch

    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if len(names) != 2 or any("T4" not in name for name in names):
        raise RuntimeError(f"Expected Kaggle T4x2, found {names}")
    print(f"GPU allocation: {names}", flush=True)
    for phase in ("preflight", "train"):
        processes = [launch(mode_name, device, root, phase)
                     for device, mode_name in enumerate(MODES)]
        codes = [process.wait() for process in processes]
        if codes != [0, 0]:
            raise RuntimeError(f"Pair {phase} workers failed: {codes}")


if __name__ == "__main__":
    main()
