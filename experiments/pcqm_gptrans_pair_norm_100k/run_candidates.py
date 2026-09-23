"""Kaggle T4x2 entry point for the frozen pair-normalization screen."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


MODES = ("pair_update_norm", "pair_post_norm")


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


def launch(mode: str, device: int, root: Path):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device)
    environment["MOLGAP_VARIANT"] = mode
    environment["MOLGAP_OUTPUT"] = str(Path("/kaggle/working") / mode)
    environment["MOLGAP_SOURCE_ROOT"] = str(root)
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
    result = run_preflight(
        dataset_root=dataset_root,
        manifest_path=dataset_manifest,
        source_archive=source_archive,
        source_archive_sha256=source_sha,
        source_commit=source_commit,
        output=output,
        platform_id="kaggle3-t4x2",
        initial_state_path=initial_state,
        variant=mode,
    )
    if result.get("accepted") is not True:
        raise RuntimeError(f"Preflight rejected {mode}: {result}")
    run_training(
        dataset_root=dataset_root,
        manifest_path=dataset_manifest,
        preflight_path=preflight_path,
        source_archive=source_archive,
        source_archive_sha256=source_sha,
        source_commit=source_commit,
        output=output,
        platform_id="kaggle3-t4x2",
        initial_state_path=initial_state,
        variant=mode,
    )


def main() -> None:
    mode = os.environ.get("MOLGAP_VARIANT")
    root = Path(os.environ["MOLGAP_SOURCE_ROOT"]) if mode else source_root()
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
    processes = [launch(mode_name, device, root) for device, mode_name in enumerate(MODES)]
    codes = [process.wait() for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"Candidate workers failed: {codes}")


if __name__ == "__main__":
    main()
