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
    "chemical-components-kaggle1-v1": (("descriptor_aux", "fingerprint_aux"), "kaggle1-t4x2"),
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
    if PROFILE == "chemical-components-kaggle1-v1":
        chemical_worker(mode, root)
        return
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


def chemical_worker(mode: str, root: Path) -> None:
    """Bind mounted artifacts to the existing chemical arm CLI, not a new trainer."""
    import json
    import shutil
    from molgap.experiment_package import SIDECARS

    release = json.loads((root / "experiments/pcqm_gptrans_chemical_aux/release_inputs.json").read_text())
    binding = release["arms"][mode]
    mounted = find_one("source_payload.bin").parent
    package = Path("/kaggle/working/_release_bindings") / mode / "package"
    package.mkdir(parents=True, exist_ok=True)
    for name in SIDECARS:
        origin = mounted / ("source_payload.bin" if name == "source.tar.gz" else name)
        shutil.copyfile(origin, package / name)
    cache = Path("/kaggle/working/_release_bindings") / mode / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "labels.npz"):
        shutil.copyfile(mounted / (mode + "_" + name), cache / name)
    output = Path(os.environ["MOLGAP_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((package / "package_manifest.json").read_text())
    manifest = find_one("manifest.json")
    args = [sys.executable, str(root / "experiments/pcqm_gptrans_chemical_aux/run_arm.py"),
            "--phase", os.environ["MOLGAP_PHASE"], "--dataset-root", str(manifest.parent),
            "--manifest-path", str(manifest), "--source-archive", str(package / "source.tar.gz"),
            "--source-archive-sha256", metadata["archive_sha256"],
            "--source-commit", metadata["source_commit"], "--output", str(output),
            "--platform-id", PLATFORM_ID, "--initial-state-path", str(mounted / "initial_state.pt"),
            "--objective-config", str(root / binding["objective_config"]),
            "--cache-root", str(cache), "--cache-manifest-sha256", binding["cache_manifest_sha256"],
            "--cache-role-sha256", binding["cache_role_sha256"],
            "--spec", str(package / "experiment_spec.json"), "--package", str(package),
            "--contract", str(root / binding["contract"]),
            "--expected-package-identity", metadata["package_identity"],
            "--arm-id", mode, "--trajectory-id", binding["trajectory_id"]]
    if os.environ["MOLGAP_PHASE"] == "train":
        args += ["--preflight-path", str(output / "preflight.json")]
    subprocess.check_call(args)


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
