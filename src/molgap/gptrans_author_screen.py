"""Thin G1/G2 transport adapter over the qualified native V5 GPTrans trainer.

No model or training loop is defined here. Native V5 traces retain the reference
float32 target convention; optional family-output transcoding is not assumed.
"""
from __future__ import annotations

import concurrent.futures
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .experiment_package import SIDECARS, verify_experiment_source_package
from .experiment_spec import ExperimentSpec
from .gptrans_screen_adapter import gptrans_screen_arguments
from .training_reproducibility import atomic_json, sha256_file

MODES = ("degree_scale", "path_bond_mean", "degree_path_bond_mean", "degree_scale_ema999",
         "degree_group_decay_ema999", "degree_path_endpoints_ema999", "degree_pair_depth_scale_ema999", "degree_path_bond_mean_ema999",
         "degree_node_mean_readout_ema999", "degree_bond_mean_readout_ema999", "degree_decay001_ema999")
MODES += ("degree_node352_ema999", "degree_pair64_ema999", "degree_ffn2_ema999", "degree_bond_local_ema999")
MODES += ("scale_ema",)
MODES += ("degree_pair_transition_ema999",)
MODES += ("degree_bond_local_cap_ema999",)


def validate_arm_allocation(config):
    modes = tuple(config["arms"])
    if any(mode not in MODES for mode in modes) or len(modes) not in (1, 2):
        raise ValueError("One or two supported independent arms are required")
    if len(modes) == 1 and not config.get("single_arm_reason", "").strip():
        raise ValueError("Single-arm allocation requires an explicit scientific reason")
    return modes


def mounted(input_root: Path, name: str, sha256: str) -> Path:
    matches = [p for p in input_root.rglob(name) if sha256_file(p) == sha256]
    if len(matches) != 1:
        raise ValueError(f"Expected one hash-qualified {name}, found {len(matches)}")
    return matches[0]


def restore_source_package(upload_root: Path, output: Path, expected_sha256: str) -> dict:
    """Restore exactly the six immutable package files from upload aliases."""
    output.mkdir(parents=True, exist_ok=False)
    for name in SIDECARS:
        origin = upload_root / ("source_payload.bin" if name == "source.tar.gz" else name)
        shutil.copyfile(origin, output / name)
    package = verify_experiment_source_package(output)
    if package["archive_sha256"] != expected_sha256:
        raise ValueError("Restored executable archive differs from the kernel pin")
    return package


def author_child(stage: str, variant: str, context: dict, root: Path):
    """Bind calls, and prevent an unrelated retained prefix from being resumed."""
    from .pcqm_gptrans_v4 import run_preflight, run_training
    import torch
    if variant == "scale_ema":
        if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
            raise ValueError("Scale arm requires one isolated T4")
        config = json.loads(Path(context["screen_config"]).read_bytes())
        package = verify_experiment_source_package(Path(context["package_dir"]))
        if package["spec_identity"] != config["spec_identity"]:
            raise ValueError("Scale source/Spec binding changed")
        study = config["scale_study"]
        root.mkdir(parents=True, exist_ok=True)
        binding = {"spec_identity": config["spec_identity"], "arm_id": variant,
            "trajectory_id": config["arms"][variant]["trajectory_id"],
            "source_archive_sha256": context["archive_sha256"], "configuration": study}
        retained = root / "arm_binding.json"
        if retained.exists() and json.loads(retained.read_bytes()) != binding:
            raise ValueError("Scale resume arm/Spec/source binding changed")
        if not retained.exists():
            if any(root.glob("training/*.pt")):
                raise ValueError("Unbound scale checkpoint cannot be adopted")
            atomic_json(retained, binding)
        from .gptrans_scale_profile import profile
        if stage == "preflight":
            result = profile(Path(context["upload_root"]), root / "training", {
                "initial_file_sha256": study["initial_file_sha256"],
                "training_estimate_cap_hours": study["training_estimate_cap_hours"]}, source_identity=package)
            if not result["qualification_passed"]:
                raise ValueError("Scale execution/calibration budget failed")
        elif stage == "training":
            from .gptrans_scale_ema import train
            train(Path(context["upload_root"]), root / "training", config=study, source_identity=package)
        else:
            raise ValueError(stage)
        return
    from .gptrans_author_variants import MODES as supported_modes, PATH_MODES
    if variant not in set(supported_modes) | {"degree_bond_local_cap_ema999"} or torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Each released author arm requires exactly one visible T4")
    package_dir = Path(context["package_dir"])
    spec = ExperimentSpec.from_json((package_dir / "experiment_spec.json").read_text())
    config = json.loads(Path(context["screen_config"]).read_text())
    if config["spec_identity"] != spec.identity:
        raise ValueError("Screen configuration differs from the immutable Spec")
    arm = config["arms"][variant]
    initial = Path(context["upload_root"]) / arm["initial_file"]
    if sha256_file(initial) != arm["initial_file_sha256"]:
        raise ValueError("Frozen initialization artifact changed")
    binding = {"spec_identity": spec.identity, "arm_id": variant,
               "trajectory_id": arm["trajectory_id"], "source_archive_sha256": context["archive_sha256"]}
    retained = root / "arm_binding.json"
    if retained.exists() and json.loads(retained.read_text()) != binding:
        raise ValueError("Resume arm/Spec/source binding changed")
    if not retained.exists():
        if any(root.glob("training/*.pt")):
            raise ValueError("Unbound native checkpoint cannot be adopted")
        atomic_json(retained, binding)
    arguments = gptrans_screen_arguments(spec, variant, package_dir=package_dir,
        dataset_root=Path(context["dataset_root"]), manifest_path=Path(context["manifest_path"]),
        initial_state_path=initial, target_transform_path=Path(context["target_transform_path"]),
        platform_id=config.get("platform_id", "kaggle2-t4-gptrans-author-inputs"),
        path_sidecar_root=Path(context["path_sidecar_root"]) if variant in PATH_MODES else None)
    if arguments["training"]["trajectory_id"] != arm["trajectory_id"]:
        raise ValueError("Executable arm and prospective trajectory disagree")
    if stage == "preflight":
        result = run_preflight(output=root / "preflight", **arguments["preflight"])
        # Reserve wall time for live/EMA evaluation and artifact publication.
        if result["optimizer_step_calibration"]["estimated_training_hours"] > config["training_estimate_cap_hours"]:
            raise RuntimeError("Author arm does not fit the independently frozen notebook budget")
    elif stage == "training":
        run_training(output=root / "training", preflight_path=root / "preflight/preflight.json",
                     **arguments["training"])
    else:
        raise ValueError(stage)


def run_author_screen(upload_root: Path, source_root: Path, output: Path, archive_sha256: str,
                      *, input_root: Path = Path("/kaggle/input"),
                      config_ref: str = "experiments/pcqm_gptrans_author_alignment/gpu/screen_config.json",
                      package_dir: Path | None = None):
    """Reuse isolated workers and native durability; no source graphs are built."""
    from . import gptrans_kaggle_runtime as scheduler
    started = time.monotonic()
    if package_dir is None:
        package_dir = output / "source_package"
        package = restore_source_package(upload_root, package_dir, archive_sha256)
    else:
        package = verify_experiment_source_package(package_dir)
        if package["archive_sha256"] != archive_sha256:
            raise ValueError("Pre-restored package differs from the frozen source archive")
    config_path = source_root / config_ref
    config = json.loads(config_path.read_text())
    modes = validate_arm_allocation(config)
    manifest = mounted(input_root, "manifest.json", config["dataset_manifest_sha256"])
    paths = mounted(input_root, "manifest.json", config["path_manifest_sha256"]) if config.get("path_manifest_sha256") else None
    transform = upload_root / "target_transform.json"
    if sha256_file(transform) != config["target_transform_sha256"]:
        raise ValueError("Portable train-only target transform changed")
    # Match the frozen reference's install policy; the runtime certificate binds
    # actual software rather than silently claiming it matches another image.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                           "torch-geometric==2.6.1", "ogb==1.3.6"])
    allocation = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout.splitlines()
    if len(allocation) != 2 or not all("T4" in line for line in allocation):
        raise RuntimeError("The frozen dual-arm release requires a T4x2 allocation")
    scheduler.ROOT = output
    # Child processes import the scheduler anew, so pin its output/mode settings.
    import os
    os.environ.update(MOLGAP_SCREEN_ROOT=str(output), MOLGAP_SCREEN_MODES=json.dumps(list(modes)))
    context = {"author_screen": True, "python_root": str(source_root / "src"),
               "package_dir": str(package_dir), "upload_root": str(upload_root),
               "screen_config": str(config_path), "archive_sha256": archive_sha256,
               "dataset_root": str(manifest.parent), "manifest_path": str(manifest),
               "path_sidecar_root": str(paths.parent) if paths else None, "target_transform_path": str(transform)}
    atomic_json(output / "startup.json", {"package_identity": package["package_identity"],
        "spec_identity": package["spec_identity"], "source_commit": package["source_commit"],
        "source_archive_sha256": archive_sha256, "allocated_gpu_inventory": allocation,
        "arms": list(modes), "maximum_wall_seconds": config["maximum_wall_seconds"],
        "single_arm_reason": config.get("single_arm_reason"),
        "official_validation_role_read": False, "test_dev_role_read": False, "test_challenge_role_read": False})
    outcomes = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(modes)) as executor:
            futures = {executor.submit(scheduler.worker, arm, device, context,
                       started + config["maximum_wall_seconds"]): arm for device, arm in enumerate(modes)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    outcomes.append(future.result())
                except Exception as error:
                    outcomes.append({"variant": futures[future], "complete": False, "error": str(error)})
    finally:
        elapsed = time.monotonic() - started
        atomic_json(output / "job_summary.json", {"outcomes": outcomes, "elapsed_seconds": elapsed})
        atomic_json(output / "native_cost.json", {"allocated_gpu_inventory": allocation,
            "allocated_gpu_count": 2, "used_gpu_count": len(modes), "wall_seconds": elapsed,
            "allocated_device_hours": elapsed * 2 / 3600, "source_archive_sha256": archive_sha256})
    if len(outcomes) != len(modes) or not all(row["complete"] for row in outcomes):
        raise RuntimeError("Retain independent arm outputs: " + str(outcomes))
