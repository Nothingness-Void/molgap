"""Forty-pass execution assay over the native trainer, not a copied epoch loop."""
import json
from pathlib import Path
import time

from .experiment_package import verify_experiment_source_package
from .experiment_spec import ExperimentSpec
from .gptrans_screen_adapter import gptrans_screen_arguments
from .training_reproducibility import atomic_json, sha256_file

ARMS = ("execution_original", "execution_optimized")
MODEL = "degree_bond_local_ema999"
EPOCHS = 40


def child(stage, arm, context, root):
    """Invoked only by the isolated, released Kaggle worker."""
    if arm not in ARMS or stage not in {"preflight", "training"}:
        raise ValueError("Unsupported execution study stage/arm")
    from . import pcqm_gptrans_v4 as native
    from .gptrans_execution_backend import backend
    import torch
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("One isolated T4 is required per execution arm")
    config = json.loads(Path(context["screen_config"]).read_text())
    study = config["execution_study"]
    if (study["epochs"] != EPOCHS or study["schedule_horizon_epochs"] != native.EPOCHS
            or study["model_variant"] != MODEL or study["physical_batch"] != 128
            or study["precision"] != "fp32" or study["tf32_enabled"] is not False):
        raise ValueError("Frozen execution study scope changed")
    package_dir = Path(context["package_dir"])
    package = verify_experiment_source_package(package_dir)
    spec = ExperimentSpec.from_json((package_dir / "experiment_spec.json").read_text())
    if config["spec_identity"] != spec.identity or package["archive_sha256"] != context["archive_sha256"]:
        raise ValueError("Source/Spec binding changed")
    plan = next(row for row in spec.to_dict()["prospective"]["arms"] if row["arm_id"] == arm)
    # Source text is LF-normalized by the packager. The transport artifact
    # preserves the declaration's exact bytes, including its Spec-bound SHA.
    staged = Path(context["upload_root"]) / (arm + "_plan.json")
    if sha256_file(staged) != plan["plan_spec_sha256"]:
        raise ValueError("Prospective declaration bytes changed")
    from .research_memory.schemas import validate_trajectory
    from .research_memory.plan import _prepare_trajectory
    trajectory = _prepare_trajectory(json.loads(staged.read_text()))
    validate_trajectory(trajectory)
    if trajectory["trajectory_id"] != plan["trajectory_id"] or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Missing bound active prospective arm")
    initial = Path(context["upload_root"]) / config["arms"][arm]["initial_file"]
    if sha256_file(initial) != config["arms"][arm]["initial_file_sha256"]:
        raise ValueError("Shared initialization changed")
    binding = {"arm": arm, "package_identity": package["package_identity"],
               "spec_identity": spec.identity, "study": study, "trajectory_id": plan["trajectory_id"],
               "source_archive_sha256": package["archive_sha256"]}
    path = root / "arm_binding.json"
    root.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError("Existing output binding differs")
    atomic_json(path, binding)
    arguments = gptrans_screen_arguments(spec, arm, package_dir=package_dir,
        dataset_root=Path(context["dataset_root"]), manifest_path=Path(context["manifest_path"]),
        initial_state_path=initial, target_transform_path=Path(context["target_transform_path"]),
        platform_id="kaggle2-t4-gptrans-execution40")
    optimized = arm == ARMS[1]
    began = time.perf_counter()
    with backend(native, optimized, root / "phase_timing" if stage == "training" else None) as timing:
        if stage == "preflight":
            result = native.run_preflight(output=root / "preflight", **arguments["preflight"])
            if result["optimizer_step_calibration"]["estimated_training_hours"] * EPOCHS / native.EPOCHS > 5:
                raise ValueError("Forty-pass train-body estimate exceeds frozen five-hour cap")
        else:
            result = native.run_training(output=root / "training", preflight_path=root / "preflight/preflight.json",
                max_epochs_this_job=EPOCHS, **arguments["training"])
            if result.get("complete") is not False or result.get("completed_epochs") != EPOCHS:
                raise ValueError("Native sixty-pass prefix must end exactly at forty passes")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            atomic_json(root / "execution_completion.json", {
                "format": "molgap-gptrans-execution40-completion-v1", "complete": True,
                "native_sixty_pass_complete": False, "native_partial_manifest": result,
                "arm": arm, "binding": binding, "epochs": EPOCHS,
                "optimizer_steps": EPOCHS * native.BATCHES_PER_EPOCH,
                "sample_presentations": EPOCHS * native.BATCHES_PER_EPOCH * native.PHYSICAL_BATCH,
                "stage_wall_seconds_including_loading_and_publication": elapsed,
                "phase_timing": dict(timing), "scientific_fields": native._scientific_fields(MODEL),
                "parameters": config["arms"][arm]["expected_parameters"],
                "initial_state_artifact_sha256": sha256_file(initial),
                "training_executed": True, "official_validation_role_read": False,
                "test_dev_role_read": False, "test_challenge_role_read": False,
                "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*")
                          if p.is_file() and p.name != "execution_completion.json"}})
