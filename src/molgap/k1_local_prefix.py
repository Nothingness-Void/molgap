"""Local one-device prefix orchestration over the registered K1 family trainer."""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from .experiment_execution import execute_training_phase
from .experiment_family_workflow import TargetIdentityBinding
from .experiment_package import verify_experiment_source_package
from .experiment_spec import ExperimentSpec
from .training_reproducibility import atomic_json, sha256_file


def run(config_path: Path):
    import torch
    from . import k1_screen_training as owner
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = Path(config["source_root"])
    if Path(owner.__file__).resolve() != (source / "src/molgap/k1_screen_training.py").resolve():
        raise ValueError("Local execution must import the frozen source, not a mutable checkout")
    package = Path(config["package"])
    manifest = verify_experiment_source_package(package)
    if manifest["package_identity"] != config["package_identity"]:
        raise ValueError("Local package binding changed")
    inventory = json.loads((package / "SOURCE_FILES.json").read_text())
    for item in inventory["files"]:
        if sha256_file(source / item["path"]) != item["sha256"]:
            raise ValueError("Frozen source changed: " + item["path"])
    spec = ExperimentSpec(json.loads((package / "experiment_spec.json").read_text()))
    if spec.to_dict()["platform"]["name"] != "local" or config["allocation_seconds"] != 3600:
        raise ValueError("Expected the frozen local prefix release")
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "run_state.json").exists():
        raise ValueError("Existing local attempt requires reconciliation, not an automatic restart")
    atomic_json(root / "executed_config.json", config)
    state = {"status": "CPU_ACCEPTANCE", "config_sha256": sha256_file(config_path), "arms": {}}
    state["process_id"] = os.getpid()
    atomic_json(root / "run_state.json", state)
    cpu_start = time.perf_counter()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU input acceptance must precede CUDA allocation")
    cache, cache_manifest = owner.find_fixed_cache(Path(config["input_root"]))
    roles = owner.load_roles(cache, cache_manifest)
    mean, std = owner._target_stats(roles["train"])
    del roles
    gc.collect()
    atomic_json(root / "cpu_acceptance.json", {"train_rows": owner.TRAIN_ROWS,
        "development_rows": owner.DEVELOPMENT_ROWS, "manifest_sha256": owner.FIXED_MANIFEST_SHA256,
        "mean": mean, "std": std, "wall_seconds": time.perf_counter() - cpu_start,
        "cuda_initialized": torch.cuda.is_initialized(), "protected_roles_read": False})
    started = time.time()

    def gpu_snapshot(name):
        observation = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        atomic_json(root / name, {"observed_unix": time.time(), "returncode": observation.returncode,
            "stdout": observation.stdout, "stderr": observation.stderr, "exclusive_device": False})

    gpu_snapshot("gpu_before.json")

    def timeout():
        atomic_json(root / "timeout.json", {"status": "STOP_FOR_COST", "elapsed_seconds": time.time() - started,
            "reason": "Local3600s ceiling; only acknowledged epoch checkpoints are authoritative"})
        os._exit(124)

    watchdog = threading.Timer(config["allocation_seconds"], timeout)
    watchdog.daemon = True
    watchdog.start()
    try:
        for phase in ("preflight", "train"):
            for job in config["jobs"]:
                arm = job["arm_id"]
                state["status"] = phase.upper() + ":" + arm
                atomic_json(root / "run_state.json", state)
                binding = TargetIdentityBinding.from_acceptance_plan(spec, Path(config["staged_root"]),
                    config["acceptance_plan"], plan_sha256=config["acceptance_plan_sha256"])
                result = execute_training_phase(spec=spec, job=job, phase=phase, source_root=source,
                    package_dir=package, expected_package_identity=config["package_identity"],
                    input_root=Path(config["input_root"]), staged_root=Path(config["staged_root"]),
                    output=root / arm, account="desktop", run_reference="desktop/" + spec.to_dict()["logical_run_id"],
                    prospective_sha256=config["prospective_sha256"][arm], target_identity=binding)
                atomic_json(root / arm / (phase + "_report.json"), result)
                if result["status"] != ("accepted" if phase == "preflight" else "MECHANICALLY_VERIFIED"):
                    raise RuntimeError("Local arm did not pass " + phase + ": " + arm)
                state["arms"].setdefault(arm, {})[phase] = result["status"]
                atomic_json(root / "run_state.json", state)
                gc.collect()
                torch.cuda.empty_cache()
        state.update(status="COMPLETE_10EPOCH_PREFIX", elapsed_seconds=time.time() - started,
            full40epoch_conclusion=False, automatic_scaleup=False)
        atomic_json(root / "run_state.json", state)
    except BaseException as error:
        state.update(status="INFRASTRUCTURE_FAILURE", error_type=type(error).__name__, error=str(error))
        atomic_json(root / "run_state.json", state)
        raise
    finally:
        watchdog.cancel()
        atomic_json(root / "allocation_terminal.json", {"hardware": "NVIDIA GeForce RTX 5060",
            "started_unix": started, "finished_unix": time.time(), "wall_seconds": time.time() - started,
            "status": state["status"], "exclusive_device": False,
            "scope": "local CUDA qualification/training/evaluation/IO; CPU acceptance excluded",
            "queue_seconds": {"status": "not_applicable", "value": None},
            "cpu_allocation_seconds": {"status": "missing", "value": None}})
        gpu_snapshot("gpu_after.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
