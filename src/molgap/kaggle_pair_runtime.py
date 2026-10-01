"""Family-independent assigned-device execution with an all-arm phase barrier."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

from .experiment_execution import validate_execution_plan
from .experiment_family_workflow import _json
from .experiment_spec import ExperimentSpec
from .training_reproducibility import atomic_json


def run_two_phase_pair(*, source_root: Path, package_dir: Path, input_root: Path,
                       launch_path: Path, output: Path) -> dict:
    """Preflight every assigned arm before any formal training is spawned."""
    source_root, package_dir, input_root, output = map(Path, (source_root, package_dir, input_root, output))
    config = _json(launch_path)
    spec = ExperimentSpec.from_json((package_dir / "experiment_spec.json").read_text(encoding="utf-8"))
    jobs = validate_execution_plan(spec, config["jobs"])
    declaration = spec.to_dict()
    if declaration["platform"]["name"] != "kaggle":
        raise ValueError("Kaggle runtime requires a Kaggle Spec")
    names = [n.strip() for n in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines() if n.strip()]
    accelerator = declaration["platform"]["accelerator"]
    if accelerator not in {"Tesla T4", "T4", "NvidiaTeslaT4"} or len(names) != declaration["platform"]["device_count"] or any("T4" not in n for n in names):
        raise RuntimeError("Observed hardware differs from the declared T4 allocation")
    output.mkdir(parents=True, exist_ok=True)
    report = {"format": "molgap-kaggle-two-phase-pair-v2", "spec_identity": spec.identity,
              "hardware": names, "phases": [], "status": "running",
              "arms": {j["arm_id"]: {"device": j["device"], "training_started": False,
                        "terminal_status": "not_started", "worker_wall_seconds": 0.0}
                       for j in jobs}}
    started = time.perf_counter()
    atomic_json(output / "pair_state.json", report)
    for phase in ("preflight", "train"):
        workers, phase_record = [], {"phase": phase, "workers": [], "status": "running"}
        try:
            for job in jobs:
                state = report["arms"][job["arm_id"]]
                state["terminal_status"] = "running"
                if phase == "train":
                    # Persist before spawn: a sudden worker/kernel death leaves
                    # unknown progress rather than a false zero-training claim.
                    state["training_started"] = True
                atomic_json(output / "pair_state.json", report)
                arm_output = output / job["arm_id"]
                arm_output.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-m", "molgap.experiment_training_worker",
                    "--source-root", str(source_root), "--package-dir", str(package_dir),
                    "--input-root", str(input_root), "--output", str(arm_output),
                    "--launch", str(launch_path), "--arm", job["arm_id"], "--phase", phase]
                env = dict(os.environ)
                env.update(CUDA_VISIBLE_DEVICES=str(job["device"]), PYTHONHASHSEED="42",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONPATH=str(source_root / "src"))
                log = (arm_output / (phase + ".log")).open("a", encoding="utf-8")
                try:
                    process = subprocess.Popen(command, cwd=source_root, env=env,
                                               stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                workers.append((process, log, {**job, "started": time.perf_counter()}))
            while any(p.poll() is None for p, _, _ in workers):
                if any(p.poll() not in (None, 0) for p, _, _ in workers):
                    raise RuntimeError(f"Arm failed during {phase}; retained logs identify the blocker")
                time.sleep(0.5)
            if any(p.returncode != 0 for p, _, _ in workers):
                raise RuntimeError(f"Arm failed during {phase}; retained logs identify the blocker")
            phase_record["status"] = "complete"
        except BaseException:
            report["status"], phase_record["status"] = "failed", "failed"
            for p, _, _ in workers:
                if p.poll() is None:
                    p.terminate()
            for p, _, _ in workers:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            raise
        finally:
            for p, _, job in workers:
                state = report["arms"][job["arm_id"]]
                state["worker_wall_seconds"] += time.perf_counter() - job["started"]
                state["terminal_status"] = ("complete" if phase == "train" else "preflight_complete") if p.returncode == 0 else "failed"
                state["exit_reason"] = phase + "_worker_exit_" + str(p.returncode)
            if report["status"] == "failed":
                for state in report["arms"].values():
                    if not state["training_started"] and state["terminal_status"] != "failed":
                        state["terminal_status"] = "cancelled"
                        state["exit_reason"] = "all_arm_preflight_barrier_blocked_training"
            phase_record["workers"] = [{"arm_id": j["arm_id"], "device": j["device"],
                "returncode": p.returncode} for p, _, j in workers]
            for _, log, _ in workers:
                log.close()
            report["phases"].append(phase_record)
            report["pair_process_wall_seconds"] = time.perf_counter() - started
            report["scope"] = "orchestration window excluding bootstrap and queue; not per-arm native device cost"
            atomic_json(output / "pair_state.json", report)
    report["status"] = "complete"
    atomic_json(output / "pair_state.json", report)
    return report
