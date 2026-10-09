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
                       launch_path: Path, output: Path, maximum_wall_seconds: float | None = None) -> dict:
    """Preflight every assigned arm before any formal training is spawned."""
    source_root, package_dir, input_root, output = map(Path, (source_root, package_dir, input_root, output))
    config = _json(launch_path)
    if maximum_wall_seconds is not None and (
        isinstance(maximum_wall_seconds, bool) or not isinstance(maximum_wall_seconds, (int, float))
        or not 0 < maximum_wall_seconds <= 14400
    ):
        raise ValueError("Allocation ceiling must be positive and at most four hours")
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
                        "started": False, "terminal_status": "not_started",
                        "exit_reason": "not_started", "worker_wall_seconds": 0.0}
                       for j in jobs}}
    started = time.perf_counter()
    deadline = None if maximum_wall_seconds is None else started + maximum_wall_seconds
    if deadline is not None:
        report["maximum_wall_seconds"] = maximum_wall_seconds
    atomic_json(output / "pair_state.json", report)
    for phase in ("preflight", "train"):
        workers = []
        worker_rows = {job["arm_id"]: {"arm_id": job["arm_id"], "device": job["device"],
            "started": False, "terminal_status": "not_started", "exit_reason": "not_started",
            "returncode": None, "worker_wall_seconds": 0.0} for job in jobs}
        phase_record = {"phase": phase, "workers": list(worker_rows.values()), "status": "running"}
        report["phases"].append(phase_record)
        atomic_json(output / "pair_state.json", report)

        def record_exit(worker, returncode, *, terminal_status=None, exit_reason=None):
            if worker["finished"]:
                return
            elapsed = max(0.0, time.perf_counter() - worker["started_at"])
            job = worker["job"]
            state = report["arms"][job["arm_id"]]
            row = worker_rows[job["arm_id"]]
            worker["finished"] = True
            worker["returncode"] = returncode
            status = terminal_status or (("preflight_complete" if phase == "preflight" else "complete")
                                         if returncode == 0 else "failed")
            reason = exit_reason or phase + "_worker_exit_" + str(returncode)
            state["worker_wall_seconds"] += elapsed
            state["terminal_status"] = status
            state["exit_reason"] = reason
            row.update(terminal_status=status, exit_reason=reason,
                       returncode=returncode, worker_wall_seconds=elapsed)
            report["pair_process_wall_seconds"] = time.perf_counter() - started
            atomic_json(output / "pair_state.json", report)

        try:
            for job in jobs:
                state = report["arms"][job["arm_id"]]
                state["terminal_status"] = "starting"
                state["exit_reason"] = "worker_starting"
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
                row = worker_rows[job["arm_id"]]
                try:
                    log = (arm_output / (phase + ".log")).open("a", encoding="utf-8")
                except BaseException as exc:
                    if phase == "train":
                        state["training_started"] = False
                    state["terminal_status"] = "failed"
                    state["exit_reason"] = phase + "_worker_spawn_error_" + type(exc).__name__
                    row.update(terminal_status="failed", exit_reason=state["exit_reason"])
                    atomic_json(output / "pair_state.json", report)
                    raise
                worker_started = time.perf_counter()
                try:
                    process = subprocess.Popen(command, cwd=source_root, env=env,
                                               stdout=log, stderr=subprocess.STDOUT)
                except BaseException as exc:
                    log.close()
                    if phase == "train":
                        state["training_started"] = False
                    state["terminal_status"] = "failed"
                    state["exit_reason"] = phase + "_worker_spawn_error_" + type(exc).__name__
                    row.update(terminal_status="failed", exit_reason=state["exit_reason"])
                    atomic_json(output / "pair_state.json", report)
                    raise
                workers.append({"process": process, "log": log, "job": job,
                                "started_at": worker_started, "finished": False,
                                "returncode": None})
                state["started"] = True
                state["terminal_status"] = "running"
                state["exit_reason"] = "worker_running"
                row.update(started=True, terminal_status="running", exit_reason="worker_running")
                atomic_json(output / "pair_state.json", report)
            while True:
                if deadline is not None and time.perf_counter() >= deadline:
                    report["stop_reason"] = "STOP_FOR_COST"
                    raise TimeoutError("Frozen allocation ceiling exhausted; retain complete-epoch resume state")
                running = False
                phase_failed = False
                for worker in workers:
                    if worker["finished"]:
                        phase_failed |= worker["returncode"] != 0
                        continue
                    code = worker["process"].poll()
                    if code is None:
                        running = True
                        continue
                    record_exit(worker, code)
                    phase_failed |= code != 0
                # Qualification is an all-arm barrier. A training peer owns its
                # independent lifetime, so it finishes after another arm exits.
                if phase == "preflight" and phase_failed:
                    raise RuntimeError(f"Arm failed during {phase}; retained logs identify the blocker")
                if not running:
                    if phase_failed:
                        raise RuntimeError(f"Arm failed during {phase}; retained logs identify the blocker")
                    break
                time.sleep(0.5)
            phase_record["status"] = "complete"
        except BaseException as exc:
            report["status"], phase_record["status"] = "failed", "failed"
            interrupted = isinstance(exc, KeyboardInterrupt)
            for worker in workers:
                process = worker["process"]
                if worker["finished"]:
                    continue
                try:
                    code = process.poll()
                except BaseException:
                    code = None
                if code is not None:
                    record_exit(worker, code)
                else:
                    worker["cleanup_requested"] = True
                    try:
                        process.terminate()
                    except BaseException:
                        pass
            for worker in workers:
                process = worker["process"]
                if worker["finished"]:
                    continue
                try:
                    code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                        code = process.wait(timeout=10)
                    except BaseException:
                        try:
                            code = process.poll()
                        except BaseException:
                            code = None
                except BaseException:
                    try:
                        code = process.poll()
                    except BaseException:
                        code = None
                if code is not None:
                    if worker.get("cleanup_requested") and code in {-15, -9}:
                        status = "interrupted" if interrupted else "cancelled"
                        reason = phase + ("_worker_interrupted" if interrupted else "_worker_cancelled_after_phase_error")
                        record_exit(worker, code, terminal_status=status, exit_reason=reason)
                    else:
                        record_exit(worker, code)
            # Arms without a spawned child have a known zero formal-training
            # start and an explicit terminal reason.
            for job in jobs:
                state = report["arms"][job["arm_id"]]
                row = worker_rows[job["arm_id"]]
                if not row["started"] and row["terminal_status"] == "not_started":
                    if phase == "train":
                        state["training_started"] = False
                    state["terminal_status"] = "cancelled"
                    state["exit_reason"] = phase + "_worker_not_started_after_phase_error"
                    row.update(terminal_status="cancelled", exit_reason=state["exit_reason"])
                elif row["started"] and row["terminal_status"] == "running":
                    state["terminal_status"] = "unknown"
                    state["exit_reason"] = phase + "_worker_exit_unknown_after_cleanup"
                    row.update(terminal_status=state["terminal_status"],
                               exit_reason=state["exit_reason"])
            raise
        finally:
            for worker in workers:
                worker["log"].close()
            report["pair_process_wall_seconds"] = time.perf_counter() - started
            report["scope"] = "orchestration window excluding bootstrap and queue; not per-arm native device cost"
            atomic_json(output / "pair_state.json", report)
    report["status"] = "complete"
    atomic_json(output / "pair_state.json", report)
    return report
