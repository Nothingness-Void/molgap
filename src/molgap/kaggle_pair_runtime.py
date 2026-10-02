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
from .experiment_allocation import AllocationLedger


def run_two_phase_pair(*, source_root: Path, package_dir: Path, input_root: Path,
                       launch_path: Path, output: Path, allocation_started=None,
                       manifest=None) -> dict:
    """Preflight every assigned arm before any formal training is spawned."""
    source_root, package_dir, input_root, output = map(Path, (source_root, package_dir, input_root, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Pair runtime requires a fresh output root; use frozen portable recovery")
    config = _json(launch_path)
    spec = ExperimentSpec.from_json((package_dir / "experiment_spec.json").read_text(encoding="utf-8"))
    if manifest is None:
        from .experiment_package import verify_experiment_source_package
        manifest = verify_experiment_source_package(package_dir)
    if (manifest["spec_identity"] != spec.identity
            or manifest["package_identity"] != config["expected_package_identity"]
            or manifest["archive_sha256"] != config["expected_source_archive_sha256"]):
        raise ValueError("Runtime launch differs from the verified source package")
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
    prior_segments = []
    if "resume" in config:
        from .experiment_family_workflow import RunContext
        from .experiment_resume import validate_resume_bundle, restore_resume_bundle
        from .screen_policy import canonical_fingerprint
        checked, seen_segments = {}, set()
        for job in jobs:
            arm_id = job["arm_id"]
            context = RunContext.for_training(spec, package_dir,
                expected_package_identity=config["expected_package_identity"], arm_id=arm_id,
                account=config["account"], run_reference=config["run_reference"])
            bundle = Path(launch_path).parent / config["resume"][arm_id]["manifest"]
            trajectory = Path(launch_path).parent / "prospective" / arm_id / "trajectory.json"
            checked[arm_id] = (bundle, context, trajectory)
            item = validate_resume_bundle(bundle, spec, arm_id=arm_id, context=context, trajectory=trajectory)
            for path in item["sidecars"]["cost_segments"]:
                segment = _json(path)
                if segment.get("format") == "molgap-allocation-ledger-v1":
                    identity = canonical_fingerprint(segment)
                    if identity not in seen_segments:
                        prior_segments.append(segment)
                        seen_segments.add(identity)
        for arm_id, (bundle, context, trajectory) in checked.items():
            restore_resume_bundle(bundle, output / arm_id, spec, arm_id=arm_id,
                                  context=context, trajectory=trajectory)
    report = {"format": "molgap-kaggle-two-phase-pair-v2", "spec_identity": spec.identity,
              "hardware": names, "phases": [], "status": "running",
              "arms": {j["arm_id"]: {"device": j["device"], "training_started": "resume" in config,
                        "resumed_from_checkpoint": "resume" in config,
                        "terminal_status": "not_started", "worker_wall_seconds": 0.0}
                       for j in jobs}}
    started = time.perf_counter()
    ledger = AllocationLedger(spec_identity=spec.identity, hardware=names,
        assignments={job["arm_id"]: job["device"] for job in jobs}, started=allocation_started,
        prior_segments=prior_segments)
    ledger.write(output, "running")
    atomic_json(output / "pair_state.json", report)
    def retain_execution():
        from .experiment_retention import seal_execution_retention
        atomic_json(output / "execution_report.json", report)
        seal_execution_retention(output, spec, package_identity=manifest["package_identity"],
            source_commit=manifest["source_commit"], source_archive_sha256=manifest["archive_sha256"],
            arm_roots={j["arm_id"]: output / j["arm_id"] for j in jobs})
    for phase in ("preflight", "train"):
        workers, phase_record = [], {"phase": phase, "workers": [], "status": "running"}
        ledger_updated = time.perf_counter()
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
                preflight_output = output / "_resume_preflight" / job["arm_id"] if "resume" in config else arm_output
                phase_output = preflight_output if phase == "preflight" else arm_output
                command = [sys.executable, "-m", "molgap.experiment_training_worker",
                    "--source-root", str(source_root), "--package-dir", str(package_dir),
                    "--input-root", str(input_root), "--output", str(phase_output),
                    "--launch", str(launch_path), "--arm", job["arm_id"], "--phase", phase]
                if phase == "train":
                    command.extend(["--preflight-output", str(preflight_output)])
                elif "resume" in config:
                    command.extend(["--resume-output", str(arm_output)])
                env = dict(os.environ)
                env.update(CUDA_VISIBLE_DEVICES=str(job["device"]), PYTHONHASHSEED="42",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONPATH=str(source_root / "src"))
                log = (arm_output / (phase + ".log")).open("a", encoding="utf-8")
                try:
                    process = subprocess.Popen(command, cwd=source_root, env=env,
                                               stdout=log, stderr=subprocess.STDOUT)
                except BaseException as exc:
                    log.close()
                    state["terminal_status"] = "failed"
                    state["exit_reason"] = phase + "_worker_spawn_" + type(exc).__name__
                    raise
                workers.append((process, log, {**job, "started": time.perf_counter()}))
            def poll_workers():
                for process, _, item in workers:
                    if process.poll() is not None and "finished" not in item:
                        item["finished"] = time.perf_counter()
                return any(p.returncode is None for p, _, _ in workers)
            while poll_workers():
                if any(p.poll() not in (None, 0) for p, _, _ in workers):
                    raise RuntimeError(f"Arm failed during {phase}; retained logs identify the blocker")
                if time.perf_counter() - ledger_updated >= 30:
                    ledger.write(output, "running", arm_roots=[output / j["arm_id"] for j in jobs])
                    ledger_updated = time.perf_counter()
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
                state["worker_wall_seconds"] += job.get("finished", time.perf_counter()) - job["started"]
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
            ledger.write(output, report["status"],
                arm_roots=[output / j["arm_id"] for j in jobs])
            if report["status"] == "failed":
                try:
                    retain_execution()
                except Exception as exc:
                    report["retention_error"] = str(exc)
                    try:
                        atomic_json(output / "pair_state.json", report)
                    except OSError:
                        pass  # Preserve the original worker failure if storage also failed.
    report["status"] = "complete"
    atomic_json(output / "pair_state.json", report)
    ledger.write(output, "complete", arm_roots=[output / j["arm_id"] for j in jobs])
    retain_execution()
    return report
