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
    from .family_pair_release import bounded_cycle_pair, WALL_CEILING_SECONDS
    bounded = bounded_cycle_pair(spec)
    campaign = (len(declaration["arms"]) == 2 and all(a["family"] == {"name": "gptrans_t", "version": "2"}
        and len(a["addons"]) == 1 and a["addons"][0]["name"].startswith("degree_aggregate_") for a in declaration["arms"]))
    if campaign:
        from .gptrans_campaign_release import check_campaign, BASE
        number = 1 if declaration["experiment_id"] == "gptrans-research-slot1" else 2
        check_campaign(spec, source_root, _json(source_root / BASE / f"slot{number}/acceptance_plan.json"),
            initial_states={j["arm_id"]: Path(launch_path).parent / j["initial_state"] for j in jobs})
    if bounded:
        from .family_pair_release import check_same_run_plan
        check_same_run_plan(spec, source_root,
            _json(source_root / "experiments/pcqm_gptrans_cycle_pair_100k/gpu/acceptance_plan.json"),
            initial_states={j["arm_id"]: Path(launch_path).parent / j["initial_state"] for j in jobs})
    bounded = bounded or campaign
    if declaration["platform"]["name"] != "kaggle":
        raise ValueError("Kaggle runtime requires a Kaggle Spec")
    names = [n.strip() for n in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines() if n.strip()]
    accelerator = declaration["platform"]["accelerator"]
    if accelerator not in {"Tesla T4", "T4", "NvidiaTeslaT4"} or len(names) != declaration["platform"]["device_count"] or any("T4" not in n for n in names):
        raise RuntimeError("Observed hardware differs from the declared T4 allocation")
    output.mkdir(parents=True, exist_ok=True)
    prior_segments = []
    if campaign:
        # A bootstrap-only restart has no checkpoint to restore, but its two
        # allocated devices still consume the unchanged campaign budget.
        retired = source_root / BASE / f"infrastructure_costs/slot{number}.json"
        if retired.is_file():
            prior_segments.append(_json(retired))
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
    prior_wall = sum(s.get("wall_seconds", 0) for s in ledger.prior_segments)
    if bounded and ledger.snapshot("running")["prior_unobserved_intervals"]:
        raise ValueError("Unknown prior allocation intervals require cost reconciliation before continuation")
    ledger.write(output, "running")
    atomic_json(output / "pair_state.json", report)
    def retain_execution():
        from .experiment_retention import seal_execution_retention
        atomic_json(output / "execution_report.json", report)
        seal_execution_retention(output, spec, package_identity=manifest["package_identity"],
            source_commit=manifest["source_commit"], source_archive_sha256=manifest["archive_sha256"],
            arm_roots={j["arm_id"]: output / j["arm_id"] for j in jobs})
    for phase in ("preflight", "train"):
        if bounded and phase == "train":
            resources = [_json(output / job["arm_id"] / "resource_qualification.json") for job in jobs]
            elapsed = prior_wall + time.perf_counter() - ledger.started
            if elapsed + max(r["estimated_remaining_seconds"] for r in resources) >= WALL_CEILING_SECONDS - 120:
                report["status"] = "failed"
                report["exit_reason"] = "native_whole_allocation_budget_gate"
                ledger.write(output, "failed", arm_roots=[output / j["arm_id"] for j in jobs])
                retain_execution()
                raise TimeoutError("Whole allocation budget blocks formal training")
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
                if bounded and prior_wall + time.perf_counter() - ledger.started >= WALL_CEILING_SECONDS - 60:
                    report["exit_reason"] = "native_allocation_hard_budget"
                    raise TimeoutError("Frozen8wall-hour allocation ceiling; stop and retain checkpoints")
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
                if phase == "train" and p.returncode == 0 and (output / job["arm_id"] / "pause_receipt.json").is_file():
                    state["terminal_status"] = "interrupted"
                    state["exit_reason"] = "durable_ten_pass_segment_complete_scientific_horizon_incomplete"
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


def main():
    import argparse
    from .experiment_launch_config import validate_launch_config
    from .experiment_package import verify_experiment_source_package
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for field in ("source-root", "package-dir", "input-root", "launch", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--allocation-started", type=float, required=True)
    args = parser.parse_args()
    manifest = verify_experiment_source_package(args.package_dir)
    spec = ExperimentSpec.from_json((args.package_dir / "experiment_spec.json").read_text(encoding="utf-8"))
    validate_launch_config(spec, args.launch, manifest=manifest, package_dir=args.package_dir,
                          staged_root=args.launch.parent)
    run_two_phase_pair(source_root=args.source_root, package_dir=args.package_dir,
        input_root=args.input_root, launch_path=args.launch, output=args.output,
        allocation_started=args.allocation_started, manifest=manifest)


if __name__ == "__main__":
    main()
