"""Two assigned T4 subprocesses, with a pair-wide preflight barrier."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

from .training_reproducibility import atomic_json


def run_two_phase_pair(*, source_root: Path, jobs: list[dict], output: Path) -> dict:
    """Each job supplies explicit run_arm CLI flags; both qualify before training."""
    if (len(jobs) != 2 or {job.get("mode") for job in jobs} != {"reference", "ssma"} or
        {job.get("device") for job in jobs} != {0, 1}):
        raise ValueError("Expected one reference/SSMA pair assigned to devices0/1")
    names = [line.strip() for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
        if line.strip()]
    if len(names) != 2 or any("T4" not in name for name in names):
        raise RuntimeError(f"Expected two physical T4s, received {names}")
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    entry = source_root / "experiments/pcqm_k1_local_mixing_clean_aux/run_arm.py"
    if not entry.is_file():
        raise FileNotFoundError(entry)
    report = {"format": "molgap-kaggle-two-phase-pair-v1", "hardware": names,
        "phases": [], "status": "running"}
    started = time.perf_counter()
    for phase in ("preflight", "train"):
        workers = []
        phase_record = {"phase": phase, "workers": []}
        try:
            for job in sorted(jobs, key=lambda item: item["device"]):
                flags = job["arguments"]
                if not isinstance(flags, dict) or any(not isinstance(key, str) for key in flags):
                    raise ValueError("Worker arguments must be explicit CLI flag mapping")
                if set(flags) & {"phase", "mode", "output", "preflight-dir"}:
                    raise ValueError("Pair runtime owns phase/mode/output/preflight paths")
                worker_output = output / job["mode"]
                worker_output.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(entry), "--phase", phase, "--mode", job["mode"],
                    "--output", str(worker_output)]
                if phase == "train":
                    command += ["--preflight-dir", str(worker_output)]
                for key, value in sorted(flags.items()):
                    if value is not None:
                        command += ["--" + key, str(value)]
                env = dict(os.environ)
                env.update(CUDA_VISIBLE_DEVICES=str(job["device"]), PYTHONHASHSEED="42",
                    CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONPATH=str(source_root / "src"))
                log = (worker_output / (phase + ".log")).open("a", encoding="utf-8")
                process = subprocess.Popen(command, cwd=source_root, env=env,
                    stdout=log, stderr=subprocess.STDOUT)
                workers.append((process, log, job))
            while any(process.poll() is None for process, _, _ in workers):
                if any(process.poll() not in (None, 0) for process, _, _ in workers):
                    raise RuntimeError(f"Paired {phase} worker failed; see retained logs")
                time.sleep(0.5)
            if any(process.returncode != 0 for process, _, _ in workers):
                raise RuntimeError(f"Paired {phase} worker failed; see retained logs")
            phase_record["workers"] = [{"mode": job["mode"], "device": job["device"],
                "returncode": process.returncode} for process, _, job in workers]
            phase_record["status"] = "complete"
        except BaseException:
            report["status"] = "failed"
            phase_record["status"] = "failed"
            for process, _, _ in workers:
                if process.poll() is None:
                    process.terminate()
            for process, _, _ in workers:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            for _, log, _ in workers:
                log.close()
            report["phases"].append(phase_record)
            report["pair_process_wall_seconds"] = time.perf_counter() - started
            report["scope"] = ("pair orchestration window; excludes kernel bootstrap and queue; "
                               "per-arm assigned T4 allocation measured by trainer")
            atomic_json(output / "pair_state.json", report)
    report["status"] = "complete"
    atomic_json(output / "pair_state.json", report)
    return report
