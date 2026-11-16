"""Serial driver for one PBS job of the GPT-T 500K capacity question: R, L20, W320, F3 of one init seed, in that order.

Each task is a child process of run_gptrans_width_500k.py. A failed child never stops the next one. Before a task starts, the
driver compares an estimate of its remaining time with the wall left (minus a reserve) and records SKIPPED_FOR_TIME instead of
starting a task that cannot finish. Every outcome goes to one summary JSON (atomic writes). SIGTERM or the wall limit pauses the
running child through the runner's own atomic checkpoint; rerun the SAME command to resume. Tasks that are not COMPLETE are tried again.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gptrans_width_500k import ARMS, PASSES, atomic_json  # noqa: E402

ORDER = ("R", "L20", "W320", "F3")
PRIOR_HOURS = {"R": 1.4, "L20": 2.3, "W320": 2.0, "F3": 2.15}  # frozen estimates (+-30%); only their ratios are used once R is measured
STARTUP_S = 300.0  # shard load, preflight hashes, model build
RESERVE_S = 1800.0  # the margin the plan names; the launcher keeps the same margin for the final checkpoint
MIN_PASSES_FOR_SPEED = 3
STATUSES = ("COMPLETE", "SKIPPED_FOR_TIME", "FAILED", "PAUSED", "NOT_STARTED")


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_log(out_dir: Path) -> list[dict]:
    path = out_dir / "train_log.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()] if path.is_file() else []
    return [r for r in rows if r.get("event") == "observation"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", choices=("s42", "s43"), required=True, help="init seed: the card runs R, L20, W320, F3 of this seed")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--wall-s", type=float, required=True, help="PBS walltime in seconds; the clock starts when this driver starts")
    ap.add_argument("--rehearsal-task-s", type=float, default=0.0,
                    help="local tests only: give each child N seconds (at least 330, the runner's own floor) with no reserve; a paused task does not stop the driver")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    root = a.root.resolve()
    summary_path = root / f"width_500k_summary_{a.card}.json"
    names = [f"{arm}-{a.card}" for arm in ORDER]
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {
        "format": "molgap-width500k-summary-v1", "card": a.card, "order": names, "tasks": {n: {"status": "NOT_STARTED", "attempts": []} for n in names}}
    summary.update(wall_s=a.wall_s, last_driver_start_utc=now(), prior_hours=PRIOR_HOURS)
    stop = {"flag": False, "child": None}

    def write() -> None:
        atomic_json(summary_path, summary)

    def on_term(signum, _frame) -> None:
        stop["flag"] = True
        if stop["child"] is not None and stop["child"].poll() is None:
            stop["child"].send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def out_dir_of(name: str) -> Path:
        return root / "runs" / ARMS[name]["run_dir"] / "gptrans_pairnorm"

    def r_pass_seconds() -> tuple[float, str]:
        """Median seconds per pass of R in this card; the prior (hours / passes) until three passes were measured."""
        passes = [r["pass_sec"] for r in read_log(out_dir_of(names[0]))]
        if len(passes) >= MIN_PASSES_FOR_SPEED:
            return statistics.median(passes), f"measured: median of {len(passes)} R passes"
        return PRIOR_HOURS["R"] * 3600 / PASSES, "prior"

    def estimate(name: str) -> float:
        pass_s, _ = r_pass_seconds()
        done = len(read_log(out_dir_of(name)))
        return (PASSES - done) * pass_s * PRIOR_HOURS[name.split("-")[0]] / PRIOR_HOURS["R"] + STARTUP_S

    summary["tasks_note"] = "status of a task is COMPLETE only when training_complete.json exists after the child ended"
    for name in names:
        task = summary["tasks"][name]
        done_file = out_dir_of(name) / "training_complete.json"
        if done_file.is_file() and json.loads(done_file.read_text(encoding="utf-8")).get("status") == "COMPLETE":
            task["status"] = "COMPLETE"
            continue
        if stop["flag"]:
            task.update(status="NOT_STARTED", reason="driver was told to stop before this task")
            continue
        left = a.wall_s - (time.monotonic() - t0)
        need, basis = estimate(name), r_pass_seconds()[1]
        if left - RESERVE_S < need:
            task.update(status="SKIPPED_FOR_TIME", reason=f"needs about {need:.0f}s, has {left - RESERVE_S:.0f}s after the {RESERVE_S:.0f}s reserve (speed {basis})")
            print(f"[driver] {name} {task['status']}: {task['reason']}", flush=True)
            write()
            continue
        log_before = len(read_log(out_dir_of(name)))
        attempt = {"start_utc": now(), "estimate_s": round(need), "estimate_basis": basis, "wall_left_s": round(left)}
        env = dict(os.environ, **{f"MOLGAP_GPTRANS_{k}": str(v) for k, v in ARMS[name]["shape"].items()})
        cmd = [sys.executable, "-u", str(Path(__file__).with_name("run_gptrans_width_500k.py")), "--arm", name, "--root", str(root), "--wall-s", str(left)]
        if a.rehearsal_task_s:
            cmd[-1:] = [str(a.rehearsal_task_s), "--margin-s", "0", "--min-run-s", "1"]
        print(f"[driver] {name} start, estimate {need:.0f}s ({basis})", flush=True)
        child = subprocess.Popen(cmd, env=env)
        stop["child"] = child
        code = child.wait()
        stop["child"] = None
        new = read_log(out_dir_of(name))[log_before:]
        attempt.update(end_utc=now(), exit_code=code, passes_this_attempt=len(new), passes_done=len(read_log(out_dir_of(name))),
                       samples_per_s=round(statistics.median(r["samples_per_sec"] for r in new), 1) if new else None)
        if done_file.is_file() and json.loads(done_file.read_text(encoding="utf-8")).get("status") == "COMPLETE":
            task.update(status="COMPLETE", reason="training_complete.json written")
        elif code == 0 and (out_dir_of(name) / "pause_receipt.json").is_file():
            task.update(status="PAUSED", reason="wall limit or SIGTERM; rerun the same command to resume")
        else:
            task.update(status="FAILED", reason=f"child exit code {code}" + (" (preflight)" if code == 3 else ""))
        task["attempts"].append(attempt)
        print(f"[driver] {name} {task['status']}: {task['reason']}", flush=True)
        write()
        if task["status"] == "PAUSED" and not a.rehearsal_task_s:
            stop["flag"] = True  # the wall is used up, or PBS asked us to stop
    pass_s, basis = r_pass_seconds()
    summary["speed"] = {"r_pass_sec": pass_s, "basis": basis, "r_samples_per_s": round(500_000 / pass_s, 1)}
    summary["last_driver_end_utc"] = now()
    write()
    print("[driver] summary", json.dumps({n: summary["tasks"][n]["status"] for n in names}), flush=True)
    return 2 if any(t["status"] in ("PAUSED", "NOT_STARTED") for t in summary["tasks"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
