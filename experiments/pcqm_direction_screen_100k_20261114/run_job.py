"""Run one packed job: up to --slots concurrent train_arm.py children on ONE GPU, longest-first.

Skips arms that already have done.json, does not start an arm whose estimated time (U * s) will not fit before the
deadline, forwards SIGTERM (CTRL_BREAK on Windows) to children at the deadline, logs nvidia-smi + job_status.json
every 30 s. Children resume from their own ckpt_last.pt, so resubmitting the same job continues the work.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_data import atomic_json  # noqa: E402

IS_WIN = os.name == "nt"
STOP = {"flag": False}


def _on_signal(signum, frame):
    STOP["flag"] = True


def terminate(proc):
    try:
        if IS_WIN:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGTERM)
    except Exception:
        pass


def gpu_line():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--plan", default=str(HERE / "jobs.json"))
    ap.add_argument("--arms-json", default=str(HERE / "arms.json"))
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--deadline-s", type=float, required=True, help="seconds from now after which children are told to stop")
    ap.add_argument("--root", required=True)
    ap.add_argument("--export-root", required=True)
    ap.add_argument("--features", default=None)
    ap.add_argument("--s", type=float, default=2.0, help="assumed slowdown per arm for the start/no-start decision")
    ap.add_argument("--only", default=None, help="comma list restricting arms (testing)")
    ap.add_argument("--grace-s", type=float, default=180.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--dev-limit", type=int, default=None)
    a = ap.parse_args()
    for sg in ("SIGTERM", "SIGINT", "SIGBREAK"):
        if hasattr(signal, sg):
            signal.signal(getattr(signal, sg), _on_signal)
    t0 = time.time()
    root = Path(a.root)
    runs, logs, stat = root / "runs", root / "logs", root / "job_status"
    for d in (runs, logs, stat):
        d.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))["jobs"][a.job]
    queue = sorted(plan["arms"], key=lambda r: -r["U"])
    if a.only:
        keep = set(a.only.split(","))
        queue = [r for r in queue if r["arm"] in keep]
    status = {r["arm"]: {"state": "queued", "U": r["U"]} for r in queue}
    running = {}
    gpu_csv = logs / f"{a.job}_gpu.csv"
    last_beat = 0.0
    deadline_hit = False

    def beat():
        atomic_json(stat / f"{a.job}.json", {"job": a.job, "elapsed_s": time.time() - t0, "deadline_s": a.deadline_s,
                                            "deadline_hit": deadline_hit, "arms": status})
        g = gpu_line()
        if g:
            with open(gpu_csv, "a", encoding="utf-8") as f:
                f.write(f"{a.job},{g}\n")

    while queue or running:
        now = time.time() - t0
        if (now > a.deadline_s or STOP["flag"]) and not deadline_hit:
            deadline_hit = True
            for arm, p in running.items():
                terminate(p)
            term_at = time.time()
            for r in queue:
                if status[r["arm"]]["state"] == "queued":
                    status[r["arm"]]["state"] = "not_started_deadline"
            queue = []
        if deadline_hit and running and time.time() - term_at > a.grace_s:
            for p in running.values():
                p.kill()
        for arm in list(running):
            rc = running[arm].poll()
            if rc is not None:
                done = (runs / arm / "done.json").is_file()
                status[arm].update(state="done" if done else ("interrupted" if rc == 75 or deadline_hit else "failed"), rc=rc)
                running.pop(arm)
        while queue and len(running) < a.slots and not deadline_hit:
            r = queue.pop(0)
            arm = r["arm"]
            if (runs / arm / "done.json").is_file():
                status[arm]["state"] = "skipped_done"
                continue
            if now + r["U"] * 3600 * a.s > a.deadline_s + 1 and a.max_steps is None:
                status[arm]["state"] = "skipped_wont_fit"
                continue
            cmd = [sys.executable, "-u", str(HERE / "train_arm.py"), "--arm", arm, "--arms-json", a.arms_json,
                   "--export-root", a.export_root, "--out", str(runs)]
            if a.features:
                cmd += ["--features", a.features]
            if a.max_steps is not None:
                cmd += ["--max-steps", str(a.max_steps)]
            if a.dev_limit is not None:
                cmd += ["--dev-limit", str(a.dev_limit)]
            lf = open(logs / f"{arm}.log", "a", encoding="utf-8")
            kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WIN else {}
            running[arm] = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, **kw)
            status[arm].update(state="running", started_s=time.time() - t0)
            print(f"[{a.job}] start {arm} (running={len(running)})", flush=True)
        if time.time() - last_beat > 30 or not running:
            beat()
            last_beat = time.time()
        time.sleep(1.0 if not IS_WIN else 1.0)
    beat()
    print(json.dumps({k: v["state"] for k, v in status.items()}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
