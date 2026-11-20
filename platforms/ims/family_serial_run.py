"""Serial driver: run several frozen family recipes one after another inside ONE IMS job.

Queue time dwarfs run time on IMS, so a chain of trials shares one job. Each item is a child process of
run_family_training.py (the existing entry that calls family_recipe_run.py). A failed child never blocks the next
item. An item that cannot finish in the wall left is recorded as SKIPPED_FOR_TIME. A finished item (its
completion_manifest.json exists) is never redone, so the same command resumes after a pause, a kill or a new job.
Every outcome goes to one summary JSON written atomically.

Spec (JSON, committed and bound by the launcher as the run contract):
  {"format": "molgap-ims-family-serial-v1",
   "items": [{"name", "recipe", "recipe_sha256", "initial_state"?, "initial_state_sha256"?, "budget_s",
              "extra_args"?, "gate"?}]}
Paths are relative to --code-root. "gate" is optional and sits on the item it may skip:
  "dev_delta": {"a": item, "b": item, "max_delta_meV": -1.5}   skip unless (MAE_a - MAE_b) * 1000 <= max_delta
  "memory":    {"from": item, "factor": 5, "margin": 1.3, "max_fraction": 0.5}
                skip if peak_rss(from) * factor * margin > max_fraction * min(cgroup memory.max, MemAvailable)
A gate that cannot read its input (item failed, no selected_eval.json, no peak) skips: no evidence, no expensive run.

Exit 0: every item is COMPLETE or gate-skipped. 2: something can resume (PAUSED, NOT_STARTED, SKIPPED_FOR_TIME).
1: an item FAILED and nothing can resume. 3: the start-of-job check rejected an input (no child ran).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FORMAT = "molgap-ims-family-serial-v1"
SUMMARY = "serial_summary.json"
REJECTED = 3
PAUSED = 2
TAIL_START = 3_250_000  # tail_holdout_v1 begins here
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{1,47}$")
HERE = Path(__file__).resolve().parent


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    names = [item.get("name") for item in spec.get("items", [])]
    if spec.get("format") != FORMAT or not names or len(set(names)) != len(names) or not all(NAME.fullmatch(str(n)) for n in names):
        raise ValueError("Serial spec needs the right format and unique safe item names")
    return spec


def check_inputs(spec: dict, code_root: Path, input_root: Path) -> list[str]:
    """Cheap identity and presence checks for every item, before any child runs. Returns the problems found."""
    problems: list[str] = []
    for item in spec["items"]:
        name = item["name"]
        try:
            recipe = code_root / item["recipe"]
            if not recipe.is_file() or sha256(recipe) != item["recipe_sha256"]:
                problems.append(f"{name}: recipe missing or its sha256 differs from the spec")
                continue
            record = json.loads(recipe.read_text(encoding="utf-8"))
            manifest_path = input_root / record["dataset"]["manifest"]
            if not manifest_path.is_file() or sha256(manifest_path) != record["dataset"]["manifest_sha256"]:
                problems.append(f"{name}: dataset manifest missing or its sha256 differs from the recipe")
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                problems.append(f"{name}: dataset manifest is not complete")
            train = manifest["roles"]["train"]
            if train["source_idx_stop"] > TAIL_START or train["rows"] != record["train_rows"]:
                problems.append(f"{name}: train rows reach tail_holdout_v1 or differ from the recipe")
            else:
                import torch
                from molgap.experiment_family_workflow import tensor_digest
                digest = tensor_digest(torch.arange(train["source_idx_start"], train["source_idx_stop"]), role="source_idx")
                if digest != record["train_role_identity"]:
                    problems.append(f"{name}: train role identity of the manifest differs from the recipe")
            for asset in manifest.get("assets", {}).get("topology", []):
                shard = input_root / asset["file"]
                if not shard.is_file() or shard.stat().st_size != asset["bytes"]:
                    problems.append(f"{name}: data file missing or wrong size: {asset['file']}")
            if item.get("initial_state"):
                state = code_root / item["initial_state"]
                if not state.is_file() or sha256(state) != item.get("initial_state_sha256"):
                    problems.append(f"{name}: initial state missing or its sha256 differs from the spec")
        except (KeyError, OSError, ValueError, TypeError) as error:
            problems.append(f"{name}: {type(error).__name__}: {error}")
    return problems


def memory_limit_bytes() -> tuple[int, str]:
    """min(cgroup memory.max, MemAvailable). Linux is the IMS path; psutil is the local fallback."""
    candidates: dict[str, int] = {}
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(path).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 1 << 60:  # "max" or a huge number means no cgroup limit
            candidates["cgroup"] = int(text)
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates["MemAvailable"] = int(line.split()[1]) * 1024
    except OSError:
        import psutil
        candidates["psutil_available"] = psutil.virtual_memory().available
    source = min(candidates, key=candidates.get)
    return candidates[source], source


def run_child(command: list[str], env: dict, on_start) -> tuple[int, int | None]:
    """Run one child and return (exit code, peak RSS in bytes of the child and its waited descendants).
    Linux: wait4 gives the rusage of this child alone (ru_maxrss is in KiB). Elsewhere: sample the process tree."""
    child = subprocess.Popen(command, env=env)
    on_start(child)
    if hasattr(os, "wait4"):
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        return child.returncode, usage.ru_maxrss * 1024
    import psutil
    peak = 0
    while child.poll() is None:
        try:
            tree = [psutil.Process(child.pid)]
            tree += tree[0].children(recursive=True)
            peak = max(peak, sum(p.memory_info().peak_wset for p in tree))
        except psutil.Error:
            pass
        time.sleep(1.0)
    return child.returncode, peak or None


def dev_mae(out_dir: Path) -> float | None:
    path = out_dir / "selected_eval.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["dev_mae_eV"]) if path.is_file() else None


def is_complete(out_dir: Path) -> bool:
    path = out_dir / "completion_manifest.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") == "COMPLETE"


def gate_decision(item: dict, tasks: dict, out_dirs: dict, limit: tuple[int, str]) -> tuple[str, str] | None:
    """(status, reason) when the gate says skip, else None."""
    gate = item.get("gate") or {}
    if "dev_delta" in gate:
        rule = gate["dev_delta"]
        a, b = (dev_mae(out_dirs[rule[key]]) if tasks.get(rule[key], {}).get("status") == "COMPLETE" else None for key in ("a", "b"))
        if a is None or b is None:
            return "SKIPPED_BY_GATE", f"no dev MAE for {rule['a'] if a is None else rule['b']}; the gate cannot show the gain"
        delta, line = (a - b) * 1000, rule.get("max_delta_meV", -1.5)
        if delta > line:
            return "SKIPPED_BY_GATE", f"dev MAE delta {rule['a']} - {rule['b']} = {delta:.3f} meV is above {line} meV"
    if "memory" in gate:
        rule = gate["memory"]
        peak = tasks[rule["from"]].get("peak_rss_bytes") if rule["from"] in tasks else None
        if not peak:
            return "SKIPPED_FOR_MEMORY", f"no measured peak memory for {rule['from']}"
        need = peak * rule.get("factor", 5) * rule.get("margin", 1.3)
        budget = rule.get("max_fraction", 0.5) * limit[0]
        if need > budget:
            return "SKIPPED_FOR_MEMORY", (f"predicted peak {need / 2**30:.1f} GiB ({rule['from']} {peak / 2**30:.2f} GiB x "
                                          f"{rule.get('factor', 5)} x {rule.get('margin', 1.3)}) exceeds {rule.get('max_fraction', 0.5)} x "
                                          f"{limit[0] / 2**30:.1f} GiB ({limit[1]})")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--code-root", type=Path, required=True)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--cpus", type=int, required=True)
    ap.add_argument("--accelerator")
    ap.add_argument("--ogb-zip", type=Path)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--training-workers", type=int, default=1, help="0 on Windows rehearsals")
    ap.add_argument("--wall-s", type=float, required=True, help="seconds this job may run; the clock starts at driver start")
    ap.add_argument("--reserve-s", type=float, default=1800.0, help="kept free at the end for the last checkpoint")
    ap.add_argument("--continue-after-pause", action="store_true", help="local rehearsal only: a paused item does not stop the driver")
    ap.add_argument("--memory-limit-bytes", type=int, help="local rehearsal only: replace the memory probe")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    spec, root = load_spec(a.spec), a.output_root.resolve()
    code_root, input_root = a.code_root.resolve(), a.input_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    problems = check_inputs(spec, code_root, input_root)
    if problems:
        atomic_json(root / "early_exit.json", {"status": "REJECTED", "utc": now(), "problems": problems})
        print("[driver] REJECTED before any child ran:\n  " + "\n  ".join(problems), flush=True)
        return REJECTED
    names = [item["name"] for item in spec["items"]]
    out_dirs = {name: root / name for name in names}
    summary_path = root / SUMMARY
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {
        "format": "molgap-ims-family-serial-summary-v1", "order": names, "driver_starts": [],
        "items": {n: {"status": "NOT_STARTED", "attempts": []} for n in names}}
    summary.update(wall_s=a.wall_s, reserve_s=a.reserve_s, spec_sha256=sha256(a.spec), source_commit=a.source_commit)
    summary["driver_starts"].append(now())
    tasks = summary["items"]
    for task in tasks.values():
        if task["status"] == "RUNNING":  # a hard kill left it; it resumes from its own checkpoint
            task["status"] = "NOT_STARTED"
    state = {"stop": False, "child": None}

    def write() -> None:
        atomic_json(summary_path, summary)

    def on_term(_signum, _frame) -> None:
        state["stop"] = True
        if state["child"] is not None and state["child"].poll() is None:
            state["child"].send_signal(signal.SIGTERM)  # the child checkpoints and exits 2

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    limit = (a.memory_limit_bytes, "override") if a.memory_limit_bytes else memory_limit_bytes()
    summary["memory_limit"] = {"bytes": limit[0], "source": limit[1]}
    write()
    for item in spec["items"]:
        name, out_dir, task = item["name"], out_dirs[item["name"]], tasks[item["name"]]
        task["output"] = str(out_dir)
        if is_complete(out_dir):
            task.update(status="COMPLETE", reason="completion_manifest.json exists", dev_mae_eV=dev_mae(out_dir))
            continue
        if state["stop"]:
            task.update(status="NOT_STARTED", reason="driver was told to stop before this item")
            continue
        skip = gate_decision(item, tasks, out_dirs, limit)
        if skip:
            task.update(status=skip[0], reason=skip[1])
            print(f"[driver] {name} {skip[0]}: {skip[1]}", flush=True)
            write()
            continue
        left = a.wall_s - (time.monotonic() - t0)
        spent = sum(x.get("wall_s", 0) for x in task["attempts"])
        need = max(item["budget_s"] - spent, 0.1 * item["budget_s"])
        if left - a.reserve_s < need:
            task.update(status="SKIPPED_FOR_TIME", reason=f"needs about {need:.0f}s, has {max(left - a.reserve_s, 0):.0f}s after the {a.reserve_s:.0f}s reserve")
            print(f"[driver] {name} SKIPPED_FOR_TIME: {task['reason']}", flush=True)
            write()
            continue
        command = [sys.executable, "-u", str(HERE / "run_family_training.py"), "--device", str(a.device),
                   "--recipe", str(code_root / item["recipe"]), "--input-root", str(input_root), "--output", str(out_dir),
                   "--phase", "train", "--no-preflight-receipt", "--source-commit", a.source_commit, "--cpus", str(a.cpus),
                   "--training-workers", str(a.training_workers), "--wall-s", str(max(left - a.reserve_s, 1))]
        if item.get("initial_state"):
            command += ["--initial-state", str(code_root / item["initial_state"])]
        if a.accelerator:
            command += ["--accelerator", a.accelerator]
        if a.ogb_zip:
            command += ["--ogb-zip", str(a.ogb_zip)]
        command += item.get("extra_args", [])
        task.update(status="RUNNING", reason="child started")
        write()
        print(f"[driver] {name} start, estimate {need:.0f}s, wall left {left:.0f}s", flush=True)
        began = time.monotonic()
        code, peak = run_child(command, dict(os.environ), lambda child: state.__setitem__("child", child))
        state["child"] = None
        task["attempts"].append({"end_utc": now(), "exit_code": code, "wall_s": round(time.monotonic() - began, 1), "peak_rss_bytes": peak})
        task["peak_rss_bytes"] = max(filter(None, [x.get("peak_rss_bytes") for x in task["attempts"]]), default=None)
        if is_complete(out_dir):
            task.update(status="COMPLETE", reason="completion_manifest.json written", dev_mae_eV=dev_mae(out_dir))
        elif code == PAUSED and (out_dir / "pause_receipt.json").is_file():
            task.update(status="PAUSED", reason="wall limit, SIGTERM or step limit; rerun the same command to resume")
        else:
            task.update(status="FAILED", reason=f"child exit code {code}" + (" (rejected by its own start checks)" if code == 1 else ""))
        print(f"[driver] {name} {task['status']}: {task['reason']}", flush=True)
        write()
        if task["status"] == "PAUSED" and not a.continue_after_pause:
            state["stop"] = True  # the wall is used up, or PBS asked us to stop
    summary["last_driver_end_utc"] = now()
    write()
    print("[driver] summary", json.dumps({n: tasks[n]["status"] for n in names}), flush=True)
    states = {t["status"] for t in tasks.values()}
    if states & {"PAUSED", "NOT_STARTED", "SKIPPED_FOR_TIME"}:
        return PAUSED
    return 1 if "FAILED" in states else 0


if __name__ == "__main__":
    sys.exit(main())
