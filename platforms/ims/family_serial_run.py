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
Paths are relative to --code-root. "gate" is optional and sits on the item it may skip; it may only name items that
come earlier in the list:
  "dev_delta": {"a": item, "b": item, "max_delta_meV": -1.5}   skip unless (MAE_a - MAE_b) * 1000 <= max_delta
  "memory":    {"from": item, "factor": 5, "margin": 1.3, "max_fraction": 0.5}
                skip if peak_rss(from) * factor * margin > max_fraction * --memory-limit-bytes
A gate that cannot read its input (item not COMPLETE, no selected_eval.json, no peak) skips: no evidence, no expensive run.
--memory-limit-bytes is the PBS mem= request, passed by the launcher. The driver never probes cgroup or /proc files:
IMS access stops at the authorized directory (platforms/REMOTE_HANDOFF.md).

Exit 0: every item is COMPLETE or gate-skipped. 2: something can resume (PAUSED, NOT_STARTED, SKIPPED_FOR_TIME).
1: an item FAILED and nothing can resume. 3: the start-of-job check rejected the spec or an input (no child ran).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from family_recipe_run import TAIL_START  # noqa: E402  (stdlib-only module; one owner of the tail boundary)

try:
    from molgap.training_reproducibility import atomic_json, sha256_file
except ImportError as error:  # reported as exit 3 with early_exit.json in main()
    atomic_json = sha256_file = None
    IMPORT_ERROR = f"{type(error).__name__}: {error}"

FORMAT = "molgap-ims-family-serial-v1"
SUMMARY = "serial_summary.json"
REJECTED = 3
PAUSED = 2
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{1,47}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
# Flags this driver sets for every child; an item must not override them.
DRIVER_FLAGS = {"--device", "--recipe", "--input-root", "--output", "--phase", "--no-preflight-receipt", "--source-commit", "--cpus",
                "--training-workers", "--wall-s", "--initial-state", "--accelerator", "--ogb-zip", "--preflight"}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def number(value, *, positive=True) -> bool:
    return type(value) in (int, float) and (value > 0 if positive else True)


def validate_spec(spec, wall_s: float, reserve_s: float) -> list[str]:
    """Shape, types and references of the serial spec. Returns the problems found."""
    if not isinstance(spec, dict) or spec.get("format") != FORMAT or not isinstance(spec.get("items"), list) or not spec["items"]:
        return ["spec: needs format and a non-empty items list"]
    problems, seen = [], []
    for index, item in enumerate(spec["items"]):
        if not isinstance(item, dict):
            problems.append(f"item {index}: not an object")
            continue
        name = item.get("name")
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in seen:
            problems.append(f"item {index}: name missing, unsafe or duplicated")
            continue
        label = f"{name}"
        if not isinstance(item.get("recipe"), str) or Path(item["recipe"]).is_absolute() or ".." in Path(item["recipe"]).parts:
            problems.append(f"{label}: recipe must be a relative path")
        if not isinstance(item.get("recipe_sha256"), str) or not SHA256.fullmatch(item["recipe_sha256"]):
            problems.append(f"{label}: recipe_sha256 must be 64 hex characters")
        if "initial_state" in item or "initial_state_sha256" in item:
            if not (isinstance(item.get("initial_state"), str) and isinstance(item.get("initial_state_sha256"), str)
                    and SHA256.fullmatch(item["initial_state_sha256"]) and not Path(item["initial_state"]).is_absolute()
                    and ".." not in Path(item["initial_state"]).parts):
                problems.append(f"{label}: initial_state needs a relative path and initial_state_sha256 together")
        budget = item.get("budget_s")
        if not number(budget):
            problems.append(f"{label}: budget_s is required and must be a positive number")
        elif budget > wall_s - reserve_s:
            problems.append(f"{label}: budget_s {budget} exceeds wall_s - reserve_s = {wall_s - reserve_s:.0f}; it could never start")
        extra = item.get("extra_args", [])
        if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
            problems.append(f"{label}: extra_args must be a list of strings")
        else:
            clash = sorted({x.split("=")[0] for x in extra} & DRIVER_FLAGS)
            if clash:
                problems.append(f"{label}: extra_args may not set driver flags {clash}")
        gate = item.get("gate", {})
        if not isinstance(gate, dict) or set(gate) - {"dev_delta", "memory"}:
            problems.append(f"{label}: gate may only hold dev_delta and memory")
            gate = {}
        for kind, refs, numbers in (("dev_delta", ("a", "b"), ("max_delta_meV",)), ("memory", ("from",), ("factor", "margin", "max_fraction"))):
            rule = gate.get(kind)
            if rule is None:
                continue
            if not isinstance(rule, dict) or set(rule) - set(refs) - set(numbers):
                problems.append(f"{label}: gate.{kind} has unknown fields")
                continue
            for ref in refs:
                if rule.get(ref) not in seen:
                    problems.append(f"{label}: gate.{kind}.{ref} must name an item listed before it")
            if kind == "dev_delta" and rule.get("a") == rule.get("b"):
                problems.append(f"{label}: gate.dev_delta compares an item with itself")
            for key in numbers:
                if key in rule and not number(rule[key], positive=key != "max_delta_meV"):
                    problems.append(f"{label}: gate.{kind}.{key} must be a number" + ("" if key == "max_delta_meV" else " above 0"))
        seen.append(name)
    return problems


def check_inputs(spec: dict, code_root: Path, input_root: Path, ogb_zip: Path | None, accelerator: str | None) -> list[str]:
    """Cheap identity and presence checks for every item, before any child runs. Returns the problems found."""
    problems: list[str] = []
    records: dict[str, dict] = {}
    cache: dict[str, dict] = {}  # one read per manifest
    if ogb_zip is not None and not ogb_zip.is_file():
        problems.append(f"--ogb-zip file is missing: {ogb_zip}")
    try:
        import torch
        if not torch.cuda.is_available():
            problems.append("CUDA is not available")
        elif accelerator:
            from molgap.training_reproducibility import validate_accelerator_request
            validate_accelerator_request(accelerator, torch.cuda.get_device_name(0), hip_runtime=False)
        from molgap.experiment_family_workflow import tensor_digest
    except (ImportError, ValueError) as error:
        return problems + [f"environment: {type(error).__name__}: {error}"]
    for item in spec["items"]:
        name = item["name"]
        try:
            recipe = code_root / item["recipe"]
            if not recipe.is_file() or sha256_file(recipe) != item["recipe_sha256"]:
                problems.append(f"{name}: recipe missing or its sha256 differs from the spec")
                continue
            record = records[name] = json.loads(recipe.read_text(encoding="utf-8"))
            relative = record["dataset"]["manifest"]
            manifest_path = input_root / relative
            if not manifest_path.is_file() or sha256_file(manifest_path) != record["dataset"]["manifest_sha256"]:
                problems.append(f"{name}: dataset manifest missing or its sha256 differs from the recipe")
                continue
            first_read = relative not in cache
            manifest = cache.setdefault(relative, json.loads(manifest_path.read_text(encoding="utf-8")))
            if manifest.get("status") != "complete":
                problems.append(f"{name}: dataset manifest is not complete")
            train = manifest["roles"]["train"]
            if train["source_idx_stop"] > TAIL_START or train["rows"] != record["train_rows"]:
                problems.append(f"{name}: train rows reach tail_holdout_v1 or differ from the recipe")
            else:
                digest = tensor_digest(torch.arange(train["source_idx_start"], train["source_idx_stop"]), role="source_idx")
                if digest != record["train_role_identity"]:
                    problems.append(f"{name}: train role identity of the manifest differs from the recipe")
            if record["dataset"].get("development"):
                if manifest["roles"]["development"]["source_idx_stop"] > TAIL_START:
                    problems.append(f"{name}: development rows reach tail_holdout_v1")
            if first_read:
                for asset in manifest.get("assets", {}).get("topology", []):
                    shard = input_root / asset["file"]
                    if not shard.is_file() or shard.stat().st_size != asset["bytes"]:
                        problems.append(f"{name}: data file missing or wrong size: {asset['file']}")
            sidecar = record.get("input_sidecar")
            if sidecar and not (input_root / sidecar["manifest"]).is_file():
                problems.append(f"{name}: input sidecar manifest is missing")
            if item.get("initial_state"):
                state = code_root / item["initial_state"]
                if not state.is_file() or sha256_file(state) != item["initial_state_sha256"]:
                    problems.append(f"{name}: initial state missing or its sha256 differs from the spec")
        except (KeyError, OSError, ValueError, TypeError) as error:
            problems.append(f"{name}: {type(error).__name__}: {error}")
    for item in spec["items"]:  # gates compare dev MAE, which needs a development role that selected_eval.json can come from
        gate = item.get("gate") or {}
        for ref in (gate["dev_delta"]["a"], gate["dev_delta"]["b"]) if "dev_delta" in gate else ():
            rec = records.get(ref)
            if rec is not None and not str(rec.get("selection", "")).startswith("best-development"):
                problems.append(f"{item['name']}: gate item {ref} has selection {rec.get('selection')!r}; it writes no selected_eval.json")
        if "dev_delta" in gate:
            ids = {records[ref]["development_role_identity"] for ref in (gate["dev_delta"]["a"], gate["dev_delta"]["b"]) if ref in records}
            if len(ids) > 1:
                problems.append(f"{item['name']}: gate.dev_delta items use different development roles; their MAE is not comparable")
    return problems


def run_child(command: list[str], env: dict, state: dict) -> tuple[int, int | None]:
    """Run one child and return (exit code, peak memory in bytes). The two platforms measure different things.
    Linux: wait4 gives the rusage of this child; ru_maxrss (KiB) is the peak RSS of the single largest process among the
    child and its waited descendants, not their sum. Windows (local rehearsal only): the largest sampled SUM of peak
    working sets over the process tree, which is never smaller. A gate fed by the Windows number is therefore conservative."""
    child = subprocess.Popen(command, env=env)
    state["child_pid"] = child.pid
    try:
        if hasattr(os, "wait4"):
            _, status, usage = os.wait4(child.pid, 0)
            return os.waitstatus_to_exitcode(status), usage.ru_maxrss * 1024
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
    finally:
        state["child_pid"] = None  # reaped: a late SIGTERM must not signal a dead (or reused) pid


def dev_mae(out_dir: Path) -> float | None:
    path = out_dir / "selected_eval.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["dev_mae_eV"]) if path.is_file() else None


def is_complete(out_dir: Path) -> bool:
    path = out_dir / "completion_manifest.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") == "COMPLETE"


def gate_decision(item: dict, tasks: dict, out_dirs: dict, limit: int) -> tuple[str, str] | None:
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
        budget = rule.get("max_fraction", 0.5) * limit
        if need > budget:
            return "SKIPPED_FOR_MEMORY", (f"predicted peak {need / 2**30:.1f} GiB ({rule['from']} {peak / 2**30:.2f} GiB x "
                                          f"{rule.get('factor', 5)} x {rule.get('margin', 1.3)}) exceeds {rule.get('max_fraction', 0.5)} x "
                                          f"{limit / 2**30:.1f} GiB (PBS mem request)")
    return None


def reject(root: Path, problems: list[str]) -> int:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"status": "REJECTED", "utc": now(), "problems": problems}
    if atomic_json is not None:
        atomic_json(root / "early_exit.json", payload)
    else:  # molgap itself did not import; there is no shared helper to use
        (root / "early_exit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("[driver] REJECTED before any child ran:\n  " + "\n  ".join(problems), flush=True)
    return REJECTED


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
    ap.add_argument("--wall-s", type=float, required=True, help="seconds this job may run (the launcher passes walltime - 120); the clock starts at driver start")
    ap.add_argument("--memory-limit-bytes", type=int, required=True, help="the PBS mem= request in bytes (the launcher passes it)")
    ap.add_argument("--reserve-s", type=float, default=1800.0, help="kept free at the end for the last checkpoint")
    ap.add_argument("--max-attempts", type=int, default=2, help="a FAILED item is retried on rerun until it has this many attempts")
    ap.add_argument("--continue-after-pause", action="store_true", help="local rehearsal only: a paused item does not stop the driver")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    root, code_root, input_root = a.output_root.resolve(), a.code_root.resolve(), a.input_root.resolve()
    if atomic_json is None:
        return reject(root, [f"environment: {IMPORT_ERROR}"])
    try:
        spec = json.loads(a.spec.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return reject(root, [f"spec: cannot read: {error}"])
    problems = validate_spec(spec, a.wall_s, a.reserve_s) if a.memory_limit_bytes > 0 else ["--memory-limit-bytes must be positive"]
    if not problems:
        problems = check_inputs(spec, code_root, input_root, a.ogb_zip, a.accelerator)
    if problems:
        return reject(root, problems)
    root.mkdir(parents=True, exist_ok=True)
    names = [item["name"] for item in spec["items"]]
    out_dirs = {name: root / name for name in names}
    summary_path = root / SUMMARY
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {
        "format": "molgap-ims-family-serial-summary-v1", "order": names, "driver_starts": [],
        "items": {n: {"status": "NOT_STARTED", "attempts": []} for n in names}}
    summary.update(wall_s=a.wall_s, reserve_s=a.reserve_s, spec_sha256=sha256_file(a.spec), source_commit=a.source_commit,
                   memory_limit_bytes=a.memory_limit_bytes)
    summary["driver_starts"].append(now())
    tasks = summary["items"]
    for task in tasks.values():
        if task["status"] == "RUNNING":  # a hard kill left it; it resumes from its own checkpoint
            task["status"] = "NOT_STARTED"
    state = {"stop": False, "child_pid": None}

    def write() -> None:
        atomic_json(summary_path, summary)

    def on_term(_signum, _frame) -> None:
        state["stop"] = True
        pid = state["child_pid"]
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)  # the child checkpoints and exits 2
            except (ProcessLookupError, OSError):
                pass  # it ended between the check and the signal

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
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
        if task["status"] == "FAILED" and len(task["attempts"]) >= a.max_attempts:
            task["reason"] = f"failed {len(task['attempts'])} times; attempt limit {a.max_attempts} reached, not retried"
            continue
        skip = gate_decision(item, tasks, out_dirs, a.memory_limit_bytes)
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
        code, peak = run_child(command, dict(os.environ), state)
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
