"""Local rehearsal of the IMS single-job serial chain. Real flow, one pass per recipe, no IMS access.

Run with the venv python. Windows, RTX-class GPU, the 100K fixed root and the J1/J2 recipes of the DeMol-CIN replication
checkout (read only; it holds demol_cin.py). The recipes are copied with one training pass (781 steps) so items really finish.
Not rehearsed here: the Linux memory probes, wait4 peak memory, a real PBS queue, and the A100 accelerator check.
Writes family_serial_rehearsal_record.json beside this file and keeps transcripts under --work (outside Git).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "family_serial_run.py"
LIMITS = {"handles": 20000, "ws_bytes": 25 * 2**30}  # the guard of the earlier Windows dry run
results: list[dict] = []


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(name: str, passed: bool, **facts) -> None:
    results.append({"check": name, "passed": bool(passed), **facts})
    print(f"[rehearsal] {'PASS' if passed else 'FAIL'} {name} {json.dumps(facts, default=str)[:400]}", flush=True)


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)


def run_driver(args: list[str], log: Path, env: dict, *, kill_when=None, watch_summary: Path | None = None, timeout=3600) -> dict:
    """Run the driver with a handle/working-set guard (DeMol must not freeze the desktop) and optional kill trigger.
    Returns exit code, guard peaks, and the count of summary reads that failed to parse (atomic-write check)."""
    out = {"exit": None, "peak_handles": 0, "peak_ws_gib": 0.0, "guard_tripped": False, "killed": False, "bad_summary_reads": 0, "summary_reads": 0}
    with log.open("w", encoding="utf-8") as stream:
        proc = subprocess.Popen([sys.executable, "-u", str(DRIVER), *args], env=env, stdout=stream, stderr=subprocess.STDOUT)
        stop = threading.Event()

        def poll_summary() -> None:
            while not stop.is_set():
                if watch_summary is not None and watch_summary.is_file():
                    try:
                        json.loads(watch_summary.read_text(encoding="utf-8"))
                        out["summary_reads"] += 1
                    except FileNotFoundError:
                        pass
                    except (json.JSONDecodeError, PermissionError, OSError):
                        out["bad_summary_reads"] += 1
                time.sleep(0.02)

        thread = threading.Thread(target=poll_summary, daemon=True)
        thread.start()
        started = time.monotonic()
        while proc.poll() is None:
            try:
                tree = [psutil.Process(proc.pid)]
                tree += tree[0].children(recursive=True)
                handles = sum(p.num_handles() for p in tree)
                ws = sum(p.memory_info().rss for p in tree)
                out["peak_handles"], out["peak_ws_gib"] = max(out["peak_handles"], handles), max(out["peak_ws_gib"], round(ws / 2**30, 2))
                if handles > LIMITS["handles"] or ws > LIMITS["ws_bytes"]:
                    out["guard_tripped"] = True
                    kill_tree(proc.pid)
            except psutil.Error:
                pass
            if kill_when is not None and not out["killed"] and kill_when():
                out["killed"] = True
                kill_tree(proc.pid)
            if time.monotonic() - started > timeout:
                kill_tree(proc.pid)
            time.sleep(0.5)
        stop.set()
        thread.join()
        out["exit"] = proc.returncode
    return out


def summary_of(root: Path) -> dict:
    return json.loads((root / "serial_summary.json").read_text(encoding="utf-8"))


def prepare(work: Path, repl: Path) -> dict:
    """One-pass copies of J1 (DeMol-CIN) and J2 (GPT-T pair-update-norm) under work/code, plus the serial spec."""
    exp = repl / "experiments/pcqm_demol_cin_s43_100k_ims_20261120"
    code = work / "code"
    (code / "recipes").mkdir(parents=True, exist_ok=True)
    for key, src in (("demol", "j1_demol_cin"), ("gptt", "j2_gptrans_pun")):
        r = json.loads((exp / src / "recipe.json").read_text(encoding="utf-8"))
        r["optimizer_steps"] = 781  # 100000 // 128; the schedule keeps a cosine tail, so it spans two units
        r["schedule"].update(reference_epochs=2, total_units=2, warmup_units=1)
        r["acceptance_requirements"].update(optimizer_steps=781, completed_passes=1, sample_presentations=781 * 128, next_batch=0)
        (code / "recipes" / f"{key}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    (code / "init").mkdir(exist_ok=True)
    (code / "init/s43.pt").write_bytes((exp / "init/s43.pt").read_bytes())
    return {"demol": code / "recipes/demol.json", "gptt": code / "recipes/gptt.json", "init": code / "init/s43.pt"}


def item(name, kind, files, **extra) -> dict:
    recipe = files[kind]
    entry = {"name": name, "recipe": recipe.relative_to(files["code"]).as_posix(), "recipe_sha256": sha(recipe), "budget_s": 600}
    if kind == "gptt":
        entry.update(initial_state="init/s43.pt", initial_state_sha256=sha(files["init"]))
    entry.update(extra)
    return entry


def write_spec(path: Path, items: list[dict]) -> Path:
    path.write_text(json.dumps({"format": "molgap-ims-family-serial-v1", "items": items}, indent=1), encoding="utf-8")
    return path


def driver_args(spec: Path, files: dict, root: Path, fixed: Path, **over) -> list[str]:
    args = ["--spec", str(spec), "--code-root", str(files["code"]), "--input-root", str(fixed), "--output-root", str(root),
            "--source-commit", "0" * 40, "--cpus", "4", "--training-workers", "0", "--wall-s", str(over.pop("wall_s", 36000)),
            "--reserve-s", str(over.pop("reserve_s", 60)), "--memory-limit-bytes", str(over.pop("memory_limit_bytes", 2**40))]
    for key, value in over.items():
        args += [f"--{key.replace('_', '-')}"] + ([] if value is True else [str(value)])
    return args


def rehearse_early_exit(work, files, fixed, env) -> None:
    """Break one input at a time; the driver must exit 3 and name the problem before any child starts."""
    base = [item("gptt-100k", "gptt", files)]
    cases = {}
    cases["recipe sha differs from spec"] = write_spec(work / "s_badsha.json", [{**base[0], "recipe_sha256": "0" * 64}])
    cases["initial state sha differs"] = write_spec(work / "s_badinit.json", [{**base[0], "initial_state_sha256": "1" * 64}])
    tampered = json.loads(files["gptt"].read_text(encoding="utf-8"))
    tampered["train_role_identity"] = "f" * 64
    bad_recipe = files["code"] / "recipes/gptt_bad_role.json"
    bad_recipe.write_text(json.dumps(tampered), encoding="utf-8")
    cases["train role identity differs"] = write_spec(work / "s_badrole.json", [{**base[0], "recipe": "recipes/gptt_bad_role.json", "recipe_sha256": sha(bad_recipe)}])
    good_spec = write_spec(work / "s_good.json", base)
    empty = work / "empty_root"
    empty.mkdir(exist_ok=True)
    partial = work / "partial_root/subsets/ogb-train-100k"  # the manifest only; every data shard is missing
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "manifest.json").write_bytes((fixed / "subsets/ogb-train-100k/manifest.json").read_bytes())
    specs_roots = [(name, spec, fixed) for name, spec in cases.items()]
    specs_roots += [("dataset manifest missing", good_spec, empty), ("data shards missing", good_spec, work / "partial_root")]
    for name, spec, root in specs_roots:
        out = work / "early" / name.replace(" ", "_")
        started = time.monotonic()
        res = run_driver(driver_args(spec, files, out, root), work / f"early_{name.replace(' ', '_')}.log", env)
        early = out / "early_exit.json"
        problems = json.loads(early.read_text(encoding="utf-8"))["problems"] if early.is_file() else []
        no_child = not (out / "gptt-100k").exists()
        record(f"exit 3 on broken input: {name}", res["exit"] == 3 and bool(problems) and no_child,
               exit=res["exit"], problem=problems[:1], child_started=not no_child, seconds=round(time.monotonic() - started, 1))
    out = work / "early" / "unbroken_check_only"
    # control: the same unbroken spec passes the check (it would start a child, so stop it after the check by a tiny wall)
    small = write_spec(work / "s_small_budget.json", [{**base[0], "budget_s": 9}])  # budget must fit wall - reserve, so a tiny wall needs tiny budgets
    res = run_driver(driver_args(small, files, out, fixed, wall_s=10, reserve_s=0), work / "early_control.log", env)
    summary = summary_of(out)["items"]["gptt-100k"]["status"] if (out / "serial_summary.json").is_file() else None
    record("unbroken input is accepted (control, tiny wall gives SKIPPED_FOR_TIME)", res["exit"] == 2 and summary == "SKIPPED_FOR_TIME", exit=res["exit"], status=summary)


def rehearse_chain(work, files, fixed, env) -> Path:
    """gptt fail-first chain with kill, resume, gates, time skip. Returns the output root."""
    root = work / "chain"
    items = [
        # DeMol is standalone and its recipe binds another frozen state: the kernel rejects the GPT-T state after loading data
        item("init-fail", "demol", files, initial_state="init/s43.pt", initial_state_sha256=sha(files["init"])),
        item("gptt-100k", "gptt", files),
        item("demol-100k", "demol", files),
        item("demol-500k-pass", "demol", files, gate={"dev_delta": {"a": "demol-100k", "b": "gptt-100k", "max_delta_meV": 1e6},
                                                      "memory": {"from": "demol-100k", "max_fraction": 0.5}}),
        item("demol-500k-gate-skip", "demol", files, gate={"dev_delta": {"a": "demol-100k", "b": "gptt-100k", "max_delta_meV": -1e6}}),
        item("demol-500k-mem-skip", "demol", files, gate={"memory": {"from": "demol-100k", "max_fraction": 1e-9}}),
        item("demol-500k-time-skip", "demol", files, budget_s=35940),  # the largest budget the spec check allows (wall 36000 - reserve 60); any elapsed time makes it skip
        item("demol-500k-no-input", "demol", files, gate={"dev_delta": {"a": "init-fail", "b": "gptt-100k"}}),
    ]
    spec = write_spec(work / "chain_spec.json", items)
    args = driver_args(spec, files, root, fixed)
    summary_path = root / "serial_summary.json"
    ckpt = root / "demol-100k/last_checkpoint.pt"
    # Run A: kill the whole tree once demol-100k has written a checkpoint (after step 128, before it can finish).
    res_a = run_driver(args, work / "chain_a.log", env, watch_summary=summary_path, kill_when=lambda: ckpt.is_file() and not (root / "demol-100k/completion_manifest.json").is_file())
    a = summary_of(root)["items"]
    gptt_done = (root / "gptt-100k/completion_manifest.json")
    stamp = gptt_done.stat().st_mtime_ns if gptt_done.is_file() else None
    record("kill mid-run leaves a parseable summary; earlier items done, killed item RUNNING",
           res_a["killed"] and a["init-fail"]["status"] == "FAILED" and a["gptt-100k"]["status"] == "COMPLETE" and a["demol-100k"]["status"] == "RUNNING",
           statuses={k: v["status"] for k, v in a.items()}, bad_summary_reads=res_a["bad_summary_reads"], summary_reads=res_a["summary_reads"])
    # Run B: the same command resumes.
    res_b = run_driver(args, work / "chain_b.log", env, watch_summary=summary_path)
    b = summary_of(root)["items"]
    log_b = (work / "chain_b.log").read_text(encoding="utf-8", errors="replace")
    resumed = [ln for ln in log_b.splitlines() if ln.startswith("resumed at step")]
    record("resume: finished item not redone", gptt_done.stat().st_mtime_ns == stamp and len(b["gptt-100k"]["attempts"]) == 1 and "[driver] gptt-100k start" not in log_b,
           gptt_attempts=len(b["gptt-100k"]["attempts"]))
    record("resume: killed item continues from its checkpoint and completes", bool(resumed) and int(resumed[0].split()[-1]) > 0 and b["demol-100k"]["status"] == "COMPLETE",
           resumed=resumed[:1], status=b["demol-100k"]["status"])
    record("failed item does not block the next item (and is retried on rerun)", b["init-fail"]["status"] == "FAILED" and len(b["init-fail"]["attempts"]) == 2 and b["gptt-100k"]["status"] == "COMPLETE",
           accel_fail_attempts=len(b["init-fail"]["attempts"]), reason=b["init-fail"]["reason"])
    delta = (b["demol-100k"]["dev_mae_eV"] - b["gptt-100k"]["dev_mae_eV"]) * 1000
    record("gate does not skip when the rule passes (delta below the line, memory under budget)", b["demol-500k-pass"]["status"] == "COMPLETE",
           delta_meV=round(delta, 3), status=b["demol-500k-pass"]["status"], peak_rss_gib=round((b["demol-100k"].get("peak_rss_bytes") or 0) / 2**30, 2))
    record("gate skips: dev MAE delta above the line -> SKIPPED_BY_GATE", b["demol-500k-gate-skip"]["status"] == "SKIPPED_BY_GATE" and not (root / "demol-500k-gate-skip").exists(),
           reason=b["demol-500k-gate-skip"]["reason"])
    record("gate skips: predicted memory above budget -> SKIPPED_FOR_MEMORY", b["demol-500k-mem-skip"]["status"] == "SKIPPED_FOR_MEMORY" and not (root / "demol-500k-mem-skip").exists(),
           reason=b["demol-500k-mem-skip"]["reason"])
    record("gate fails closed when an input item failed -> SKIPPED_BY_GATE", b["demol-500k-no-input"]["status"] == "SKIPPED_BY_GATE", reason=b["demol-500k-no-input"]["reason"])
    record("budget longer than wall left -> SKIPPED_FOR_TIME, no child", b["demol-500k-time-skip"]["status"] == "SKIPPED_FOR_TIME" and not (root / "demol-500k-time-skip").exists(),
           reason=b["demol-500k-time-skip"]["reason"])
    record("summary writes are atomic (no unparseable read, no temp file left)", res_a["bad_summary_reads"] == 0 and res_b["bad_summary_reads"] == 0 and not list(root.glob("*.tmp")),
           reads=res_a["summary_reads"] + res_b["summary_reads"])
    record("driver exit code 2 when an item could resume (SKIPPED_FOR_TIME)", res_b["exit"] == 2, exit=res_b["exit"])
    record("guard did not trip (handles/working set), DeMol ran with training workers 0", not (res_a["guard_tripped"] or res_b["guard_tripped"]),
           peak_handles=max(res_a["peak_handles"], res_b["peak_handles"]), peak_ws_gib=max(res_a["peak_ws_gib"], res_b["peak_ws_gib"]))
    res_c = run_driver(args, work / "chain_c.log", env)
    c = summary_of(root)["items"]
    log_c = (work / "chain_c.log").read_text(encoding="utf-8", errors="replace")
    record("failed item stops being retried at the attempt limit (default 2); finished items and gates unchanged on a third run",
           len(c["init-fail"]["attempts"]) == 2 and "attempt limit" in c["init-fail"]["reason"] and "[driver] init-fail start" not in log_c
           and "start" not in "".join(ln for ln in log_c.splitlines() if ln.startswith("[driver]") and "start," in ln),
           exit=res_c["exit"], reason=c["init-fail"]["reason"])
    prov = json.loads((root / "demol-100k/run_provenance.json").read_text(encoding="utf-8"))
    record("receipt waiver is written to run_provenance.json", prov.get("preflight_receipt") == "waived", preflight_receipt=prov.get("preflight_receipt"))
    return root


def rehearse_launcher(work, files, fixed, repl, env) -> None:
    """Chain mode of launcher.py through stage, fake jsub, worker (really runs the driver) and resume."""
    spec_mod = importlib.util.spec_from_file_location("launcher", HERE / "v4_submit/launcher.py")
    launcher = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(launcher)
    infra = work / "infra"
    shutil.rmtree(infra, ignore_errors=True)  # scratch inside --work; a rerun starts clean
    infra.mkdir()
    launcher.BOUNDARY, launcher.INFRA_ROOT = work, infra
    datasets = json.loads((HERE / "v4_submit/dataset_registry.json").read_text(encoding="utf-8"))
    datasets["fixed_root"] = str(fixed)
    runtimes = json.loads((HERE / "v4_submit/runtime_registry.json").read_text(encoding="utf-8"))
    runtimes["runtimes"]["ims-a100-torch271-v1"].update(python=sys.executable, pythonpath=[str(repl / "src")])
    for name, payload in (("dataset_registry.json", datasets), ("runtime_registry.json", runtimes)):
        (infra / name).write_text(json.dumps(payload), encoding="utf-8")
    launcher.validate_registries = lambda base: (datasets, runtimes)  # the lustre acceptance file does not exist here
    # Source tarball with only what the chain needs; contract = the serial spec.
    items = [item("gptt-20-steps", "gptt", files, extra_args=["--max-steps", "20"])]
    serial = write_spec(work / "launcher_serial_spec.json", items)
    code_files = {"platforms/ims/family_serial_run.py": DRIVER, "platforms/ims/run_family_training.py": HERE / "run_family_training.py",
                  "platforms/ims/family_recipe_run.py": HERE / "family_recipe_run.py", "contract/serial_spec.json": serial,
                  "recipes/gptt.json": files["gptt"], "init/s43.pt": files["init"]}
    inbox = infra / "inbox" / "rehearsal-chain-20261120"
    inbox.mkdir(parents=True, exist_ok=True)
    with tarfile.open(inbox / "source.tar.gz", "w:gz") as tar:
        for rel, src in code_files.items():
            tar.add(src, arcname=rel)
    (inbox / "source_files.json").write_text(json.dumps({"format": "molgap-source-files-v1",
        "files": [{"path": rel, "sha256": sha(src)} for rel, src in code_files.items()]}), encoding="utf-8")
    train_args = ["--spec", "{code_root}/contract/serial_spec.json", "--code-root", "{code_root}", "--input-root", "{dataset_root}", "--output-root", "{training_output}",
                  "--source-commit", "{source_commit}", "--cpus", "8", "--training-workers", "0", "--wall-s", "{train_wall_s}", "--memory-limit-bytes", "{mem_bytes}", "--reserve-s", "60"]
    spec = {"format": "molgap-ims-v4-run-v1", "mode": "serial_chain", "run_id": "rehearsal-chain-20261120", "model_id": "serial-chain",
            "dataset_ids": ["ogb-train-100k", "ogb-train-500k-scnet-v1"], "runtime_id": "ims-a100-torch271-v1",
            "source": {"commit": "0" * 40, "archive_sha256": sha(inbox / "source.tar.gz")},
            "contract": {"path": "contract/serial_spec.json", "sha256": sha(serial), "physical_batch_per_device": 128, "precision": "fp32", "tf32": False, "tail_batch_policy": "drop_last"},
            "resources": {"queue": "H", "ncpus": 8, "ompthreads": 4, "ngpus": 1, "train_walltime": "08:00:00"},
            "stages": {"train": {"entrypoint": "platforms/ims/family_serial_run.py", "args": train_args, "required_outputs": ["training/serial_summary.json"]}},
            "sealed_roles": {"official_validation_read": False, "test_dev_read": False, "test_challenge_read": False}}
    (inbox / "run_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    # 1. spec validation: a chain with a preflight stage, or with dataset_id, is refused; the single-recipe example stays valid.
    def refused(mutate) -> bool:
        bad = json.loads(json.dumps(spec))
        mutate(bad)
        try:
            launcher.validate_spec(bad, datasets, runtimes)
        except RuntimeError:
            return True
        return False
    launcher.validate_spec(spec, datasets, runtimes)
    record("launcher: chain spec validates; preflight stage / dataset_id / unknown dataset / unknown mode are refused",
           refused(lambda s: s["stages"].update(preflight=s["stages"]["train"])) and refused(lambda s: s.update(dataset_id="ogb-train-100k"))
           and refused(lambda s: s.update(dataset_ids=["nope"])) and refused(lambda s: s.update(mode="other"))
           and refused(lambda s: s["resources"].update(ncpus=32)) and refused(lambda s: s["resources"].update(ngpus=4, ncpus=48))
           and refused(lambda s: s["stages"]["train"]["args"].__setitem__(s["stages"]["train"]["args"].index("--cpus") + 1, "16"))
           and refused(lambda s: s.update(run_id="1chain-bad-name")) and refused(lambda s: s["stages"]["train"]["args"].__setitem__(s["stages"]["train"]["args"].index("--wall-s") + 1, "86000"))
           and refused(lambda s: s["stages"]["train"]["args"].__setitem__(s["stages"]["train"]["args"].index("--memory-limit-bytes") + 1, "1")))
    single = json.loads((HERE / "v4_submit/run_spec.example.json").read_text(encoding="utf-8"))
    single.update(run_id="single-recipe-example")
    single["source"] = {"commit": "0" * 40, "archive_sha256": "0" * 64}
    single["contract"]["sha256"] = "0" * 64
    launcher.validate_spec(single, datasets, runtimes)
    bad_single = json.loads(json.dumps(single))
    bad_single["stages"]["train"]["args"] = bad_single["stages"]["train"]["args"] + ["--memory-limit-bytes", "{mem_bytes}"]
    try:
        launcher.validate_spec(bad_single, datasets, runtimes)
        single_refused = False
    except RuntimeError:
        single_refused = True
    record("launcher: the single-recipe example spec still validates, stages are preflight + train; {mem_bytes} in a single-recipe spec is refused",
           launcher.stage_names(single) == ("preflight", "train") and single_refused)
    # 2. PBS text: render with a fake run root and let bash parse it.
    pbs = launcher.render_pbs(Path("/lustre/home/users/sm2/chou/molgap-v4-submit/runs/rehearsal-chain-20261120"), "train", spec, runtimes["runtimes"]["ims-a100-torch271-v1"])
    (work / "train.pbs").write_text(pbs, encoding="utf-8", newline="\n")
    bash_exe = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
    bash = subprocess.run([bash_exe, "-n", str(work / "train.pbs").replace("\\", "/")], capture_output=True, text=True)
    record("PBS script for the chain parses (bash -n)", bash.returncode == 0 and "walltime=08:00:00" in pbs and "mem=" not in pbs and "ncpus=8:" in pbs, select_line=[ln for ln in pbs.splitlines() if "select" in ln], stderr=bash.stderr[:200])
    # 3. maintenance guard: a refusal happens before staging and leaves no run directory and no jsub.
    calls: list[list[str]] = []
    real_run, real_guard = subprocess.run, launcher.maintenance_guard

    def fake_run(command, **kw):
        if command[0] == "jsub":
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="12345.ccfep\n", stderr="")
        return real_run(command, **kw)

    launcher.subprocess.run = fake_run
    from datetime import datetime
    from zoneinfo import ZoneInfo
    try:
        real_guard(3600, now=datetime(2026, 12, 6, 10, 0, tzinfo=ZoneInfo("Asia/Tokyo")))  # Sunday before the first Monday (2026-12-07)
        first_sunday = False
    except RuntimeError:
        first_sunday = True
    launcher.maintenance_guard = lambda seconds: (_ for _ in ()).throw(RuntimeError("IMS maintenance guard: rehearsal refusal"))
    try:
        launcher.submit_run(inbox, infra)
        refused_run = False
    except RuntimeError:
        refused_run = True
    record("maintenance guard: Sunday before first Monday is refused by the real guard; chain submit refuses before staging (no run dir, no jsub)",
           first_sunday and refused_run and not (infra / "runs/rehearsal-chain-20261120").exists() and not calls, first_sunday=first_sunday)
    seen = []
    launcher.maintenance_guard = lambda seconds: seen.append(seconds)
    submission = launcher.submit_run(inbox, infra)
    run_root = infra / "runs/rehearsal-chain-20261120"
    record("chain submit: exactly one jsub, no depend, no preflight job/file/dir; guard saw the train walltime",
           len(calls) == 1 and "-W" not in calls[0] and calls[0][-1].endswith("train.pbs") and submission["status"] == "submitted" and "preflight_job" not in submission
           and not (run_root / "preflight.pbs").exists() and not (run_root / "output/preflight").exists() and seen == [8 * 3600], jsub=calls[0], status=submission["status"])
    # 4. the worker runs the driver for real (20 steps -> paused, exit 2); stage status paused; resume path.
    worker_env = {"PYTHONPATH": str(repl / "src"), "CUDA_VISIBLE_DEVICES": "0"}
    os.environ.update(worker_env)
    try:
        launcher.worker(run_root, "train")
        worker_exit = 0
    except SystemExit as stop:
        worker_exit = stop.code
    status = json.loads((run_root / "output/train/stage_status.json").read_text(encoding="utf-8"))["status"]
    full = json.loads((run_root / "output/training/serial_summary.json").read_text(encoding="utf-8"))
    summary = full["items"]["gptt-20-steps"]["status"]
    record("launcher injects the wall and memory numbers (wall = walltime - 120 s, memory = ncpus x 1.875e9 bytes; both as the driver saw them)",
           full["wall_s"] == 8 * 3600 - 120 and full["memory_limit_bytes"] == 8 * 1_875_000_000, wall_s=full["wall_s"], memory_limit_bytes=full["memory_limit_bytes"])
    record("worker -> driver -> child end to end: paused exits 2, stage status paused, item PAUSED", worker_exit == 2 and status == "paused" and summary == "PAUSED",
           worker_exit=worker_exit, stage_status=status, item=summary)
    calls.clear()
    resumed = launcher.submit_run(inbox, infra)
    record("same command after a pause resubmits train.pbs only (one jsub, no depend)", len(calls) == 1 and "-W" not in calls[0] and calls[0][-1].endswith("train.pbs") and resumed["status"] == "submitted",
           jsub=calls[0] if calls else None)
    single = json.loads(json.dumps(spec))
    single.pop("mode"), single.pop("dataset_ids")
    single["dataset_id"] = "ogb-train-100k"
    (inbox / "run_spec.json").write_text(json.dumps(single), encoding="utf-8")
    atomic_status = run_root / "output/train/stage_status.json"
    atomic_status.write_text(json.dumps({"status": "paused", "stage": "train"}), encoding="utf-8")
    calls.clear()
    try:
        launcher.submit_run(inbox, infra)
        conflict = False
    except RuntimeError as error:
        conflict = "mode" in str(error)
    record("a single-recipe spec reusing a chain run id is refused, not turned into a resume", conflict and not calls)
    # 5. single-recipe mode is unchanged: two jsub calls, the second held by afterok on the first.
    inbox2 = infra / "inbox" / "rehearsal-single-20261120"
    inbox2.mkdir(parents=True)
    for name in ("source.tar.gz", "source_files.json"):
        shutil.copy2(inbox / name, inbox2 / name)
    alone = json.loads(json.dumps(spec))
    alone.pop("mode"), alone.pop("dataset_ids")
    alone.update(run_id="rehearsal-single-20261120", dataset_id="ogb-train-100k")
    alone["resources"]["preflight_walltime"] = "00:45:00"
    plain = {**alone["stages"]["train"], "args": ["--spec", "x"]}  # chain placeholders are refused outside chain mode
    alone["stages"] = {"preflight": plain, "train": plain}
    (inbox2 / "run_spec.json").write_text(json.dumps(alone), encoding="utf-8")
    calls.clear()
    out = launcher.submit_run(inbox2, infra)
    record("single-recipe mode unchanged: preflight job, then train job with depend=afterok",
           len(calls) == 2 and calls[0][-1].endswith("preflight.pbs") and "-W" not in calls[0] and calls[1][4] == "rehearsal-singl"
           and "depend=afterok:12345.ccfep" in calls[1] and out.get("preflight_job") == "12345.ccfep", jsub=calls)
    launcher.subprocess.run, launcher.maintenance_guard = real_run, real_guard


def rehearse_hardening(work, files, fixed, repl, env) -> None:
    """Spec validation, tiny wall, no-ogb environment and the maintenance guard on fixed dates."""
    items = [item("gptt-100k", "gptt", files), item("demol-100k", "demol", files)]
    # (a) PYTHONNOUSERSITE=1 and ogb blocked: the start-of-job check must not need ogb.
    spec = write_spec(work / "h_spec.json", items)
    code = ("import sys, json; sys.modules['ogb'] = None; sys.path.insert(0, sys.argv[1]); import family_serial_run as m; from pathlib import Path\n"
            "spec = json.load(open(sys.argv[2])); print('PROBLEMS', json.dumps(m.check_inputs(spec, Path(sys.argv[3]), Path(sys.argv[4]), None, None)))")
    res = subprocess.run([sys.executable, "-c", code, str(HERE), str(spec), str(files["code"]), str(fixed)], env={**env, "PYTHONNOUSERSITE": "1"}, capture_output=True, text=True)
    line = next((ln for ln in res.stdout.splitlines() if ln.startswith("PROBLEMS")), "")
    record("start-of-job check with PYTHONNOUSERSITE=1 and ogb blocked: no ImportError, no problems", res.returncode == 0 and line == "PROBLEMS []",
           returncode=res.returncode, output=line or res.stderr[-300:])
    # (b) tiny wall. A budget longer than the wall is rejected at the start (exit 3); budgets that fit but cannot finish are skipped (exit 2).
    out = work / "h_tiny_reject"
    res = run_driver(driver_args(spec, files, out, fixed, wall_s=10, reserve_s=0), work / "h_tiny_reject.log", env)
    record("tiny --wall-s with the real budgets (600 s): rejected at start, exit 3, budget named", res["exit"] == 3 and "budget_s" in (out / "early_exit.json").read_text(encoding="utf-8"), exit=res["exit"])
    tiny = write_spec(work / "h_tiny.json", [{**i, "budget_s": 9} for i in items])
    out = work / "h_tiny_skip"
    res = run_driver(driver_args(tiny, files, out, fixed, wall_s=10, reserve_s=0), work / "h_tiny_skip.log", env)
    states = {k: v["status"] for k, v in summary_of(out)["items"].items()}
    record("tiny --wall-s with fitting budgets: exit 2, every item SKIPPED_FOR_TIME, no child started",
           res["exit"] == 2 and set(states.values()) == {"SKIPPED_FOR_TIME"} and not any((out / n).exists() for n in states), exit=res["exit"], states=states)
    # (c) bad specs
    good = items[0]
    bad = {"missing budget_s": {k: v for k, v in good.items() if k != "budget_s"},
           "gate names a missing item": {**good, "gate": {"dev_delta": {"a": "nope", "b": "gptt-100k"}}},
           "gate names a later item": {**good, "gate": {"memory": {"from": "demol-100k"}}},
           "extra_args sets --wall-s": {**good, "extra_args": ["--wall-s", "999999"]},
           "extra_args sets --phase=": {**good, "extra_args": ["--phase=preflight"]},
           "budget above wall - reserve": {**good, "budget_s": 10**6},
           "wrong type": {**good, "budget_s": "long"}}
    for name, bad_item in bad.items():
        path = write_spec(work / "h_bad.json", [bad_item, items[1]])
        out = work / "h_bad_out" / name.replace(" ", "_")
        res = run_driver(driver_args(path, files, out, fixed), work / "h_bad.log", env)
        problem = json.loads((out / "early_exit.json").read_text(encoding="utf-8"))["problems"][0] if (out / "early_exit.json").is_file() else None
        record(f"bad spec exits 3: {name}", res["exit"] == 3 and problem is not None and not (out / "gptt-100k").exists(), exit=res["exit"], problem=problem)
    # (d) maintenance guard on fixed dates (JST)
    from datetime import datetime
    from zoneinfo import ZoneInfo
    spec_mod = importlib.util.spec_from_file_location("launcher_dates", HERE / "v4_submit/launcher.py")
    launcher = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(launcher)
    expect = {(2026, 12, 6, 8, 59): False, (2026, 12, 6, 9, 0): True, (2027, 1, 31, 8, 59): False, (2027, 1, 31, 10, 0): True,
              (2027, 2, 1, 10, 0): False, (2027, 2, 7, 10, 0): False}
    seen = {}
    for when, refuse in expect.items():
        try:
            launcher.maintenance_guard(3600, now=datetime(*when, tzinfo=ZoneInfo("Asia/Tokyo")))
            seen[str(when)] = False
        except RuntimeError:
            seen[str(when)] = True
    record("maintenance guard on fixed dates (train 1 h): window = 24 h before first Monday 09:00 JST; Monday 09:00 onward is open", seen == {str(k): v for k, v in expect.items()},
           refused={k: v for k, v in seen.items()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("D:/molgap-artifacts/ims_serial_rehearsal_20261120"))
    ap.add_argument("--repl", type=Path, default=Path("D:/molgap-worktrees/demol-cin-ims-replication-20261120"))
    ap.add_argument("--fixed-root", type=Path, default=Path("D:/molgap-artifacts/ims_weekend_20261120/local_fixed_root"))
    ap.add_argument("--only", choices=("early", "chain", "launcher", "hardening"), action="append")
    a = ap.parse_args()
    work = a.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(a.repl / "src"), CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
    files = {**prepare(work, a.repl), "code": work / "code"}
    only = a.only or ["early", "hardening", "chain", "launcher"]
    if "early" in only:
        rehearse_early_exit(work, files, a.fixed_root, env)
    if "hardening" in only:
        rehearse_hardening(work, files, a.fixed_root, a.repl, env)
    if "chain" in only:
        rehearse_chain(work, files, a.fixed_root, env)
    if "launcher" in only:
        rehearse_launcher(work, files, a.fixed_root, a.repl, env)
    path = HERE / "family_serial_rehearsal_record.json"  # parts may run separately; keep earlier results of the other parts
    old = json.loads(path.read_text(encoding="utf-8"))["results"] if path.is_file() else []
    merged = [r for r in old if r["check"] not in {x["check"] for x in results}] + results
    path.write_text(json.dumps({"date": time.strftime("%Y-%m-%d"), "results": merged}, indent=1) + "\n", encoding="utf-8")
    failed = [r["check"] for r in results if not r["passed"]]
    print("[rehearsal] FAILED:" if failed else "[rehearsal] all passed", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
