"""Local rehearsal of the DeMol-CIN 500K serial chain on the REAL spec. Windows, one RTX GPU, no IMS access.

Parts (run one at a time, each under a handle and working-set guard; DeMol always with --training-workers 0):
  early     start-of-job check on the real spec (ogb blocked), then the exit-3 matrix (one broken input each)
  launcher  real run_spec through launcher validation, PBS text, argument expansion, and an end-to-end worker run
  chain     short copies of the five recipes (one pass each) driven by the real item names, gates and extra_args
Differences from IMS: no --accelerator A100 and no --ogb-zip, one pass per recipe, local 100K/500K mirror root.
It reuses the helpers of platforms/ims/family_serial_rehearsal.py and never calls that file's main()
(which rewrites its tracked record). Results go to local_rehearsal_record.json beside this file.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = HERE.relative_to(REPO).as_posix()
REAL_SPEC = HERE / "serial_spec.json"
spec_mod = importlib.util.spec_from_file_location("reh", REPO / "platforms/ims/family_serial_rehearsal.py")
reh = importlib.util.module_from_spec(spec_mod)
spec_mod.loader.exec_module(reh)
reh.LIMITS["ws_bytes"] = 22 * 2**30  # 31 GB machine
WALL, RESERVE, MEM = 86400 - 120, 1800, 16 * 1_875_000_000  # what the launcher injects for ncpus 16, 24 h
JOB = ["--cpus", "16", "--training-workers", "0", "--source-commit", "0" * 40]


def load(path=REAL_SPEC) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def put(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def args_for(spec, code, root, out, wall=WALL, reserve=RESERVE, mem=MEM, extra=()):
    return ["--spec", str(spec), "--code-root", str(code), "--input-root", str(root), "--output-root", str(out), *JOB,
            "--wall-s", str(wall), "--reserve-s", str(reserve), "--memory-limit-bytes", str(mem), *extra]


def copy_code(work: Path) -> Path:
    """Scratch code root with the real relative layout: recipes and init files only (the driver reads nothing else)."""
    code = work / "code_real"
    for item in load()["items"]:
        for key in ("recipe", "initial_state"):
            if item.get(key):
                (code / item[key]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / item[key], code / item[key])
    return code


def link_root(fixed: Path, dest: Path, *, skip=(), truncate=()) -> Path:
    """Hard-linked copy of the mirror root with chosen files left out or replaced by a 1 KB stub."""
    shutil.rmtree(dest, ignore_errors=True)
    for src in fixed.rglob("*"):
        rel = src.relative_to(fixed).as_posix()
        if src.is_dir() or any(rel.endswith(s) for s in skip):
            continue
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        if any(rel.endswith(t) for t in truncate):
            (dest / rel).write_bytes(b"x" * 1000)
        else:
            os.link(src, dest / rel)
    return dest


def part_early(work: Path, fixed: Path, env: dict) -> None:
    code = copy_code(work)
    spec = load()
    # 1. real spec, real data: spec validation and check_inputs with ogb blocked and the user site off
    probe = ("import sys, json; sys.modules['ogb'] = None; sys.path.insert(0, sys.argv[1]); import family_serial_run as m; from pathlib import Path\n"
             "spec = json.load(open(sys.argv[2])); print('SPEC', json.dumps(m.validate_spec(spec, float(sys.argv[5]), float(sys.argv[6]))))\n"
             "print('PROBLEMS', json.dumps(m.check_inputs(spec, Path(sys.argv[3]), Path(sys.argv[4]), None, None)))")
    res = subprocess.run([sys.executable, "-c", probe, str(REPO / "platforms/ims"), str(REAL_SPEC), str(REPO), str(fixed), str(WALL), str(RESERVE)],
                         env={**env, "PYTHONNOUSERSITE": "1"}, capture_output=True, text=True)
    lines = {ln.split(" ", 1)[0]: ln.split(" ", 1)[1] for ln in res.stdout.splitlines() if ln.startswith(("SPEC", "PROBLEMS"))}
    reh.record("REAL spec + real 100K/500K data: validate_spec and check_inputs return no problem (ogb blocked, PYTHONNOUSERSITE=1, repo checkout as code root)",
               res.returncode == 0 and lines.get("SPEC") == "[]" and lines.get("PROBLEMS") == "[]", spec_problems=lines.get("SPEC"), input_problems=lines.get("PROBLEMS"),
               stderr=res.stderr[-300:])
    budgets = [i["budget_s"] for i in spec["items"]]
    reh.record("budgets: each at most wall - reserve, and the sum too", max(budgets) <= WALL - RESERVE and sum(budgets) <= WALL - RESERVE,
               budgets=budgets, sum=sum(budgets), limit=WALL - RESERVE)
    # 2. the exit-3 matrix: one broken input at a time on the real spec
    def item_index(name):
        return next(i for i, x in enumerate(spec["items"]) if x["name"] == name)

    def variant(edit):
        s = json.loads(json.dumps(spec))
        edit(s)
        return s
    cases = []
    cases.append(("recipe sha differs (J3)", variant(lambda s: s["items"][item_index("j3-demol-500k-s42")].update(recipe_sha256="0" * 64)), code, fixed, {}))
    cases.append(("initial state sha differs (J2, s43)", variant(lambda s: s["items"][item_index("j2-gptt-100k-s43")].update(initial_state_sha256="1" * 64)), code, fixed, {}))
    cases.append(("initial state sha differs (J5, s44)", variant(lambda s: s["items"][item_index("j5-gptt-500k-s44")].update(initial_state_sha256="1" * 64)), code, fixed, {}))
    tampered = json.loads((REPO / spec["items"][item_index("j3-demol-500k-s42")]["recipe"]).read_text(encoding="utf-8"))
    tampered["train_role_identity"] = "f" * 64
    bad = put(code / EXP / "role_bad.json", tampered)
    cases.append(("train role identity differs (J3)", variant(lambda s: s["items"][item_index("j3-demol-500k-s42")].update(
        recipe=bad.relative_to(code).as_posix(), recipe_sha256=reh.sha(bad))), code, fixed, {}))
    final = json.loads((REPO / spec["items"][item_index("j3-demol-500k-s42")]["recipe"]).read_text(encoding="utf-8"))
    final["selection"] = "final-ema"
    fin = put(code / EXP / "final_ema.json", final)
    cases.append(("gate reads a final-ema item (J4 gate on J3)", variant(lambda s: s["items"][item_index("j3-demol-500k-s42")].update(
        recipe=fin.relative_to(code).as_posix(), recipe_sha256=reh.sha(fin))), code, fixed, {}))
    cases.append(("gate mixes the 100K and 500K development roles", variant(lambda s: s["items"][item_index("j5-gptt-500k-s44")]["gate"].update(
        dev_delta={"a": "j1-demol-100k-s43", "b": "j3-demol-500k-s42", "max_delta_meV": -3.0})), code, fixed, {}))
    cases.append(("budget above wall - reserve (J3)", variant(lambda s: s["items"][item_index("j3-demol-500k-s42")].update(budget_s=WALL - RESERVE + 1)), code, fixed, {}))
    cases.append(("extra_args sets a driver flag (J3 --wall-s)", variant(lambda s: s["items"][item_index("j3-demol-500k-s42")].update(extra_args=["--gpu-resident-cells", "--wall-s", "1"])), code, fixed, {}))
    cases.append(("gate names an item listed later", variant(lambda s: s["items"][item_index("j1-demol-100k-s43")].update(
        gate={"memory": {"from": "j3-demol-500k-s42"}})), code, fixed, {}))
    cases.append(("--memory-limit-bytes 0", spec, code, fixed, {"mem": 0}))
    manifest_gone = link_root(fixed, work / "root_no_500k_manifest", skip=("subsets/ogb-train-500k-scnet-v1/manifest.json",))
    cases.append(("500K manifest missing", spec, code, manifest_gone, {}))
    dev_gone = link_root(fixed, work / "root_no_dev_shard", skip=("train_shard_0010.pt",))
    cases.append(("500K development shard (0010) missing", spec, code, dev_gone, {}))
    dev_small = link_root(fixed, work / "root_small_dev_shard", truncate=("train_shard_0010.pt",))
    cases.append(("500K development shard (0010) has the wrong size", spec, code, dev_small, {}))
    train_small = link_root(fixed, work / "root_small_train_shard", truncate=("train_shard_0007.pt",))
    cases.append(("500K train shard (0007) has the wrong size", spec, code, train_small, {}))
    for number, (name, case_spec, case_code, root, over) in enumerate(cases):
        out = work / "early" / f"case{number}"
        shutil.rmtree(out, ignore_errors=True)
        started = time.monotonic()
        path = put(work / "early_specs" / f"case{number}.json", case_spec)
        result = reh.run_driver(args_for(path, case_code, root, out, mem=over.get("mem", MEM)), work / f"early_case{number}.log", env)
        early = out / "early_exit.json"
        problems = json.loads(early.read_text(encoding="utf-8"))["problems"] if early.is_file() else []
        children = [p.name for p in out.iterdir() if p.is_dir()] if out.is_dir() else []
        reh.record(f"exit 3 on broken input: {name}", result["exit"] == 3 and bool(problems) and not children,
                   exit=result["exit"], problem=problems[:2], child_started=bool(children), seconds=round(time.monotonic() - started, 1))


def import_launcher():
    mod = importlib.util.spec_from_file_location("launcher_real", REPO / "platforms/ims/v4_submit/launcher.py")
    launcher = importlib.util.module_from_spec(mod)
    mod.loader.exec_module(launcher)
    return launcher


def part_launcher(work: Path, fixed: Path, env: dict) -> None:
    launcher = import_launcher()
    datasets = json.loads((REPO / "platforms/ims/v4_submit/dataset_registry.json").read_text(encoding="utf-8"))
    runtimes = json.loads((REPO / "platforms/ims/v4_submit/runtime_registry.json").read_text(encoding="utf-8"))
    spec = load(HERE / "run_spec.json")
    spec["source"] = {"commit": "0" * 40, "archive_sha256": "0" * 64}
    spec["contract"]["sha256"] = reh.sha(REAL_SPEC)
    launcher.validate_spec(spec, datasets, runtimes)

    def refused(edit) -> bool:
        broken = json.loads(json.dumps(spec))
        edit(broken)
        try:
            launcher.validate_spec(broken, datasets, runtimes)
        except RuntimeError:
            return True
        return False
    args = spec["stages"]["train"]["args"]
    reh.record("REAL run_spec validates; resources ncpus 16 / 1 GPU / 24 h; --cpus equals ncpus; --cpus 8, ncpus 8 with --cpus 16, 25 h walltime, and a typed wall are refused",
               spec["resources"] == {"queue": "H", "ncpus": 16, "ompthreads": 4, "ngpus": 1, "train_walltime": "24:00:00"} and args[args.index("--cpus") + 1] == "16"
               and refused(lambda s: s["stages"]["train"]["args"].__setitem__(s["stages"]["train"]["args"].index("--cpus") + 1, "8"))
               and refused(lambda s: s["resources"].update(ncpus=8)) and refused(lambda s: s["resources"].update(train_walltime="25:00:00"))
               and refused(lambda s: s["stages"]["train"]["args"].__setitem__(s["stages"]["train"]["args"].index("--wall-s") + 1, "86400")),
               resources=spec["resources"])
    runtime = runtimes["runtimes"][spec["runtime_id"]]
    run_root = Path("/lustre/home/users/sm2/chou/molgap-v4-submit/runs") / spec["run_id"]
    pbs = launcher.render_pbs(run_root, "train", spec, runtime)
    (work / "train.pbs").write_text(pbs, encoding="utf-8", newline="\n")
    bash = subprocess.run([shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe", "-n", str(work / "train.pbs").replace("\\", "/")], capture_output=True, text=True)
    select = [ln for ln in pbs.splitlines() if "select" in ln or "walltime" in ln]
    reh.record("train.pbs of the real spec: bash -n passes, select ncpus=16 ngpus=1, walltime 24:00:00, no mem=", bash.returncode == 0 and "ncpus=16:" in pbs and "ngpus=1" in pbs
               and "walltime=24:00:00" in pbs and "mem=" not in pbs, lines=select, stderr=bash.stderr[:200])
    context = {"code_root": "/code", "dataset_root": "/fixed", "training_output": "/out", "source_commit": "c" * 40,
               "train_wall_s": str(launcher.walltime_seconds("24:00:00") - launcher.WALL_MARGIN_S), "mem_bytes": str(16 * launcher.MEM_PER_CORE_BYTES)}
    expanded = launcher.expand_args(spec["stages"]["train"]["args"], context)
    flat = " ".join(expanded)
    reh.record("launcher expands the real train args: --wall-s 86280, --memory-limit-bytes 30000000000, --cpus 16, accelerator A100",
               "--wall-s 86280" in flat and "--memory-limit-bytes 30000000000" in flat and "--cpus 16" in flat and "--accelerator A100" in flat, args=flat[-230:])
    # end to end: stage, fake jsub, worker runs the driver for real on a one-item variant that keeps the real resources (16 cores, 24 h)
    infra = work / "infra"
    shutil.rmtree(infra, ignore_errors=True)
    infra.mkdir()
    launcher.BOUNDARY, launcher.INFRA_ROOT = work, infra
    datasets["fixed_root"] = str(fixed)
    runtimes["runtimes"]["ims-a100-torch271-v1"].update(python=sys.executable, pythonpath=[str(REPO / "src")])
    for name, payload in (("dataset_registry.json", datasets), ("runtime_registry.json", runtimes)):
        (infra / name).write_text(json.dumps(payload), encoding="utf-8")
    launcher.validate_registries = lambda base: (datasets, runtimes)
    one = load()["items"][1]  # J2, GPT-T 100K with the s43 init file
    one = {**one, "name": "j2-gptt-20-steps", "budget_s": 600, "extra_args": ["--max-steps", "20"]}
    serial = put(work / "variant_serial.json", {"format": "molgap-ims-family-serial-v1", "items": [one]})
    files = {"platforms/ims/family_serial_run.py": REPO / "platforms/ims/family_serial_run.py",
             "platforms/ims/run_family_training.py": REPO / "platforms/ims/run_family_training.py",
             "platforms/ims/family_recipe_run.py": REPO / "platforms/ims/family_recipe_run.py", f"{EXP}/serial_spec.json": serial,
             one["recipe"]: REPO / one["recipe"], one["initial_state"]: REPO / one["initial_state"]}
    inbox = infra / "inbox" / spec["run_id"]
    inbox.mkdir(parents=True)
    with tarfile.open(inbox / "source.tar.gz", "w:gz") as tar:
        for rel, src in files.items():
            tar.add(src, arcname=rel)
    (inbox / "source_files.json").write_text(json.dumps({"format": "molgap-source-files-v1", "files": [{"path": rel, "sha256": reh.sha(src)} for rel, src in files.items()]}), encoding="utf-8")
    local = json.loads(json.dumps(spec))  # only the machine-specific args change: no A100 check, no lustre zip, no spawn workers
    local["source"]["archive_sha256"], local["contract"]["sha256"] = reh.sha(inbox / "source.tar.gz"), reh.sha(serial)
    local_args = local["stages"]["train"]["args"]
    for flag in ("--accelerator", "--ogb-zip"):
        index = local_args.index(flag)
        del local_args[index:index + 2]
    local_args += ["--training-workers", "0", "--reserve-s", "60"]
    (inbox / "run_spec.json").write_text(json.dumps(local), encoding="utf-8")
    calls = []
    real_run = subprocess.run

    def fake_run(command, **kw):
        if command[0] == "jsub":
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="12345.ccfep\n", stderr="")
        return real_run(command, **kw)
    launcher.subprocess.run, launcher.maintenance_guard = fake_run, lambda seconds: None
    launcher.submit_run(inbox, infra)
    run = infra / "runs" / spec["run_id"]
    os.environ.update(PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="0")
    try:
        launcher.worker(run, "train")
        code = 0
    except SystemExit as stop:
        code = stop.code
    launcher.subprocess.run = real_run
    summary = json.loads((run / "output/training/serial_summary.json").read_text(encoding="utf-8"))
    reh.record("worker -> driver -> child with the real resources: injected wall_s 86280 and memory_limit_bytes 30000000000; item paused (exit 2)",
               code == 2 and summary["wall_s"] == 86280 and summary["memory_limit_bytes"] == 30_000_000_000 and summary["items"]["j2-gptt-20-steps"]["status"] == "PAUSED",
               worker_exit=code, wall_s=summary["wall_s"], memory_limit_bytes=summary["memory_limit_bytes"], jsub=calls[0][-3:] if calls else None)


def short_code(work: Path) -> Path:
    """Copy of the real code layout with one-pass recipes (100K: 781 steps, 500K: 3906 steps); keeps checkpoint_every, seeds, selection."""
    code = work / "code_short"
    shutil.rmtree(code, ignore_errors=True)
    for item in load()["items"]:
        recipe = json.loads((REPO / item["recipe"]).read_text(encoding="utf-8"))
        steps = recipe["train_rows"] // recipe["batch_size"]
        recipe["optimizer_steps"] = steps
        recipe["schedule"].update(reference_epochs=2, total_units=2, warmup_units=1)
        recipe["acceptance_requirements"].update(optimizer_steps=steps, completed_passes=1, sample_presentations=steps * recipe["batch_size"], next_batch=0)
        put(code / item["recipe"], recipe)
        if item.get("initial_state"):
            (code / item["initial_state"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / item["initial_state"], code / item["initial_state"])
    return code


def short_spec(code: Path, *, dev_line=None, line45=None, mem_fraction=None, probes=False) -> dict:
    """The real spec with the short recipes' hashes; thresholds can be forced to drive a chosen gate path."""
    spec = load()
    for item in spec["items"]:
        item["recipe_sha256"] = reh.sha(code / item["recipe"])
        gate = item.get("gate", {})
        if "dev_delta" in gate:
            rule = gate["dev_delta"]
            if rule["a"].startswith("j1") and dev_line is not None:
                rule["max_delta_meV"] = dev_line
            if rule["a"].startswith("j3") and line45 is not None:
                rule["max_delta_meV"] = line45
        if "memory" in gate and mem_fraction is not None and item["name"].startswith("j3"):
            gate["memory"]["max_fraction"] = mem_fraction
    if probes:  # a DeMol recipe given the GPT-T state fails after loading data; the last item can never fit in the wall left
        j1 = spec["items"][0]
        spec["items"].append({**{k: j1[k] for k in ("recipe", "recipe_sha256", "extra_args")}, "name": "probe-fail", "budget_s": 600,
                              "initial_state": f"{EXP.rsplit('/', 1)[0]}/pcqm_demol_cin_s43_100k_ims_20261120/init/s43.pt",
                              "initial_state_sha256": "2ccd1eb79de4b82fffb58522b9e2325026d36f04f398be0d49d3eea55ee05f3c"})
        spec["items"].append({**{k: j1[k] for k in ("recipe", "recipe_sha256", "extra_args")}, "name": "probe-time", "budget_s": 35940})
    return spec


def statuses(root: Path) -> dict:
    return {k: v["status"] for k, v in reh.summary_of(root)["items"].items()}


def part_chain(work: Path, fixed: Path, env: dict, stage: str) -> None:
    code = short_code(work) if stage == "a" else work / "code_short"
    root = work / "chain"
    wall, reserve = 36000, 60
    summary_path = root / "serial_summary.json"

    def go(spec_path, log, **kw):
        return reh.run_driver(args_for(spec_path, code, fixed, root, wall=wall, reserve=reserve, extra=["--max-attempts", "2"]), work / log, env, watch_summary=summary_path, **kw)
    if stage == "a":  # kill during J1 after its first checkpoint, then resume; real thresholds
        shutil.rmtree(root, ignore_errors=True)
        spec = put(work / "chain_spec_real.json", short_spec(code, probes=True))
        ckpt = root / "j1-demol-100k-s43/last_checkpoint.pt"
        first = go(spec, "chain_a.log", kill_when=lambda: ckpt.is_file() and not (root / "j1-demol-100k-s43/completion_manifest.json").is_file())
        a = reh.summary_of(root)["items"]
        reh.record("A: kill mid J1 leaves a parseable summary, J1 RUNNING", first["killed"] and a["j1-demol-100k-s43"]["status"] == "RUNNING" and first["bad_summary_reads"] == 0,
                   statuses=statuses(root), bad_reads=first["bad_summary_reads"], handles=first["peak_handles"], ws_gib=first["peak_ws_gib"])
        second = go(spec, "chain_b.log")
        b = reh.summary_of(root)["items"]
        log = (work / "chain_b.log").read_text(encoding="utf-8", errors="replace")
        resumed = [ln for ln in log.splitlines() if ln.startswith("resumed at step")]
        delta = (b["j1-demol-100k-s43"]["dev_mae_eV"] - b["j2-gptt-100k-s43"]["dev_mae_eV"]) * 1000 if all(b[n].get("dev_mae_eV") for n in ("j1-demol-100k-s43", "j2-gptt-100k-s43")) else None
        reh.record("A: resume continues J1 from its checkpoint; J1 and J2 complete (1 pass)", bool(resumed) and b["j1-demol-100k-s43"]["status"] == "COMPLETE" and b["j2-gptt-100k-s43"]["status"] == "COMPLETE",
                   resumed=resumed[:1], statuses=statuses(root), dev_delta_meV=delta, j1_peak_rss_gib=round((b["j1-demol-100k-s43"].get("peak_rss_bytes") or 0) / 2**30, 2))
        want_skip = delta is not None and delta > -3.0
        reh.record("A: REAL dev gate (-3.0) after one pass: J3 and J5 skipped when delta is above the line, J4 skipped with them (no dev input); otherwise they ran",
                   (not want_skip) or (b["j3-demol-500k-s42"]["status"] == "SKIPPED_BY_GATE" and b["j5-gptt-500k-s44"]["status"] == "SKIPPED_BY_GATE" and b["j4-demol-500k-s43"]["status"] == "SKIPPED_BY_GATE"),
                   delta_meV=delta, statuses=statuses(root), j3=b["j3-demol-500k-s42"]["reason"][:120], j4=b["j4-demol-500k-s43"]["reason"][:120])
        reh.record("A: failed probe retried up to the limit; time probe skipped for time; exit code 2 (resumable)", len(b["probe-fail"]["attempts"]) >= 1 and b["probe-time"]["status"] == "SKIPPED_FOR_TIME" and second["exit"] == 2,
                   probe_fail=b["probe-fail"]["status"], attempts=len(b["probe-fail"]["attempts"]), probe_time=b["probe-time"]["reason"][:100], exit=second["exit"])
    elif stage == "b":  # third run: no retry beyond the attempt limit; then force the dev gate open with J3's memory gate shut (J3 skipped for memory, J5 runs, J4 follows J3)
        third = go(work / "chain_spec_real.json", "chain_c.log")  # rerun: the failed probe gets its second attempt; finished items are not redone
        c = reh.summary_of(root)["items"]
        log = (work / "chain_c.log").read_text(encoding="utf-8", errors="replace")
        reh.record("B: rerun retries the FAILED probe once (attempt 2) and starts no finished item", len(c["probe-fail"]["attempts"]) == 2 and c["probe-fail"]["status"] == "FAILED"
                   and "[driver] probe-fail start" in log and "[driver] j1-demol-100k-s43 start" not in log and "[driver] j2-gptt-100k-s43 start" not in log,
                   attempts=len(c["probe-fail"]["attempts"]), exit=third["exit"])
        forced = put(work / "chain_spec_memshut.json", short_spec(code, dev_line=1e6, mem_fraction=1e-9, probes=True))
        run = go(forced, "chain_d.log")
        d = reh.summary_of(root)["items"]
        log = (work / "chain_d.log").read_text(encoding="utf-8", errors="replace")
        reh.record("B: a FAILED item stops at the attempt limit (2) and is not started again", len(d["probe-fail"]["attempts"]) == 2 and "attempt limit" in d["probe-fail"]["reason"]
                   and "[driver] probe-fail start" not in log, attempts=len(d["probe-fail"]["attempts"]), reason=d["probe-fail"]["reason"])
        reh.record("B: dev gate open, memory gate shut: J3 SKIPPED_FOR_MEMORY with the predicted GiB; J5 (same dev gate, real memory gate) runs and completes; J4 SKIPPED_BY_GATE because J3 is not COMPLETE (no dev input)",
                   d["j3-demol-500k-s42"]["status"] == "SKIPPED_FOR_MEMORY" and d["j5-gptt-500k-s44"]["status"] == "COMPLETE" and d["j4-demol-500k-s43"]["status"] == "SKIPPED_BY_GATE",
                   statuses=statuses(root), j3=d["j3-demol-500k-s42"]["reason"][:200], j4=d["j4-demol-500k-s43"]["reason"][:120], handles=run["peak_handles"], ws_gib=run["peak_ws_gib"])
    elif stage == "c":  # J3 runs (all gates open); J4 gate forced shut -> skip path
        spec = put(work / "chain_spec_j4shut.json", short_spec(code, dev_line=1e6, line45=-1e6, probes=True))
        run = go(spec, "chain_e.log")
        e = reh.summary_of(root)["items"]
        delta = (e["j3-demol-500k-s42"]["dev_mae_eV"] - e["j5-gptt-500k-s44"]["dev_mae_eV"]) * 1000 if e["j3-demol-500k-s42"].get("dev_mae_eV") and e["j5-gptt-500k-s44"].get("dev_mae_eV") else None
        reh.record("C: all gates open, J3 (DeMol 500K, GPU-resident, 1 pass) completes; J4 gate (J3 - J5, forced line) skips J4 with the measured delta in the reason",
                   e["j3-demol-500k-s42"]["status"] == "COMPLETE" and e["j4-demol-500k-s43"]["status"] == "SKIPPED_BY_GATE",
                   statuses=statuses(root), j3_vs_j5_delta_meV=delta, j4=e["j4-demol-500k-s43"]["reason"][:160], j3_peak_rss_gib=round((e["j3-demol-500k-s42"].get("peak_rss_bytes") or 0) / 2**30, 2),
                   handles=run["peak_handles"], ws_gib=run["peak_ws_gib"])
    elif stage == "d":  # J4 gate opened: J4 runs; the real J4 line (-1.5) is evaluated on the measured delta
        spec = put(work / "chain_spec_j4open.json", short_spec(code, dev_line=1e6, line45=1e6, probes=True))
        run = go(spec, "chain_f.log")
        f = reh.summary_of(root)["items"]
        reh.record("D: J4 gate open: J4 (DeMol 500K s43, 1 pass) completes after J3 and J5; a rerun redoes nothing", f["j4-demol-500k-s43"]["status"] == "COMPLETE",
                   statuses=statuses(root), handles=run["peak_handles"], ws_gib=run["peak_ws_gib"], exit=run["exit"])
        pause = work / "pause_probe"
        shutil.rmtree(pause, ignore_errors=True)
        j3 = next(i for i in short_spec(code)["items"] if i["name"].startswith("j3"))
        j3["name"] = "j3-pause-probe"
        j3.pop("gate", None)
        j3["budget_s"] = 200  # must fit wall 300 - reserve 60
        pspec = put(work / "pause_spec.json", {"format": "molgap-ims-family-serial-v1", "items": [j3]})
        small = reh.run_driver(args_for(pspec, code, fixed, pause, wall=300, reserve=60), work / "pause_a.log", env)
        status = reh.summary_of(pause)["items"]["j3-pause-probe"]["status"]
        ckpt = (pause / "j3-pause-probe/last_checkpoint.pt").is_file()
        big = reh.run_driver(args_for(pspec, code, fixed, pause, wall=36000, reserve=60), work / "pause_b.log", env)
        log = (work / "pause_b.log").read_text(encoding="utf-8", errors="replace")
        resumed = [ln for ln in log.splitlines() if ln.startswith("resumed at step")]
        reh.record("D: 500K DeMol item paused by its own wall clock (exit 2, PAUSED, checkpoint written), then resumed by the same command and completed",
                   small["exit"] == 2 and status == "PAUSED" and ckpt and bool(resumed) and reh.summary_of(pause)["items"]["j3-pause-probe"]["status"] == "COMPLETE",
                   first_exit=small["exit"], first_status=status, resumed=resumed[:1], final=reh.summary_of(pause)["items"]["j3-pause-probe"]["status"], handles=max(small["peak_handles"], big["peak_handles"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("early", "launcher", "chain-a", "chain-b", "chain-c", "chain-d"))
    ap.add_argument("--work", type=Path, default=Path("D:/molgap-artifacts/ims_chain_20261121/rehearsal"))
    ap.add_argument("--fixed-root", type=Path, default=Path("D:/molgap-artifacts/ims_chain_20261121/fixed_root"))
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
    if a.part == "early":
        part_early(a.work, a.fixed_root, env)
    elif a.part == "launcher":
        part_launcher(a.work, a.fixed_root, env)
    else:
        part_chain(a.work, a.fixed_root, env, a.part.split("-")[1])
    path = HERE / "local_rehearsal_record.json"
    old = json.loads(path.read_text(encoding="utf-8"))["results"] if path.is_file() else []
    merged = [r for r in old if r["check"] not in {x["check"] for x in reh.results}] + reh.results
    path.write_text(json.dumps({"date": time.strftime("%Y-%m-%d"), "results": merged}, indent=1) + "\n", encoding="utf-8")
    failed = [r["check"] for r in reh.results if not r["passed"]]
    print("[rehearsal] FAILED:" if failed else "[rehearsal] all passed", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
