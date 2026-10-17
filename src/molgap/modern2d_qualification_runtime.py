"""Bounded Kaggle qualification orchestration; no training or submission route.

This module starts with stdlib imports so the frozen bootstrap can load it
before preparing family-shared Python. Only isolated workers import the bridge.
Launch paths select exact inputs; no role, payload or label discovery occurs.

Launch v1: format, spec_identity, expected_package_identity,
expected_source_archive_sha256, account, run_reference, max_wall_seconds,
authority={path,sha256}, private_asset={dataset_source,mount,data_root,
acceptance_sha256,version_identity,train_logical_content_sha256,
development_logical_content_sha256}, jobs=[{arm_id,device,context,recipe,
initial_state}]. The last three job fields are {path,sha256} references.
Recipes are relative to extracted source; other references are relative to
the launch directory. mount is relative to mounted_root; data_root is relative
to that mount. Each recipe's input binding must exist directly in that mount.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import types


LAUNCH_FORMAT = "molgap-modern2d-qualification-launch-v1"
LAUNCH_FILENAME = "modern2d_qualification_launch.json"
SEAL_RESERVE_SECONDS = 60
POLL_SECONDS = 5
INTERFACES = {"ppgt_author_v2": ("ppgt_small", "2"), "esa_native_v3": ("esa_small", "3")}
_FIELDS = set("format spec_identity expected_package_identity expected_source_archive_sha256 account run_reference authority private_asset max_wall_seconds jobs".split())
_ASSET_FIELDS = set("dataset_source mount data_root acceptance_sha256 version_identity train_logical_content_sha256 development_logical_content_sha256".split())


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected lowercase SHA256")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    def unique(pairs):
        record = {}
        for key, value in pairs:
            if key in record:
                raise ValueError("Duplicate JSON key")
            record[key] = value
        return record
    path = Path(path)
    if path.stat().st_size > 1024**2:
        raise ValueError("Qualification metadata exceeds one MiB")
    value = json.loads(path.read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if type(value) is not dict:
        raise ValueError("Expected JSON object")
    return value


def _under(root, relative):
    if (type(relative) is not str or not relative or "\\" in relative or ":" in relative
            or any(p in {"", ".", ".."} or p.endswith((".", " ")) or
                   any(ord(c) < 32 or c in '<>"|?*' for c in p) for p in relative.split("/"))):
        raise ValueError("Expected confined relative POSIX path")
    root = Path(root).absolute()
    path = root / relative
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Links are forbidden in qualification inputs")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Qualification path escaped its root")
    return path


def _ref(root, record):
    if type(record) is not dict or set(record) != {"path", "sha256"}:
        raise ValueError("Each artifact reference requires path and sha256")
    path = _under(root, record["path"])
    if not path.is_file() or _sha(path) != _digest(record["sha256"]):
        raise ValueError("Qualification artifact bytes differ: " + record["path"])
    return path


def _atomic(path, value):
    # Reuse the stdlib-only durable IO owner after the source root is installed.
    from molgap.research_memory.trace import atomic_write, json_bytes
    atomic_write(path, json_bytes(value))


def read_launch(path, expected_sha256):
    """Pin bytes before interpreting the deliberately separate qualification ABI."""
    path = Path(path).absolute()
    if _sha(path) != _digest(expected_sha256):
        raise ValueError("Qualification launch differs from the pinned entrypoint")
    config = _json(path)
    if set(config) != _FIELDS or config["format"] != LAUNCH_FORMAT:
        raise ValueError("Unknown or missing qualification launch fields")
    for key in ("spec_identity", "expected_package_identity", "expected_source_archive_sha256"):
        _digest(config[key])
    if (type(config["max_wall_seconds"]) is not int or
            not 1 <= config["max_wall_seconds"] <= 1800):
        raise ValueError("Qualification ceiling must be at most 1800 total seconds")
    if (type(config["account"]) is not str or re.fullmatch(r"[a-z0-9-]+", config["account"]) is None or
            type(config["run_reference"]) is not str or
            re.fullmatch(re.escape(config["account"]) + r"/[a-z0-9-]+", config["run_reference"]) is None):
        raise ValueError("Qualification account/run binding is invalid")
    asset = config["private_asset"]
    if type(asset) is not dict or set(asset) != _ASSET_FIELDS:
        raise ValueError("An explicit selected private asset is required")
    if (type(asset["dataset_source"]) is not str or
            re.fullmatch(re.escape(config["account"]) + r"/[a-z0-9-]+", asset["dataset_source"]) is None):
        raise ValueError("Private data must bind the owning account's exact dataset")
    for key in _ASSET_FIELDS - {"dataset_source", "mount", "data_root"}:
        _digest(asset[key])
    _under(Path("/kaggle/input"), asset["mount"])
    _under(Path("/kaggle/input"), asset["data_root"])
    jobs = config["jobs"]
    if (type(jobs) is not list or len(jobs) != 2 or any(type(j) is not dict or
            set(j) != {"arm_id", "device", "context", "recipe", "initial_state"} for j in jobs)):
        raise ValueError("Exactly two qualification jobs are required")
    if ({j["arm_id"] for j in jobs} != set(INTERFACES) or
            any(type(j["device"]) is not int for j in jobs) or {j["device"] for j in jobs} != {0, 1}):
        raise ValueError("Qualification requires the exact two arms on distinct devices 0/1")
    return config


def validate_launch(*, launch_path, expected_launch_sha256, source_root, package_dir, mounted_root):
    """Metadata-only binding gate; it never calls the formal execution planner."""
    from molgap.experiment_package import verify_experiment_source_package
    from molgap.experiment_spec import ExperimentSpec
    from molgap.experiment_family_workflow import RunContext
    config = read_launch(launch_path, expected_launch_sha256)
    manifest = verify_experiment_source_package(Path(package_dir))
    spec = ExperimentSpec.from_json((Path(package_dir) / "experiment_spec.json").read_text(encoding="utf-8"))
    if (manifest["package_identity"] != config["expected_package_identity"] or
            manifest["archive_sha256"] != config["expected_source_archive_sha256"] or
            manifest["spec_identity"] != config["spec_identity"] or spec.identity != config["spec_identity"]):
        raise ValueError("Qualification Spec/package/archive binding differs")
    declaration = spec.to_dict()
    platform = declaration["platform"]
    if (platform["name"] != "kaggle" or platform["accelerator"] != "NvidiaTeslaT4" or
            platform["device_count"] != 2 or len(declaration["arms"]) != 2 or
            {a["arm_id"]: (a["family"]["name"], a["family"]["version"]) for a in declaration["arms"]} != INTERFACES):
        raise ValueError("Qualification requires the exact PPGT/2 + ESA/3 T4x2 Spec")
    staged = Path(launch_path).parent
    authority_path = _ref(staged, config["authority"])
    authority = _json(authority_path)
    if (authority.get("purpose") != "EXECUTION_QUALIFICATION_ONLY" or
            authority.get("user_authorized") is not True or
            authority.get("gpu", {}).get("owner") != config["account"] or
            any(authority.get(key) is not False for key in ("scientific_training_authorized",
                "protected_role_access_authorized", "extra_seed_authorized", "automatic_successor_authorized"))):
        raise ValueError("Qualification-only authority required")
    asset = config["private_asset"]
    input_root = _under(mounted_root, asset["mount"])
    data_root = _under(input_root, asset["data_root"])
    if not input_root.is_dir() or not data_root.is_dir():
        raise ValueError("Exact private input/data root is absent")
    acceptance_path = _under(data_root, "acceptance.json")
    if _sha(acceptance_path) != asset["acceptance_sha256"]:
        raise ValueError("Private accepted-data receipt differs")
    acceptance = _json(acceptance_path)
    if (acceptance.get("rule_identity") != asset["version_identity"] or
            acceptance.get("status") != "DATA_ROLE_ACCEPTED" or
            acceptance.get("scope") != "OFFICIAL_TRAIN_CPU_PREPARATION_ONLY" or
            set(acceptance.get("roles", {})) != {"train_100k", "common_dev"}):
        raise ValueError("Private asset must contain only accepted 100K/Common-Dev roles")
    for role, logical in (("train_100k", "train_logical_content_sha256"), ("common_dev", "development_logical_content_sha256")):
        if acceptance["roles"][role].get("logical_content_sha256") != asset[logical]:
            raise ValueError("Selected asset logical identity differs")
    inventory = {item["path"]: item["sha256"] for item in _json(Path(package_dir) / "SOURCE_FILES.json")["files"]}
    contexts = {}
    for job in config["jobs"]:
        context_path = _ref(staged, job["context"])
        # Typed permit construction imports the model data stack. Leave it to
        # the isolated worker after family-shared Python has been qualified.
        context = _json(context_path)
        expected = RunContext.for_training(spec, Path(package_dir),
            expected_package_identity=config["expected_package_identity"], arm_id=job["arm_id"],
            account=config["account"], run_reference=config["run_reference"])
        if (context.get("run_context") != expected.to_dict() or
                context.get("authority_sha256") != config["authority"]["sha256"] or
                context.get("max_wall_seconds") != config["max_wall_seconds"]):
            raise ValueError("Reviewed context differs from launch/authority/run")
        for name, role in (("train_permit", "train_100k"), ("development_permit", "common_dev")):
            permit = context[name]
            role_record = acceptance["roles"][role]
            if (permit.get("role") != role or
                    permit.get("acceptance_sha256") != asset["acceptance_sha256"] or
                    permit.get("version_identity") != asset["version_identity"] or
                    role_record.get("sha256") != permit.get("manifest_sha256")):
                raise ValueError("Reviewed role permit differs from selected private asset")
        recipe_path = _ref(source_root, job["recipe"])
        if (inventory.get(job["recipe"]["path"]) != job["recipe"]["sha256"] or
                expected.training_recipe_sha256 != job["recipe"]["sha256"]):
            raise ValueError("Qualification recipe is not the exact packaged Spec recipe")
        recipe = _json(recipe_path)
        if recipe.get("qualification") != {"purpose": "numerical-resume-resources-only",
                "steps_per_path": 2, "scratch_updates_per_model": 6}:
            raise ValueError("Exactly six scratch updates must be explicitly declared")
        selection = recipe["dataset"]
        binding_path = _under(input_root, selection["manifest"])
        if not binding_path.is_file() or _sha(binding_path) != selection["manifest_sha256"]:
            raise ValueError("Exact indexed binding is absent; discovery is forbidden")
        binding = _json(binding_path)
        if (binding.get("format") != "molgap-indexed-training-input-v1" or
                binding.get("scope") != "ACCEPTED_INDEXED_INPUT_V1" or
                _under(binding_path.parent, binding["data_root"]).resolve() != data_root.resolve()):
            raise ValueError("Recipe indexed binding resolves to a different data root")
        _ref(staged, job["initial_state"])
        contexts[job["arm_id"]] = context
    return config, spec, manifest, contexts, input_root, authority_path


def _stop_owned(children):
    """Only process groups created by this invocation are signalled."""
    # Reaped bootstrap PIDs may have been reused while workers were running.
    # Snapshot live children and never signal a completed child's old group.
    alive = [child for child in children if child.poll() is None]
    for child in alive:
        try:
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGTERM)
            elif child.poll() is None:
                child.terminate()
        except ProcessLookupError:
            pass
    deadline = time.perf_counter() + 3
    while time.perf_counter() < deadline and any(p.poll() is None for p in alive):
        time.sleep(.05)
    for child in alive:
        if child.poll() is not None:
            continue
        try:
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGKILL)
            elif child.poll() is None:
                child.kill()
        except ProcessLookupError:
            pass
        child.wait(timeout=2)


def _observe_hardware(deadline):
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("Qualification deadline exhausted before hardware observation")
    names = [line.strip() for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
        timeout=min(5, remaining)).splitlines() if line.strip()]
    if len(names) != 2 or any(name not in {"Tesla T4", "NVIDIA Tesla T4", "NVIDIA T4", "T4"} for name in names):
        raise ValueError("Qualification requires exactly two observed Tesla T4 devices")
    return names


def _wait(children, *, deadline, tick):
    while True:
        codes = [child.poll() for child in children]
        tick()
        if time.perf_counter() >= deadline:
            raise TimeoutError("Whole allocation qualification deadline; reserve output sealing")
        if any(code not in (None, 0) for code in codes):
            raise RuntimeError("Qualification subprocess failed; retained logs identify the blocker")
        if all(code == 0 for code in codes):
            return
        time.sleep(min(POLL_SECONDS, max(0, deadline - time.perf_counter())))


def _seal_files(output, report, jobs):
    # Explicit known filenames keep failure retention away from arbitrary files.
    names = ["execution_report.json", "allocation_ledger.json", "bootstrap.log",
             "environment_qualification.json", "environment_result.json", "failure.log", "execution_retention.json"]
    for job in jobs:
        names.extend(job["arm_id"] + "/" + name for name in (
            "qualification.log", "allocation_ledger.json"))
        names.extend(job["arm_id"] + "/scratch/" + name for name in (
            "worker_result.json", "failure.log",
            "runtime_certificate.json", "runtime_provenance.json", "resource_qualification.json",
            "preflight_costs.json", "repeat-a.pt", "repeat-b.pt", "resumed.pt", "resumed-first.pt"))
    files = []
    for relative in names:
        path = _under(output, relative)
        if path.is_file():
            files.append({"path": relative, "sha256": _sha(path), "bytes": path.stat().st_size})
    _atomic(output / "qualification_manifest.json", {
        "format": "molgap-modern2d-qualification-output-v1", "status": report["status"],
        "scope": "QUALIFICATION_ONLY", "spec_identity": report["spec_identity"],
        "launch_sha256": report["launch_sha256"], "files": files,
        "scientific_acceptance": "NOT_EVALUATED", "native_execution_released": False})


def run_qualification_pair(*, source_root, package_dir, mounted_root, launch_path,
                           expected_launch_sha256, output, runtime_root,
                           allocation_started=None):
    """Supervise environment preparation and exactly two native scratch workers."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.perf_counter() if allocation_started is None else allocation_started
    if not math.isfinite(started) or started > time.perf_counter():
        raise ValueError("Invalid allocation start")
    source_root, package_dir, output = map(Path, (source_root, package_dir, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Qualification requires a fresh output root")
    output.mkdir(parents=True, exist_ok=True)
    children, logs, ledger, spec, manifest, jobs = [], [], None, None, None, []
    config = read_launch(launch_path, expected_launch_sha256)
    hard_deadline = started + config["max_wall_seconds"]
    work_deadline = hard_deadline - SEAL_RESERVE_SECONDS
    report = {"format": "molgap-modern2d-qualification-pair-v1", "status": "running",
        "scope": "QUALIFICATION_ONLY", "spec_identity": config["spec_identity"],
        "launch_sha256": expected_launch_sha256, "phase": "metadata",
        "max_wall_seconds": config["max_wall_seconds"], "seal_reserve_seconds": SEAL_RESERVE_SECONDS,
        "scientific_acceptance": "NOT_EVALUATED", "native_execution_released": False,
        "formal_training_exposure": 0, "workers": {}}
    try:
        names = _observe_hardware(work_deadline)
        from molgap.experiment_allocation import AllocationLedger
        jobs = config["jobs"]
        for job in jobs:
            (output / job["arm_id"]).mkdir(exist_ok=False)
        ledger = AllocationLedger(spec_identity=config["spec_identity"], hardware=names,
            assignments={j["arm_id"]: j["device"] for j in jobs}, started=started)
        roots = [output / j["arm_id"] for j in jobs]
        def tick():
            ledger.write(output, "running", arm_roots=roots)
            report["wall_seconds"] = time.perf_counter() - started
            _atomic(output / "execution_report.json", report)
        tick()
        config, spec, manifest, contexts, input_root, _ = validate_launch(
            launch_path=launch_path, expected_launch_sha256=expected_launch_sha256,
            source_root=source_root, package_dir=package_dir, mounted_root=mounted_root)
        if time.perf_counter() >= work_deadline:
            raise TimeoutError("Qualification metadata exhausted allocation budget")
        report["hardware"] = names
        report["phase"] = "environment"
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(source_root / "src"),
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        log = (output / "bootstrap.log").open("x", encoding="utf-8")
        logs.append(log)
        environment_started = time.perf_counter()
        command = [sys.executable, str(source_root / "src/molgap/modern2d_qualification_runtime.py"),
            "environment", "--source-root", str(source_root), "--runtime-root", str(runtime_root),
            "--output", str(output), "--deadline", repr(time.time() + work_deadline - time.perf_counter())]
        child = subprocess.Popen(command, cwd=source_root, env=environment,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
        children.append(child)
        _wait([child], deadline=work_deadline, tick=tick)
        executable = _json(output / "environment_result.json")["executable"]
        environment_report = _json(output / "environment_qualification.json")
        if (environment_report.get("packages", {}).get("torch") != "2.7.1+cu126" or
                environment_report.get("status") != "IMPORTS_QUALIFIED_NOT_GPU_CALIBRATED"):
            raise ValueError("Pinned family-shared environment qualification missing")
        report["environment_wall_seconds"] = time.perf_counter() - environment_started
        report["bootstrap_through_environment_wall_seconds"] = time.perf_counter() - started
        report["phase"] = "qualification"
        workers = []
        for job in jobs:
            if time.perf_counter() >= work_deadline:
                raise TimeoutError("Qualification allocation exhausted before worker spawn")
            arm = job["arm_id"]
            env = dict(environment, CUDA_VISIBLE_DEVICES=str(job["device"]), PYTHONHASHSEED="42",
                CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            log = (output / arm / "qualification.log").open("x", encoding="utf-8")
            logs.append(log)
            command = [str(executable), "-m", "molgap.modern2d_qualification_runtime", "worker",
                "--source-root", str(source_root), "--package-dir", str(package_dir),
                "--mounted-root", str(mounted_root), "--launch", str(launch_path),
                "--expected-launch-sha256", expected_launch_sha256, "--arm", arm,
                "--device", str(job["device"]), "--output", str(output / arm / "scratch"),
                "--allocation-started", repr(started)]
            process = subprocess.Popen(command, cwd=source_root, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
            children.append(process)
            workers.append(process)
            report["workers"][arm] = {"pid": process.pid, "device": job["device"], "status": "running"}
            tick()
        _wait(workers, deadline=work_deadline, tick=tick)
        for job, process in zip(jobs, workers):
            arm = job["arm_id"]
            from molgap.common.wire import canonical_fingerprint
            context_identity = canonical_fingerprint(contexts[arm])
            result = _json(output / arm / "scratch/worker_result.json")
            certificate = _json(output / arm / "scratch/runtime_certificate.json")
            if (result.get("status") != "QUALIFICATION_VERIFIED" or
                    result.get("launch_sha256") != expected_launch_sha256 or
                    result.get("context_identity") != context_identity or
                    certificate.get("format") != "molgap-modern2d-qualification-certificate-v1" or
                    certificate.get("status") != "QUALIFICATION_VERIFIED" or
                    certificate.get("qualification_context_identity") != context_identity or
                    certificate.get("scratch_optimizer_updates") != 6 or
                    certificate.get("formal_training_exposure") != 0 or
                    certificate.get("native_execution_released") is not False):
                raise ValueError("Worker did not produce the distinct bounded qualification certificate")
            report["workers"][arm].update(status="QUALIFICATION_VERIFIED", returncode=process.returncode)
        report["status"] = "complete"
        return report
    except BaseException as exc:
        report.update(status="failed", failure_type=type(exc).__name__, failure=str(exc))
        _atomic(output / "failure.log", {"traceback": traceback.format_exc()})
        raise
    finally:
        final_error = None
        try:
            _stop_owned(children)
        except Exception as exc:
            report.update(status="failed", cleanup_error=str(exc))
            final_error = exc
        for log in logs:
            log.close()
        for state in report["workers"].values():
            process = next(p for p in children if p.pid == state["pid"])
            state["returncode"] = process.returncode
            if state["status"] == "running":
                state["status"] = "failed" if process.returncode else "execution_finished_unverified"
        report["wall_seconds"] = time.perf_counter() - started
        report["phase"] = "sealed"
        try:
            _atomic(output / "execution_report.json", report)
            if ledger is not None:
                ledger.write(output, report["status"], arm_roots=[output / j["arm_id"] for j in jobs])
            if spec is not None and ledger is not None:
                from molgap.experiment_retention import seal_execution_retention
                seal_execution_retention(output, spec, package_identity=manifest["package_identity"],
                    source_commit=manifest["source_commit"], source_archive_sha256=manifest["archive_sha256"],
                    arm_roots={j["arm_id"]: output / j["arm_id"] for j in jobs})
        except Exception as exc:
            report.update(status="failed", retention_error=str(exc))
            final_error = exc
            _atomic(output / "execution_report.json", report)
        finally:
            _seal_files(output, report, jobs)
        if final_error is not None and sys.exc_info()[0] is None:
            raise RuntimeError("Qualification finalization failed; retained frame identifies the blocker") from final_error


def run_worker(args):
    # This assignment precedes every package import that can import torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # Match CPU metadata's one-thread train-statistics reduction before torch
    # is imported; later Session tuning may allocate its independent CPU share.
    os.environ.update(PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if args.device not in (0, 1):
        raise ValueError("Only the two explicitly assigned T4 devices are admissible")
    config, spec, _, contexts, input_root, authority_path = validate_launch(
        launch_path=args.launch, expected_launch_sha256=args.expected_launch_sha256,
        source_root=args.source_root, package_dir=args.package_dir, mounted_root=args.mounted_root)
    job = next((j for j in config["jobs"] if j["arm_id"] == args.arm), None)
    if job is None or job["device"] != args.device:
        raise ValueError("Worker arm/device differs from pinned launch")
    if time.perf_counter() >= args.allocation_started + config["max_wall_seconds"] - SEAL_RESERVE_SECONDS:
        raise TimeoutError("Whole allocation deadline exhausted before native preflight")
    from molgap.modern2d_qualification import Modern2DQualificationContext
    from molgap.family_training_runner import run_screen_preflight
    context = Modern2DQualificationContext.from_dict(contexts[args.arm])
    certificate = run_screen_preflight(qualification=context, spec=spec,
        package_dir=args.package_dir, expected_package_identity=config["expected_package_identity"],
        arm_id=args.arm, mode="reference", recipe_path=_ref(args.source_root, job["recipe"]),
        initial_state_path=_ref(args.launch.parent, job["initial_state"]), input_root=input_root,
        account=config["account"], run_reference=config["run_reference"], authority_path=authority_path,
        output=args.output)
    _atomic(args.output / "worker_result.json", {"format": "molgap-modern2d-qualification-worker-v1",
        "status": certificate["status"], "context_identity": context.identity,
        "launch_sha256": args.expected_launch_sha256, "device": args.device,
        "certificate_sha256": _sha(args.output / "runtime_certificate.json"),
        "scope": "QUALIFICATION_ONLY", "native_execution_released": False})


def bootstrap(*, launch_path, expected_launch_sha256, allocation_started,
              mounted_root=Path("/kaggle/input"), output=Path("/kaggle/working/qualification"),
              scratch_root=Path("/kaggle/temp/molgap-modern2d-qualification")):
    """Unpack the pinned source using the existing extractor, then supervise."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = read_launch(launch_path, expected_launch_sha256)
    scratch_root = Path(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=False)
    package, source = scratch_root / "package", scratch_root / "source"
    package.mkdir()
    source.mkdir()
    archive = Path(launch_path).parent / "source_payload.bin"
    if _sha(archive) != config["expected_source_archive_sha256"]:
        raise ValueError("Qualification source archive differs from launch")
    shutil.copyfile(archive, package / "source.tar.gz")
    for name in ("experiment_spec.json", "package_manifest.json", "SOURCE_COMMIT.txt",
                 "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
        shutil.copyfile(Path(launch_path).parent / name, package / name)
    with tarfile.open(archive, "r:gz") as bundle:
        matches = [m for m in bundle if m.name == "src/molgap/experiment_preflight.py"]
        if len(matches) != 1 or not matches[0].isfile() or matches[0].linkname:
            raise ValueError("Frozen shared source extractor missing or duplicated")
        with bundle.extractfile(matches[0]) as stream:
            code = stream.read()
    extractor = types.ModuleType("_modern2d_frozen_extractor")
    extractor.__file__ = str(archive) + ":experiment_preflight.py"
    exec(compile(code, extractor.__file__, "exec"), extractor.__dict__)
    extractor._unpack(package, source)
    sys.path.insert(0, str(source / "src"))
    try:
        return run_qualification_pair(source_root=source, package_dir=package,
            mounted_root=mounted_root, launch_path=launch_path, expected_launch_sha256=expected_launch_sha256,
            output=output, runtime_root=scratch_root / "runtime", allocation_started=allocation_started)
    finally:
        # The bootstrap's cost record covers source/extractor failures as well.
        sys.path.remove(str(source / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    environment = commands.add_parser("environment", allow_abbrev=False)
    for field in ("source-root", "runtime-root", "output"):
        environment.add_argument("--" + field, type=Path, required=True)
    environment.add_argument("--deadline", type=float, required=True)
    worker = commands.add_parser("worker", allow_abbrev=False)
    for field in ("source-root", "package-dir", "mounted-root", "launch", "output"):
        worker.add_argument("--" + field, type=Path, required=True)
    worker.add_argument("--expected-launch-sha256", required=True)
    worker.add_argument("--arm", choices=tuple(INTERFACES), required=True)
    worker.add_argument("--device", type=int, choices=(0, 1), required=True)
    worker.add_argument("--allocation-started", type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == "worker":
        try:
            run_worker(args)
        except BaseException:
            _atomic(args.output / "failure.log", {"traceback": traceback.format_exc()})
            raise
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        prepare = runpy.run_path(str(args.source_root / "src/molgap/kaggle_python_environment.py"))["prepare_python"]
        executable, qualified = prepare(args.runtime_root, deadline=args.deadline,
            source_root=args.source_root, profile="family-shared")
        _atomic(args.output / "environment_qualification.json", qualified)
        _atomic(args.output / "environment_result.json", {"executable": str(executable)})


if __name__ == "__main__":
    main()
