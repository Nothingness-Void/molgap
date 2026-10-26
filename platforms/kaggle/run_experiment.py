"""Standard Kaggle source bootstrap. The staged launch config owns the arms."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import types
import time

# Local preparation replaces this marker and binds the resulting entry bytes.
EXPECTED_LAUNCH_SHA256 = None


def _modern2d_wall_ceiling(declaration, source):
    """Stdlib-only startup admission; native recipe/role checks still run later."""
    arms = declaration["arms"]
    supported = {("ppgt_small", "3"), ("esa_small", "4")}
    if not (1 <= len(arms) <= 2 and all(
            (a["family"]["name"], a["family"]["version"]) in supported for a in arms)):
        return None
    owners = {str(Path(b["plan_spec_ref"]).parent.parent) for b in declaration["prospective"]["arms"]}
    if len(owners) != 1:
        raise ValueError("Modern2D startup requires one frozen authority owner")
    def small_json(relative):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Modern2D metadata must stay inside the frozen source")
        target = source / path
        if not target.resolve().is_relative_to(source.resolve()) or target.stat().st_size > 1024**2:
            raise ValueError("Modern2D startup metadata escaped its bounded source")
        raw = target.read_bytes()
        return raw, json.loads(raw)
    _, plan = small_json(owners.pop() + "/acceptance_plan.json")
    pointer = plan["authority"]
    raw, authority = small_json(pointer["path"])
    if hashlib.sha256(raw).hexdigest() != pointer["sha256"]:
        raise ValueError("Modern2D startup authority hash differs")
    gpu = authority.get("gpu", {})
    wall = gpu.get("wall_ceiling_seconds")
    budget = authority.get("allocated_t4_hours_ceiling", gpu.get("allocated_t4_hours_ceiling"))
    if (authority.get("user_authorized") is not True or type(wall) is not int
            or not 660 <= wall <= 43200 or type(budget) not in {int, float}
            or not math.isfinite(budget) or budget <= 0 or 2 * wall / 3600 > budget
            or gpu.get("allocated_devices") != 2 or gpu.get("accelerator") != "NvidiaTeslaT4"):
        raise ValueError("Modern2D startup requires the frozen full-allocation ceiling")
    purpose = authority.get("purpose")
    if purpose == "CONFIGURED_EXECUTION_PROFILE":
        if (authority.get("scientific_training_authorized") is not False or wall > 1800
                or budget > 1 or len(arms) != 1 or authority.get("scratch_optimizer_updates_per_model") != 6):
            raise ValueError("Profiling startup cannot release scientific training or an expanded allocation")
    elif purpose == "CONFIGURED_EXECUTION_REPAIR":
        if (authority.get("scientific_training_authorized") is not False or wall > 1800
                or budget > 1 or authority.get("scratch_optimizer_updates_per_model") != 8
                or authority.get("execution_reference_required") is not True
                or [(a["family"]["name"], a["family"]["version"]) for a in arms]
                    != [("ppgt_small", "3"), ("esa_small", "4")]):
            raise ValueError("Execution repair startup requires two train-only reference probes within1 T4h")
    elif purpose == "CONFIGURED_TRANSFER_STUDY":
        pointer = authority.get("transfer_study_ref", {})
        if (authority.get("scientific_training_authorized") is not True
                or authority.get("scale_500k_authorized") is not True
                or plan.get("transfer") != pointer or "comparison" in plan
                or set(pointer) != {"path", "sha256"}):
            raise ValueError("Transfer startup requires its explicit500K authority and pinned plan")
        raw, study = small_json(pointer["path"])
        if (hashlib.sha256(raw).hexdigest() != pointer["sha256"]
                or study.get("format") != "molgap-modern2d-transfer-plan-v1"
                or study.get("experiment_purpose") != "transfer_study"
                or set(study.get("arms", {})) != {a["arm_id"] for a in arms}):
            raise ValueError("Transfer startup declaration differs from frozen source")
    elif purpose not in {"CONFIGURED_SCREEN", "FORMAL_100K_TRAINING"} or authority.get("scientific_training_authorized") is not True:
        raise ValueError("Unknown Modern2D startup purpose")
    return wall


def _run_trial_pair(command, *, cwd, env, timeout):
    """The outer allocation cutoff owns the entire nested worker process group."""
    process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        # The coordinator may be killed while its children are checkpointing.
        # Kill the process group even if its leader exits before the grace ends.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        raise


def _workload(allocation_started):
    # The parent verifies CPU artifacts. Each worker declares one physical
    # device before importing its owning trainer; nvidia-smi still sees the full allocation.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    mounted = Path("/kaggle/input")
    configs = list(mounted.rglob("experiment_launch.json"))
    if len(configs) != 1:
        raise RuntimeError("Expected one frozen experiment_launch.json source mount")
    launch = configs[0]
    if EXPECTED_LAUNCH_SHA256 is None or hashlib.sha256(launch.read_bytes()).hexdigest() != EXPECTED_LAUNCH_SHA256:
        raise RuntimeError("Launch configuration differs from the prepared entrypoint")
    config = json.loads(launch.read_text(encoding="utf-8"))
    archive = launch.parent / "source_payload.bin"
    with archive.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != config["expected_source_archive_sha256"]:
        raise RuntimeError("Frozen source archive hash mismatch")
    root = Path("/kaggle/temp/molgap-workflow")
    root.mkdir(parents=True, exist_ok=False)
    package, source = root / "package", root / "source"
    package.mkdir()
    source.mkdir()
    shutil.copyfile(archive, package / "source.tar.gz")
    for name in ("experiment_spec.json", "package_manifest.json", "SOURCE_COMMIT.txt",
                 "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
        shutil.copyfile(launch.parent / name, package / name)
    with tarfile.open(archive, "r:gz") as bundle:
        stream = bundle.extractfile("src/molgap/experiment_preflight.py")
        if stream is None:
            raise RuntimeError("Shared source extractor absent")
        with stream:
            code = stream.read()
    bootstrap = types.ModuleType("_frozen_source_bootstrap")
    bootstrap.__file__ = str(archive) + ":experiment_preflight.py"
    exec(compile(code, bootstrap.__file__, "exec"), bootstrap.__dict__)
    bootstrap._unpack(package, source)
    declaration = json.loads((package / "experiment_spec.json").read_text(encoding="utf-8"))
    arms = declaration["arms"]
    shared_cycle_pair = (len(arms) == 2 and all(a["family"] == {"name": "gptrans_t", "version": "2"}
        and len(a["addons"]) == 1 for a in arms) and
        ({a["addons"][0]["name"] for a in arms} == {"degree_bond_local_ema999", "degree_local_cycle_common_ema999"}
         or all(a["addons"][0]["name"].startswith("degree_aggregate_") for a in arms)))
    k1_comparison = declaration["experiment_id"] == "k1-infrastructure-t4-100k"
    gptrans_comparison = declaration["experiment_id"] == "gptrans-infrastructure-t4-100k"
    modern2d_wall = _modern2d_wall_ceiling(declaration, source)
    modern2d_formal = modern2d_wall is not None
    workflow_trial = declaration["experiment_id"] == "family-workflow-trial-100k"
    if workflow_trial:
        # The source is already byte-verified. Check the small authority before
        # provisioning the pinned runtime; the owning release validator repeats
        # the complete recipe/role checks inside the qualified interpreter.
        plans = {str(Path(b["plan_spec_ref"]).parent.parent).replace("\\", "/")
                 for b in declaration["prospective"]["arms"]}
        if len(plans) != 1:
            raise RuntimeError("Workflow trial has no unique authority owner")
        authority = json.loads((source / plans.pop() / "authority.json").read_bytes())
        if (authority.get("user_authorized") is not True or
                authority.get("purpose") != "WORKFLOW_EXECUTION_TRIAL" or
                authority.get("wall_ceiling_seconds") != 1800 or
                authority.get("allocated_t4_hours_ceiling") != 1):
            raise RuntimeError("Workflow trial allocation is not authorized")
        os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    diagnostic = None
    if gptrans_comparison:
        plan_ref = declaration["prospective"]["arms"][0]["plan_spec_ref"]
        diagnostic_path = source / plan_ref.removesuffix("/original/plan_input.json") / "diagnostic.json"
        if diagnostic_path.is_file():
            diagnostic = json.loads(diagnostic_path.read_bytes())
    if shared_cycle_pair or k1_comparison or gptrans_comparison or modern2d_formal or workflow_trial:
        # Read the stdlib-only owner from the already hash-verified source archive.
        import runpy
        prepare = runpy.run_path(str(source / "src/molgap/kaggle_python_environment.py"))["prepare_python"]
        if workflow_trial:
            print("WORKFLOW_STAGE preparing_pinned_runtime", flush=True)
            remaining = 1800 - (time.perf_counter() - allocation_started) - 120
        elif modern2d_formal:
            remaining = min(1200, modern2d_wall - (time.perf_counter() - allocation_started) - 600)
        else:
            remaining = 1200 if diagnostic is None else diagnostic["wall_ceiling_seconds"] - (time.perf_counter() - allocation_started)
        executable, qualified = prepare(root / "runtime", deadline=time.time() + remaining - 60,
            source_root=source, profile="family-shared" if k1_comparison or modern2d_formal or workflow_trial else "gptrans-shared")
        working = Path("/kaggle/working")
        (working / "environment_qualification.json").write_text(json.dumps(qualified, indent=2), encoding="utf-8")
        if workflow_trial:
            print("WORKFLOW_STAGE native_preflight_then_32_updates", flush=True)
        command = [str(executable), "-m", "molgap.kaggle_pair_runtime",
            "--source-root", str(source), "--package-dir", str(package), "--input-root", str(mounted),
            "--launch", str(launch), "--output", str(working / "experiment"),
            "--allocation-started", repr(allocation_started)]
        environment = dict(os.environ, PYTHONPATH=str(source / "src"))
        if workflow_trial or modern2d_formal:
            _run_trial_pair(command, cwd=source, env=environment,
                timeout=max(1, (1800 if workflow_trial else modern2d_wall)
                            - (time.perf_counter() - allocation_started) - 15))
        else:
            subprocess.run(command, check=True, env=environment,
                timeout=(None if diagnostic is None else max(1, diagnostic["wall_ceiling_seconds"] - (time.perf_counter() - allocation_started) - 10)))
        return
    # Legacy/default bootstrap remains separate from the pinned shared runtime.
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy<2",
                    "torch-geometric==2.6.1", "ogb==1.3.6"], check=True)
    sys.path.insert(0, str(source / "src"))
    from molgap.experiment_package import verify_experiment_source_package
    from molgap.experiment_spec import ExperimentSpec
    from molgap.experiment_launch_config import validate_launch_config
    from molgap.kaggle_pair_runtime import run_two_phase_pair
    manifest = verify_experiment_source_package(package)
    if manifest["package_identity"] != config["expected_package_identity"] or manifest["spec_identity"] != config["spec_identity"]:
        raise RuntimeError("Frozen launch/Spec/package binding mismatch")
    spec = ExperimentSpec.from_json((package / "experiment_spec.json").read_text(encoding="utf-8"))
    validate_launch_config(spec, launch, manifest=manifest, package_dir=package,
                           staged_root=launch.parent)
    run_two_phase_pair(source_root=source, package_dir=package, input_root=mounted,
                       launch_path=launch, output=Path("/kaggle/working/experiment"),
                       allocation_started=allocation_started, manifest=manifest)


def main():
    started = time.perf_counter()
    status, hardware = "failed", []
    workload_error = None
    try:
        hardware = [s.strip() for s in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines() if s.strip()]
        _workload(started)
        status = "complete"
    except BaseException as error:
        workload_error = error
        raise
    finally:
        # Even environment/bootstrap failures retain their full observed allocation.
        # This overlaps the runtime ledger: it must never be added to it again.
        elapsed = time.perf_counter() - started
        destination = Path("/kaggle/working/bootstrap_allocation.json")
        temporary = destination.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps({"status": status, "hardware": hardware,
                "wall_seconds": elapsed, "allocation_device_count": len(hardware) if hardware else None,
                "allocated_device_seconds": len(hardware) * elapsed if hardware else None,
                "cost_status": "measured" if hardware else "measurement_missing",
                "overlap": "Same allocation as experiment/allocation_ledger.json; do not double count",
                "provisioning_before_python_seconds": None, "queue_seconds": None}), encoding="utf-8")
            os.replace(temporary, destination)
        except OSError as error:
            # A receipt failure must not obscure the startup fault that caused it.
            if workload_error is None:
                raise
            print(f"Bootstrap allocation publication failed: {error}", file=sys.stderr)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                print(f"Bootstrap allocation temporary cleanup failed: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
