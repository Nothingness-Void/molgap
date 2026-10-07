"""Frozen-source bootstrap for the owning bounded 500K trainer (no submission)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import types
import time
import threading
from datetime import datetime, timezone

EXPECTED_LAUNCH_SHA256 = '5948fefd3d22661b38dfc668ac90678201f4e15481485d9caf603f9b4e345971'
WINDOWS = {}
STARTED = None
ALLOCATION_VERIFIED = False
CPU_CHECK = '''
import json, sys
from pathlib import Path
from molgap.pcqm_k1_scale_runner import find_cache, load_roles
from molgap.pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
from molgap.pcqm_composed_500k import make_model, scientific_contract, target_statistics
launch = Path(sys.argv[1])
config = json.loads(launch.read_text())
root, manifest = find_cache(FIXED_500K_MANIFEST_SHA256)
if config['graph_mount'] not in root.relative_to(Path('/kaggle/input')).parts:
    raise RuntimeError('Accepted graph identity mounted under unexpected dataset')
roles = load_roles(root, manifest)
for arm in config['arms']:
    model = make_model(arm['arm_id'], launch.parent / arm['initial_state'])
    del model
    expected = scientific_contract(arm['arm_id'])
    recipe = json.loads((Path(sys.argv[2]) / arm['recipe']).read_text())
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Frozen recipe differs from owning trainer')
    mean, std = target_statistics(arm['arm_id'], roles['train'], launch.parent / config['target_transform'])
    if not __import__('math').isfinite(mean) or not __import__('math').isfinite(std) or std <= 0:
        raise RuntimeError('Target statistics invalid')
print('ALL_ARM_CPU_MODEL_CACHE_RECIPE_PASS', flush=True)
'''


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def start_logged(command, path, *, label, env=None):
    """Retain worker output and forward each flushed line to the platform log."""
    log = Path(path).open("w", encoding="utf-8", buffering=1)
    try:
        worker = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
    except BaseException:
        log.close()
        raise
    def forward():
        try:
            for line in worker.stdout:
                log.write(line)
                print(f"[{label}] {line.rstrip()}", flush=True)
        finally:
            worker.stdout.close()
            log.close()
    thread = threading.Thread(target=forward, name=label, daemon=True)
    thread.start()
    return worker, thread


def resolve_resume(mounted, arm):
    if "resume" not in arm:
        return None
    resume = arm["resume"]
    manifests = [p for p in Path(mounted).rglob("stage_manifest.json")
                 if resume["mount"] in p.relative_to(mounted).parts
                 and p.parent.name == arm["arm_id"]]
    if len(manifests) != 1 or digest(manifests[0]) != resume["manifest_sha256"]:
        raise RuntimeError(f"Pinned resume manifest changed: {arm['arm_id']}")
    manifest = json.loads(manifests[0].read_text())
    if (manifest["arm"] != arm["arm_id"] or manifest["source_sha256"] != resume["source_sha256"]
            or manifest["next_epoch"] != resume["next_epoch"]):
        raise RuntimeError(f"Resume cursor/source changed: {arm['arm_id']}")
    root = manifests[0].parent
    for name, checksum in manifest["artifacts"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != checksum:
            raise RuntimeError(f"Resume artifact changed: {arm['arm_id']}/{name}")
    print(f"RESUME VERIFIED {arm['arm_id']} completed_epochs={resume['next_epoch']}", flush=True)
    return root


def retain_completed_arm(resume_root, destination, arm, stage_epochs=60):
    """Keep a finished peer's original evidence without another GPU worker."""
    resume = arm.get("resume", {})
    if resume.get("next_epoch") != stage_epochs:
        return False
    if resume_root is None:
        raise RuntimeError("Completed arm requires verified retained output")
    root = Path(resume_root)
    if digest(root / "stage_manifest.json") != resume["manifest_sha256"]:
        raise RuntimeError("Completed arm manifest changed after resume validation")
    manifest = json.loads((root / "stage_manifest.json").read_text())
    if (manifest["status"] != "COMPLETE" or manifest["next_epoch"] != stage_epochs
            or manifest["arm"] != arm["arm_id"] or manifest["source_sha256"] != resume["source_sha256"]):
        raise RuntimeError("Completed arm evidence disagrees with the pinned resume")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, checksum in manifest["artifacts"].items():
        source = root / name
        target = destination / name
        if not source.resolve().is_relative_to(root.resolve()) or not target.resolve().is_relative_to(destination.resolve()):
            raise RuntimeError("Completed artifact escapes its retained directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != checksum:
            raise RuntimeError("Completed artifact changed during retention: " + name)
    shutil.copyfile(root / "stage_manifest.json", destination / "stage_manifest.json")
    print(f"RETAIN COMPLETE {arm['arm_id']} epochs={stage_epochs}; no GPU worker", flush=True)
    return True


def _main():
    global ALLOCATION_VERIFIED
    mounted = Path("/kaggle/input")
    launches = list(mounted.rglob("legacy_500k_launch.json"))
    if len(launches) != 1 or digest(launches[0]) != EXPECTED_LAUNCH_SHA256:
        raise RuntimeError("Expected one independently pinned legacy500K launch")
    launch = launches[0]
    config = json.loads(launch.read_text(encoding="utf-8"))
    graph_mounts = [p for p in mounted.rglob(config["graph_mount"]) if p.is_dir()]
    if config["source_mount"] not in launch.relative_to(mounted).parts or len(graph_mounts) != 1:
        raise RuntimeError("Frozen source/graph dataset mounts changed")
    if config["format"] != "molgap-legacy-500k-pair-v1":
        raise RuntimeError("Unsupported legacy500K launch")
    if not 0 < config["max_stage_seconds"] <= 32400:
        raise RuntimeError("Stage exceeds the approved 9h bound")
    devices = [a["device"] for a in config["arms"]]
    if devices != [0, 1] or config["stage_epochs"] != 60:
        raise RuntimeError("Expected two isolated 60-pass arms")
    archive = launch.parent / "source_payload.bin"
    if digest(archive) != config["source_archive_sha256"]:
        raise RuntimeError("Source archive changed")
    names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()
    if len(names) != 2 or any("T4" not in name for name in names):
        raise RuntimeError(f"Expected T4x2 allocation: {names}")
    ALLOCATION_VERIFIED = True
    print(f"ALLOCATION VERIFIED {names}; installing frozen runtime", flush=True)
    constraints = []
    runtime_requirements = config.get("runtime_distributions")
    if any("resume" in arm for arm in config["arms"]) and not runtime_requirements:
        raise RuntimeError("Continuation requires the retained runtime distribution pins")
    if runtime_requirements:
        requirements = Path("/kaggle/temp/molgap-legacy500k-constraints.txt")
        requirements.write_text("\n".join(runtime_requirements) + "\n", encoding="utf-8")
        constraints = ["-c", str(requirements)]
    # Torch imports happen only after installing the same frozen legacy runtime.
    subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu121"] + constraints, check=True, timeout=1200)
    dependencies = ([value for value in runtime_requirements if not value.startswith(("torch==", "nvidia-", "triton=="))]
                    if runtime_requirements else ["numpy<2", "torch-geometric==2.6.1", "ogb==1.3.6"])
    subprocess.run([sys.executable, "-m", "pip", "install"] + dependencies + constraints, check=True, timeout=1200)
    root = Path("/kaggle/temp/molgap-legacy500k")
    root.mkdir(parents=True, exist_ok=False)
    package, source = root / "package", root / "source"
    package.mkdir()
    source.mkdir()
    shutil.copyfile(archive, package / "source.tar.gz")
    for name in ("experiment_spec.json", "package_manifest.json", "SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
        shutil.copyfile(launch.parent / name, package / name)
    with tarfile.open(archive, "r:gz") as bundle:
        stream = bundle.extractfile("src/molgap/experiment_preflight.py")
        if stream is None:
            raise RuntimeError("Shared frozen extractor missing")
        code = stream.read()
    bootstrap = types.ModuleType("_frozen_500k_bootstrap")
    bootstrap.__file__ = str(archive) + ":experiment_preflight.py"
    exec(compile(code, bootstrap.__file__, "exec"), bootstrap.__dict__)
    bootstrap._unpack(package, source)
    sys.path.insert(0, str(source / "src"))
    from molgap.experiment_package import verify_experiment_source_package
    from molgap.experiment_spec import ExperimentSpec
    from molgap.experiment_preflight import check_release_inputs
    from molgap.screen_policy import canonical_fingerprint
    manifest = verify_experiment_source_package(package)
    spec = ExperimentSpec.from_json((package / "experiment_spec.json").read_text(encoding="utf-8"))
    if (manifest["package_identity"] != config["package_identity"] or spec.identity != config["spec_identity"]
            or manifest["source_commit"] != config["source_commit"]):
        raise RuntimeError("Launch/source/Spec binding changed")
    declared = {a["arm_id"]: a for a in spec.to_dict()["arms"]}
    inputs, recipes, resumes = {}, {}, {}
    for arm in config["arms"]:
        aid = arm["arm_id"]
        if aid not in declared or canonical_fingerprint(declared[aid]) != arm["source_config_identity"]:
            raise RuntimeError("Arm configuration changed")
        for field in ("initial_state", "binding", "trajectory"):
            path = launch.parent / arm[field]
            if not path.resolve().is_relative_to(launch.parent.resolve()) or digest(path) != arm[field + "_sha256"]:
                raise RuntimeError(f"Pinned {aid}/{field} changed")
        trajectory = json.loads((launch.parent / arm["trajectory"]).read_text(encoding="utf-8"))
        binding = json.loads((launch.parent / arm["binding"]).read_text(encoding="utf-8"))
        if (trajectory["record_mode"] != "prospective" or trajectory["trajectory_id"] != arm["trajectory_id"]
                or trajectory["state_at_start"]["source_config_identity"] != arm["source_config_identity"]
                or binding != {"spec_identity": spec.identity, "trajectory_id": arm["trajectory_id"], "source_config_identity": arm["source_config_identity"]}):
            raise RuntimeError("Prospective identity mismatch")
        inputs[aid] = launch.parent / arm["initial_state"]
        recipes[aid] = arm["recipe"]
        resumes[aid] = resolve_resume(mounted, arm)
    transform = launch.parent / config["target_transform"]
    if not transform.resolve().is_relative_to(launch.parent.resolve()) or digest(transform) != config["target_transform_sha256"]:
        raise RuntimeError("Pinned target transform changed")
    output = Path("/kaggle/working/evidence")
    output.mkdir(parents=True, exist_ok=False)
    state = {"format": config["format"], "spec_identity": spec.identity, "package_identity": manifest["package_identity"],
             "source_sha256": config["source_archive_sha256"], "training_started": False, "phase": "cpu_release", "arms": {}}
    active_arms = []
    state["reused_arms"] = {}
    for arm in config["arms"]:
        aid = arm["arm_id"]
        if retain_completed_arm(resumes[aid], output / "stages" / aid, arm, config["stage_epochs"]):
            state["reused_arms"][aid] = dict(arm["resume"], disposition="RETAINED_COMPLETE_NO_EXECUTION")
        else:
            active_arms.append(arm)
    atomic(output / "pair_state.json", state)
    report = check_release_inputs(spec, package, expected_package_identity=manifest["package_identity"],
        recipe_files=recipes, initial_states=inputs, required_modules=config["required_modules"], input_root=launch.parent)
    atomic(output / "cpu_release_report.json", report)
    if report["errors"]:
        raise RuntimeError("All-arm CPU release validation failed")
    cpu_env = os.environ.copy()
    cpu_env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(source / "src"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    print("PHASE cpu_model_cache_recipe: loading accepted training/development cache", flush=True)
    cpu_worker, cpu_thread = start_logged([sys.executable, "-u", "-c", CPU_CHECK, str(launch), str(source)],
        output / "cpu_model_cache_recipe.log", label="cpu", env=cpu_env)
    try:
        cpu_code = cpu_worker.wait(timeout=1200)
    finally:
        if cpu_worker.poll() is None:
            cpu_worker.terminate()
            cpu_worker.wait()
        cpu_thread.join()
    if cpu_code:
        raise RuntimeError(f"CPU model/cache/recipe check failed: {cpu_code}")
    for phase in ("preflight", "training"):
        WINDOWS[phase] = {"start_monotonic": time.monotonic(), "stop_monotonic": None}
        state["phase"] = phase
        state["training_started"] = phase == "training" and bool(active_arms)
        atomic(output / "pair_state.json", state)
        workers = []
        threads = []
        print(f"PHASE {phase}: active arms={[a['arm_id'] for a in active_arms]}", flush=True)
        try:
            for arm in active_arms:
                aid = arm["arm_id"]
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(arm["device"]), PYTHONPATH=str(source / "src"), PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MOLGAP_V4_LOADER_WORKERS="2")
                command = [sys.executable, "-u", "-m", "molgap.pcqm_500k_v4_evidence", "--arm", aid,
                    "--output", str(output / ("qualification" if phase == "preflight" else "stages") / aid),
                    "--source-sha", config["source_archive_sha256"], "--stage-epochs", str(config["stage_epochs"]),
                    "--max-stage-seconds", str(max(1, int(config["max_stage_seconds"] - (time.monotonic() - STARTED)))), "--platform-id", "kaggle1-t4x2",
                    "--initial-state", str(inputs[aid]), "--initial-state-sha256", arm["initial_state_sha256"],
                    "--binding-identity", str(launch.parent / arm["binding"])]
                if aid == "gptrans_g1_bond_local_ema999":
                    command += ["--target-transform", str(transform)]
                if resumes[aid] is not None:
                    command += ["--resume", str(resumes[aid]), "--resume-source-sha", arm["resume"]["source_sha256"]]
                if phase == "preflight":
                    command += ["--preflight-only"]
                worker, thread = start_logged(command, output / f"{phase}_{aid}.log", label=f"{phase}/{aid}", env=env)
                threads.append(thread)
                workers.append((aid, worker))
            last_report = time.monotonic()
            while any(worker.poll() is None for _, worker in workers):
                if time.monotonic() - last_report >= 30:
                    elapsed = int(time.monotonic() - WINDOWS[phase]["start_monotonic"])
                    print(f"PHASE {phase} elapsed={elapsed}s worker_codes={[(aid, w.poll()) for aid, w in workers]}", flush=True)
                    last_report = time.monotonic()
                if phase == "preflight" and time.monotonic() - WINDOWS[phase]["start_monotonic"] > 1200:
                    raise TimeoutError("All-arm GPU preflight exceeded 1200s")
                time.sleep(1)
            codes = {aid: worker.returncode for aid, worker in workers}
        finally:
            for _, worker in workers:
                if worker.poll() is None:
                    worker.terminate()
                    worker.wait()
            for thread in threads:
                thread.join()
            WINDOWS[phase]["stop_monotonic"] = time.monotonic()
        state["arms"] = codes
        atomic(output / "pair_state.json", state)
        if any(codes.values()):
            raise RuntimeError(f"{phase} failed: {codes}")
    state["phase"] = "bounded_stages_published"
    atomic(output / "pair_state.json", state)


def main():
    global STARTED
    STARTED = float(os.environ.get("MOLGAP_BOOTSTRAP_STARTED", time.monotonic()))
    started_utc = os.environ.get("MOLGAP_BOOTSTRAP_UTC", datetime.now(timezone.utc).isoformat())
    succeeded = False
    try:
        print(f"BOOTSTRAP START python={sys.version.split()[0]}", flush=True)
        if sys.version_info[:2] != (3, 11):
            # The owning Torch2.4.1 recipe has no Python3.13 wheel. Reuse uv's
            # interpreter/venv management; preserve the full bootstrap cost clock.
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
            environment = Path("/kaggle/temp/molgap-legacy500k-python311")
            subprocess.run([sys.executable, "-m", "uv", "--system-certs", "venv", "--python", "3.11.17", "--seed", str(environment)], check=True, timeout=600)
            os.environ.update(MOLGAP_BOOTSTRAP_STARTED=str(STARTED), MOLGAP_BOOTSTRAP_UTC=started_utc)
            python = str(environment / "bin/python")
            os.execv(python, [python, "-u", str(Path(__file__).resolve())])
        _main()
        succeeded = True
    finally:
        stopped = time.monotonic()
        output = Path("/kaggle/working/evidence")
        output.mkdir(parents=True, exist_ok=True)
        windows = {name: {"elapsed_wall_seconds": (entry["stop_monotonic"] or stopped) - entry["start_monotonic"],
            "allocated_T4_seconds": 2 * ((entry["stop_monotonic"] or stopped) - entry["start_monotonic"])}
            for name, entry in WINDOWS.items()}
        atomic(output / "invocation_cost.json", {"format": "molgap-kaggle-allocated-cost-v1", "started_utc": started_utc,
            "stopped_utc": datetime.now(timezone.utc).isoformat(), "process_succeeded": succeeded,
            "allocation_verified": ALLOCATION_VERIFIED, "hardware": "T4x2" if ALLOCATION_VERIFIED else "unknown",
            "scope": "observed bootstrap process window including install, CPU checks, qualification, training and idle peer capacity; scheduler startup outside this process is missing",
            "elapsed_wall_seconds": stopped - STARTED, "allocated_T4_seconds": 2 * (stopped - STARTED) if ALLOCATION_VERIFIED else None,
            "phase_windows": windows, "scheduler_queue_seconds": None, "cpu_core_seconds": None})


if __name__ == "__main__":
    main()
