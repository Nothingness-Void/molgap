"""Standard Kaggle source bootstrap. The staged launch config owns the arms."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import types
import time
import tempfile

# Local preparation replaces this marker and binds the resulting entry bytes.
EXPECTED_LAUNCH_SHA256 = '62a5e2c4991f19a4d5567728966044ef952488b713ad41c69999e02a077d70ac'
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/temp/molgap-workflow")
OUTPUT_ROOT = Path("/kaggle/working/experiment")


def _atomic_entry_json(path, value):
    """Publish even when setup failed before shared source imports were possible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".allocation-entry-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    allocation_started = time.perf_counter()
    wall_start = datetime.now(timezone.utc).isoformat()
    observation = {"format": "molgap-kaggle-allocation-entry-v1",
                   "status": "running", "error": None,
                   "wall_start_utc": wall_start, "observed_gpu_count": None,
                   "gpu_count_source": None,
                   "scope": "entry invocation through bootstrap completion, including setup, pip, extraction, preflight, training/evaluation and idle assigned GPU time; not device busy time",
                   "platform_release_scope": "not observed: queue, allocation before entry and platform release after entry are excluded",
                   "entire_platform_release_seconds": None}
    state = {"writer": _atomic_entry_json}
    try:
        # Read-only platform metadata, not a CUDA/model probe or recipe declaration.
        try:
            names = [name.strip() for name in subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True, timeout=10).splitlines() if name.strip()]
            if names:
                observation.update(observed_gpu_count=len(names), gpu_count_source="nvidia-smi")
        except (OSError, subprocess.SubprocessError):
            pass
        _bootstrap(allocation_started, state)
        observation["status"] = "complete"
    except BaseException as exc:
        observation.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        elapsed = time.perf_counter() - allocation_started
        count = observation["observed_gpu_count"]
        observation.update(monotonic_seconds=elapsed,
                           wall_end_utc=datetime.now(timezone.utc).isoformat(),
                           allocated_device_seconds=None if count is None else elapsed * count)
        try:
            state["writer"](OUTPUT_ROOT / "allocation_entry_observation.json", observation)
        except Exception as exc:
            if observation["error"] is None:
                raise
            # A publication failure must not replace the original bootstrap error.
            print(f"Allocation entry observation publication failed: {exc}", file=sys.stderr)


def _bootstrap(allocation_started, state):
    mounted = INPUT_ROOT
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
    with tarfile.open(archive, "r:gz") as bundle:
        limits = []
        for job in config["jobs"]:
            stream = bundle.extractfile(job["recipe"])
            if stream is None:
                raise RuntimeError("Frozen arm recipe absent")
            with stream:
                limits.append(json.load(stream).get("allocation_wall_limit_seconds"))
    if any(value is not None and (type(value) is not int or value <= 0) for value in limits):
        raise ValueError("Frozen allocation ceiling must be positive integer seconds or null")
    if len(set(limits)) != 1:
        raise ValueError("Every arm must agree on the allocation ceiling")
    limit = config.get("allocation_wall_limit_seconds", limits[0])
    reserve = config.get("cleanup_reserve_seconds", 60)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Allocation ceiling must be positive integer seconds or null")
    if type(reserve) is not int or reserve < 0:
        raise ValueError("Cleanup reserve must be nonnegative integer seconds")
    if limits[0] is not None and limit != limits[0]:
        raise ValueError("Launch allocation ceiling conflicts with frozen arm recipes")
    if limit is not None and limit <= reserve:
        raise ValueError("Allocation ceiling must exceed cleanup reserve")

    def remaining():
        if limit is None:
            return None
        seconds = limit - (time.perf_counter() - allocation_started) - reserve
        if seconds <= 0:
            raise TimeoutError("Allocation budget exhausted during bootstrap")
        return seconds
    # Dependencies are platform bootstrap, never imported from another experiment.
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy<2",
                    "torch-geometric==2.6.1", "ogb==1.3.6"], check=True, timeout=remaining())
    root = WORK_ROOT
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
    sys.path.insert(0, str(source / "src"))
    from molgap.training_reproducibility import atomic_json
    state["writer"] = atomic_json
    from molgap.experiment_package import verify_experiment_source_package
    from molgap.kaggle_pair_runtime import run_two_phase_pair
    manifest = verify_experiment_source_package(package)
    if manifest["package_identity"] != config["expected_package_identity"] or manifest["spec_identity"] != config["spec_identity"]:
        raise RuntimeError("Frozen launch/Spec/package binding mismatch")
    resolution = {"format": "molgap-kaggle-input-resolution-v1",
                  "spec_identity": config["spec_identity"], "arms": {}}
    for job in config["jobs"]:
        recipe_path = source / job["recipe"]
        record = json.loads(recipe_path.read_text(encoding="utf-8"))
        # Historical screen recipes keep their owning input adapters.
        if record.get("format") != "molgap-family-training-recipe-v1":
            continue
        from molgap.family_training_runner import read_recipe
        from molgap.family_training_data import resolve_training_manifest
        from molgap.experiment_family_workflow import _artifact_path
        _, recipe = read_recipe(recipe_path)
        graph_manifest = resolve_training_manifest(mounted, recipe.dataset)
        graph_metadata = json.loads(graph_manifest.read_text(encoding="utf-8"))
        records = (graph_metadata.get("geometry_shards", []) if
                   graph_metadata.get("format") == "molgap-pcqm4mv2-kaggle-fixed-subset-v1"
                   else graph_metadata.get("assets", {}).get("topology", []))
        roles = {"train", "development"} if recipe.dataset.development else {"train"}
        selected = [row for row in records if row.get("role") in roles]
        if not selected or {row.get("role") for row in selected} != roles:
            raise RuntimeError("Mounted dataset is missing declared train/development metadata")
        for row in selected:
            _artifact_path(graph_manifest.parent, row["file"])
        observed_path = graph_manifest.relative_to(mounted).as_posix()
        resolution["arms"][job["arm_id"]] = {
            "manifest": observed_path, "manifest_sha256": recipe.dataset.manifest_sha256,
            "selected_shard_paths_present": len(selected), "shards_decoded": False}
        print("INPUT_RESOLVED", job["arm_id"], observed_path, flush=True)
    if resolution["arms"]:
        atomic_json(OUTPUT_ROOT / "input_resolution.json", resolution)
    run_two_phase_pair(source_root=source, package_dir=package, input_root=mounted,
                       launch_path=launch, output=OUTPUT_ROOT,
                       maximum_wall_seconds=remaining())


if __name__ == "__main__":
    main()
