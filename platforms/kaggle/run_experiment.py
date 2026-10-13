"""Standard Kaggle source bootstrap. The staged launch config owns the arms."""
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

# Local preparation replaces this marker and binds the resulting entry bytes.
EXPECTED_LAUNCH_SHA256 = None


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
    if shared_cycle_pair:
        # Read the stdlib-only owner from the already hash-verified source archive.
        import runpy
        prepare = runpy.run_path(str(source / "src/molgap/kaggle_python_environment.py"))["prepare_python"]
        executable, qualified = prepare(root / "runtime", deadline=time.time() + 1200,
            source_root=source, profile="gptrans-shared")
        working = Path("/kaggle/working")
        (working / "environment_qualification.json").write_text(json.dumps(qualified, indent=2), encoding="utf-8")
        subprocess.run([str(executable), "-m", "molgap.kaggle_pair_runtime",
            "--source-root", str(source), "--package-dir", str(package), "--input-root", str(mounted),
            "--launch", str(launch), "--output", str(working / "experiment"),
            "--allocation-started", repr(allocation_started)], check=True,
            env=dict(os.environ, PYTHONPATH=str(source / "src")))
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
    try:
        hardware = [s.strip() for s in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines() if s.strip()]
        _workload(started)
        status = "complete"
    finally:
        # Even environment/bootstrap failures retain their full observed allocation.
        # This overlaps the runtime ledger: it must never be added to it again.
        elapsed = time.perf_counter() - started
        destination = Path("/kaggle/working/bootstrap_allocation.json")
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps({"status": status, "hardware": hardware,
            "wall_seconds": elapsed, "allocation_device_count": len(hardware) if hardware else None,
            "allocated_device_seconds": len(hardware) * elapsed if hardware else None,
            "cost_status": "measured" if hardware else "measurement_missing",
            "overlap": "Same allocation as experiment/allocation_ledger.json; do not double count",
            "provisioning_before_python_seconds": None, "queue_seconds": None}), encoding="utf-8")
        os.replace(temporary, destination)


if __name__ == "__main__":
    main()
