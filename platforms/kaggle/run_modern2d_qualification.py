"""Pinned Modern2D qualification bootstrap; never submits or trains a screen."""
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import tarfile
import time
import traceback
import types


# Preparation injects these from the verified source package and launch.
EXPECTED_LAUNCH_SHA256 = None
EXPECTED_SOURCE_DATASET = None
MOUNT_RESOLVER_SOURCE = None


def _regular_input(path):
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Links are forbidden in qualification bootstrap inputs")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Expected regular qualification bootstrap input")
    return path


def main():
    started = time.perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output = Path("/kaggle/working")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "phase": "deadline_setup", "hardware": None,
        "source_discovery": {},
        "measurement_scope": "PYTHON_OBSERVED_WINDOW_ONLY",
        "full_allocation_seconds": None, "full_allocation_device_seconds": None,
        "full_allocation_cost": None,
        "provisioning_before_python_seconds": None, "queue_seconds": None,
        "overlap": "Same allocation as qualification/allocation_ledger.json; do not double count"}
    # The runtime supervises children at 1740s. This outer timer also bounds
    # source bootstrap and synchronous metadata work, leaving final cost IO.
    def deadline_expired(*_):
        raise TimeoutError("Qualification bootstrap allocation hard deadline")
    handler_installed = timer_attempted = False
    original_failure = False
    try:
        previous_handler = signal.signal(signal.SIGALRM, deadline_expired)
        handler_installed = True
        timer_attempted = True
        signal.setitimer(signal.ITIMER_REAL, max(.001, 1790 - (time.perf_counter() - started)))
        report["phase"] = "hardware_observation"
        report["hardware"] = [line.strip() for line in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5).splitlines() if line.strip()]
        report["phase"] = "source_discovery"
        if type(MOUNT_RESOLVER_SOURCE) is not str or not MOUNT_RESOLVER_SOURCE:
            raise ValueError("Prepared mount resolver source is required")
        resolver = types.ModuleType("_prepared_kaggle_mounts")
        resolver.__file__ = "<prepared src/molgap/kaggle_mounts.py>"
        # This source is trusted as part of the prepared entry, not imported
        # from an unverified mounted archive or the executing checkout.
        exec(compile(MOUNT_RESOLVER_SOURCE, resolver.__file__, "exec"), resolver.__dict__)
        source_root = resolver.resolve_dataset_mount(Path("/kaggle/input"),
            EXPECTED_SOURCE_DATASET, marker="modern2d_qualification_launch.json",
            report=report["source_discovery"])
        report["phase"] = "launch_verification"
        launch = _regular_input(source_root / "modern2d_qualification_launch.json")
        with launch.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("Qualification launch exceeds one MiB")
        if EXPECTED_LAUNCH_SHA256 is None or hashlib.sha256(raw).hexdigest() != EXPECTED_LAUNCH_SHA256:
            raise ValueError("Qualification launch differs from prepared entrypoint")
        config = json.loads(raw)
        report["phase"] = "archive_verification"
        archive = _regular_input(launch.parent / "source_payload.bin")
        with archive.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != config["expected_source_archive_sha256"]:
                raise ValueError("Qualification source archive hash mismatch")
        with tarfile.open(archive, "r:gz") as bundle:
            report["phase"] = "resolver_source_verification"
            helpers = [m for m in bundle if m.name == "src/molgap/kaggle_mounts.py"]
            if len(helpers) != 1 or not helpers[0].isfile() or helpers[0].linkname:
                raise ValueError("Frozen mount resolver missing or duplicated")
            if helpers[0].size > 1024 * 1024:
                raise ValueError("Frozen mount resolver exceeds one MiB")
            with bundle.extractfile(helpers[0]) as stream:
                if stream.read(1024 * 1024) != MOUNT_RESOLVER_SOURCE.encode("utf-8"):
                    raise ValueError("Frozen mount resolver differs from prepared entrypoint")
            report["phase"] = "runtime_source_verification"
            matches = [m for m in bundle if m.name == "src/molgap/modern2d_qualification_runtime.py"]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].linkname:
                raise ValueError("Frozen qualification runtime missing or duplicated")
            with bundle.extractfile(matches[0]) as stream:
                code = stream.read()
        report["phase"] = "runtime_initialization"
        runtime = types.ModuleType("_frozen_modern2d_qualification")
        runtime.__file__ = str(archive) + ":modern2d_qualification_runtime.py"
        exec(compile(code, runtime.__file__, "exec"), runtime.__dict__)
        report["phase"] = "runtime_bootstrap"
        result = runtime.bootstrap(launch_path=launch, expected_launch_sha256=EXPECTED_LAUNCH_SHA256,
            allocation_started=started)
        report.update(status=result["status"], hardware=result.get("hardware") or report["hardware"])
        report["phase"] = "runtime_returned"
    except BaseException:
        original_failure = True
        report["failure"] = traceback.format_exc()
        raise
    finally:
        # Setup or cleanup can fail on mocked/non-POSIX hosts. Neither may
        # bypass the atomic failure/cost record or mask the original failure.
        cleanup_errors = []
        if timer_attempted:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except BaseException:
                cleanup_errors.append(traceback.format_exc())
        if handler_installed:
            try:
                signal.signal(signal.SIGALRM, previous_handler)
            except BaseException:
                cleanup_errors.append(traceback.format_exc())
        if cleanup_errors:
            report["deadline_cleanup_errors"] = cleanup_errors
            report.update(status="failed", original_phase=report["phase"], phase="deadline_cleanup")
        # Even an environment/bootstrap failure retains measured and unknown cost.
        ledger = output / "qualification/allocation_ledger.json"
        if ledger.is_file():
            try:
                observed = json.loads(ledger.read_bytes())
                report["hardware"] = [d["hardware"] for d in observed["devices"]]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                report["ledger_read_error"] = str(exc)
        elapsed = time.perf_counter() - started
        count = len(report["hardware"]) if report["hardware"] else None
        report.update(wall_seconds=elapsed, allocation_device_count=count,
            allocated_device_seconds=None,
            python_observed_wall_seconds=elapsed,
            python_observed_device_seconds=count * elapsed if count is not None else None,
            hardware_measurement_status="observed" if count is not None else "measurement_missing",
            cost_status="PYTHON_OBSERVED_WINDOW_ONLY")
        temporary = output / "bootstrap_allocation.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / "bootstrap_allocation.json")
        if cleanup_errors and not original_failure:
            raise RuntimeError("Qualification deadline cleanup failed; see bootstrap_allocation.json")


if __name__ == "__main__":
    main()
