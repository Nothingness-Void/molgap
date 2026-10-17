"""Pinned Modern2D qualification bootstrap; never submits or trains a screen."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time
import traceback
import types


# Preparation injects the exact launch digest, as for run_experiment.py.
EXPECTED_LAUNCH_SHA256 = None


def main():
    started = time.perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output = Path("/kaggle/working")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "hardware": None,
        "provisioning_before_python_seconds": None, "queue_seconds": None,
        "overlap": "Same allocation as qualification/allocation_ledger.json; do not double count"}
    # The runtime supervises children at 1740s. This outer timer also bounds
    # source bootstrap and synchronous metadata work, leaving final cost IO.
    def deadline_expired(*_):
        raise TimeoutError("Qualification bootstrap allocation hard deadline")
    previous_handler = signal.signal(signal.SIGALRM, deadline_expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 1790 - (time.perf_counter() - started)))
    try:
        report["hardware"] = [line.strip() for line in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5).splitlines() if line.strip()]
        # Only source metadata at direct mount roots is inspected. Data selection
        # is an explicit path in the digest-bound launch, never a recursive scan.
        candidates = [root / "modern2d_qualification_launch.json"
            for root in Path("/kaggle/input").iterdir() if root.is_dir()
            and (root / "modern2d_qualification_launch.json").is_file()]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one direct mounted qualification launch")
        launch = candidates[0]
        raw = launch.read_bytes()
        if EXPECTED_LAUNCH_SHA256 is None or hashlib.sha256(raw).hexdigest() != EXPECTED_LAUNCH_SHA256:
            raise ValueError("Qualification launch differs from prepared entrypoint")
        config = json.loads(raw)
        archive = launch.parent / "source_payload.bin"
        with archive.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != config["expected_source_archive_sha256"]:
                raise ValueError("Qualification source archive hash mismatch")
        with tarfile.open(archive, "r:gz") as bundle:
            matches = [m for m in bundle if m.name == "src/molgap/modern2d_qualification_runtime.py"]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].linkname:
                raise ValueError("Frozen qualification runtime missing or duplicated")
            with bundle.extractfile(matches[0]) as stream:
                code = stream.read()
        runtime = types.ModuleType("_frozen_modern2d_qualification")
        runtime.__file__ = str(archive) + ":modern2d_qualification_runtime.py"
        exec(compile(code, runtime.__file__, "exec"), runtime.__dict__)
        result = runtime.bootstrap(launch_path=launch, expected_launch_sha256=EXPECTED_LAUNCH_SHA256,
            allocation_started=started)
        report.update(status=result["status"], hardware=result.get("hardware"))
    except BaseException:
        report["failure"] = traceback.format_exc()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
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
            allocated_device_seconds=count * elapsed if count is not None else None,
            cost_status="measured" if count is not None else "measurement_missing")
        temporary = output / "bootstrap_allocation.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / "bootstrap_allocation.json")


if __name__ == "__main__":
    main()
