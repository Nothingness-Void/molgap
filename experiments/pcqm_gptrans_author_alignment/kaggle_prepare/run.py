"""CPU-only preparation of accepted fixed-graph path inputs and G1 tensors."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tarfile
import time
import traceback

OUTPUT = Path("/kaggle/working/gptrans_author_inputs")
MANIFEST_SHA = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
INITIAL_SHA = "9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c"
EXPECTED_SOURCE_SHA256 = "__PIN_SOURCE_ARCHIVE_SHA256__"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def mounted(name, expected):
    matches = [p for p in Path("/kaggle/input").rglob(name) if digest(p) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"Ambiguous or missing frozen mount: {name}")
    return matches[0]


def main():
    started = time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                           "torch-geometric==2.6.1", "ogb==1.3.6"])
    # A private immutable source mount supplies the archive SHA via its inventory.
    candidates = []
    for archive in Path("/kaggle/input").rglob("source_payload.bin"):
        sidecar = archive.parent / "SOURCE_ARCHIVE_SHA256.txt"
        inventory = archive.parent / "SOURCE_FILES.json"
        if not sidecar.is_file() or not inventory.is_file():
            continue
        if digest(archive) != EXPECTED_SOURCE_SHA256 or sidecar.read_text().strip() != EXPECTED_SOURCE_SHA256:
            continue
        files = {item["path"] for item in json.loads(inventory.read_text())["files"]}
        if "src/molgap/gptrans_author_inputs.py" in files:
            candidates.append(archive)
    if len(candidates) != 1:
        raise RuntimeError("Expected one verified author-input source package")
    archive = candidates[0]
    source = OUTPUT / "verified_source"
    source.mkdir()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise RuntimeError("Unsafe source archive member")
            target = source / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as stream, target.open("xb") as output:
                import shutil
                shutil.copyfileobj(stream, output)
    sys.path.insert(0, str(source / "src"))
    from molgap.gptrans_author_inputs import build_sidecar, validate_sidecar, prepare_degree_initial_state
    from molgap.training_reproducibility import atomic_json
    manifest = mounted("manifest.json", MANIFEST_SHA)
    initial = mounted("initial_state.pt", INITIAL_SHA)
    atomic_json(OUTPUT / "startup.json", {
        "source_archive_sha256": digest(archive),
        "source_commit": (archive.parent / "SOURCE_COMMIT.txt").read_text().strip(),
        "manifest_sha256": digest(manifest), "initial_state_file_sha256": digest(initial),
        "training_executed": False, "model_inference_executed": False,
        "labels_read": False, "official_validation_role_read": False,
        "test_dev_role_read": False, "test_challenge_role_read": False,
    })
    degree = prepare_degree_initial_state(initial, OUTPUT / "degree_initial_state.pt")
    build_sidecar(manifest.parent, OUTPUT / "paths")
    accepted = validate_sidecar(OUTPUT / "paths", dataset_root=manifest.parent, rederive=True)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    atomic_json(OUTPUT / "preparation_result.json", {
        "degree_initialization": degree, "path_acceptance": accepted,
        "wall_seconds": time.monotonic() - started,
        "process_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "allocated_cpu_count": os.cpu_count(), "allocated_gpu_count": 0,
        "training_executed": False, "model_inference_executed": False,
        "labels_read": False, "official_validation_role_read": False,
        "test_dev_role_read": False, "test_challenge_role_read": False,
    })


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def deadline(_signal, _frame):
        raise TimeoutError("CPU preparation exceeded its three-hour wall budget")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(3 * 60 * 60)
    status = "FAILED"
    try:
        main()
        status = "COMPLETE"
    except BaseException:
        failure = OUTPUT / "failure.json.tmp"
        failure.write_text(json.dumps({"status": status, "traceback": traceback.format_exc(),
                                     "training_executed": False, "model_inference_executed": False}))
        os.replace(failure, OUTPUT / "failure.json")
        raise
    finally:
        signal.alarm(0)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cost = OUTPUT / "native_cost.json.tmp"
        cost.write_text(json.dumps({"status": status, "wall_seconds": time.monotonic() - started,
                                  "process_cpu_seconds": usage.ru_utime + usage.ru_stime,
                                  "allocated_cpu_count": os.cpu_count(), "allocated_gpu_count": 0}))
        os.replace(cost, OUTPUT / "native_cost.json")
