"""Kaggle CPU prerequisite; source SHA is rendered by the staging adapter."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

EXPECTED_ARCHIVE = "__SOURCE_ARCHIVE_SHA256__"
EXPECTED_COMMIT = "__SOURCE_COMMIT__"


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    began, process_began = time.monotonic(), time.process_time()
    output = Path("/kaggle/working/cycle_inputs")
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
            "torch-geometric==2.6.1", "ogb==1.3.6", "networkx==3.6.1"], check=True)
        archives = [p for p in Path("/kaggle/input").rglob("source_payload.bin")
                    if hashlib.sha256(p.read_bytes()).hexdigest() == EXPECTED_ARCHIVE]
        if len(archives) != 1:
            raise ValueError("Expected one source archive with the frozen identity")
        source, package = Path("/tmp/cycle_source"), Path("/tmp/cycle_package")
        if source.exists() or package.exists():
            raise ValueError("Refuse to reuse an unverified execution directory")
        package.mkdir()
        shutil.copyfile(archives[0], package / "source.tar.gz")
        with tarfile.open(package / "source.tar.gz", "r:gz") as archive:
            bootstrap = archive.extractfile("src/molgap/experiment_preflight.py").read()
        namespace = {"__name__": "cycle_frozen_extractor"}
        exec(compile(bootstrap, "frozen-experiment-preflight.py", "exec"), namespace)
        namespace["_unpack"](package, source)
        inventory = json.loads((archives[0].parent / "SOURCE_FILES.json").read_bytes())
        if inventory["source_commit"] != EXPECTED_COMMIT:
            raise ValueError("Source commit changed")
        for record in inventory["files"]:
            path = namespace["_under"](source, record["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("Source inventory mismatch")
        sys.path.insert(0, str(source / "src"))
        from molgap.gptrans_cycle_inputs import build_sidecar, FIXED_100K_SHA, FIXED_500K_SHA
        from molgap.training_reproducibility import sha256_file
        selections = []
        for digest, roles in ((FIXED_100K_SHA, ("train", "development")),
                              (FIXED_500K_SHA, ("development",))):
            candidates = [p for p in Path("/kaggle/input").rglob("manifest.json")
                          if sha256_file(p) == digest]
            if len(candidates) != 1:
                raise ValueError("Expected one exact accepted fixed manifest")
            selections.append((candidates[0].parent, digest, roles))
        build_sidecar(selections, output, source_commit=EXPECTED_COMMIT,
                      source_archive_sha256=EXPECTED_ARCHIVE,
                      maximum_wall_seconds=max(1, 7200 - (time.monotonic() - began)))
    finally:
        # Cost is execution evidence, never a success marker. Atomic even on error.
        cost = {"wall_seconds": time.monotonic() - began,
                "process_cpu_seconds": time.process_time() - process_began,
                "gpu_device_hours": 0, "allocated_cpu_hours": None,
                "allocated_cpu_hours_status": "measurement_missing",
                "source_commit": EXPECTED_COMMIT, "archive_sha256": EXPECTED_ARCHIVE}
        temporary = output / ".cost.tmp"
        temporary.write_text(json.dumps(cost, indent=2), encoding="utf-8")
        os.replace(temporary, output / "cost.json")


if __name__ == "__main__":
    main()
