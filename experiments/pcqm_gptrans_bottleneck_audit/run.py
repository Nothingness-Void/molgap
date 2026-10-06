"""Thin Kaggle bootstrap for one NO_TRAIN diagnostic, not a trainer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time


def one(name, dataset=None):
    matches = list(Path("/kaggle/input").glob(f"**/{dataset}/train/{name}")) if dataset else list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected one declared input {name}: {matches}")
    return matches[0]


def main():
    started, started_unix = time.perf_counter(), time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("mechanism", "portability"))
    args = parser.parse_args()
    inputs = one("audit_release.json").parent
    root = Path("/kaggle/working/bottleneck_source")
    # Extract only the shared integrity helper before trusting any package import.
    release = json.loads((inputs/"audit_release.json").read_text())
    if hashlib.sha256((inputs/"source_payload.bin").read_bytes()).hexdigest() != release["archive_sha256"]:
        raise ValueError("Mounted archive differs")
    if not args.task:
        inventory = json.loads((inputs/"SOURCE_FILES.json").read_text())
        expected = {row["path"]:row for row in inventory["files"]}
        with tarfile.open(inputs/"source_payload.bin") as archive:
            for member in archive.getmembers():
                if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts or member.linkname:
                    raise ValueError("Unsafe source archive")
                if member.name not in expected or hashlib.sha256(archive.extractfile(member).read()).hexdigest() != expected[member.name]["sha256"]:
                    raise ValueError("Source inventory differs")
            root.mkdir(parents=True, exist_ok=True)
            archive.extractall(root)
    sys.path.insert(0, str(root/"src"))
    from molgap.frozen_audit_runtime import extract_source, run_workers
    from molgap.training_reproducibility import atomic_json
    output = Path("/kaggle/working/gptrans_bottleneck")
    if args.task:
        import traceback
        from molgap.gptrans_bottleneck import run_worker
        directory = output/args.task
        try:
            allocation = json.loads((output/"allocation.json").read_text())
            run_worker(args.task, inputs,
                one("train_shard_0002.pt", "pcqm4mv2-ogb-fixed-100k-v1").parent.parent,
                one("train_shard_0010.pt", "pcqm4mv2-ogb-fixed-500k-scnet-v1").parent.parent,
                directory, allocation["deadline_unix"])
        except BaseException:
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory/"failure.json", dict(error=traceback.format_exc(), training_executed=False))
            raise
    else:
        extract_source(inputs, root)
        run_workers(inputs=inputs, output=output, source_root=root, entry_script=Path(__file__),
            tasks=("mechanism", "portability"), cap_seconds=1800,
            allocation_started=started, started_unix=started_unix)


if __name__ == "__main__":
    main()
