"""Only bootstrap the hash-bound frozen portability owner; never train."""
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
    started, unix = time.perf_counter(), time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("portability",))
    args = parser.parse_args()
    inputs, root = one("audit_release.json").parent, Path("/kaggle/working/triplet_audit_source")
    if not args.task:
        release = json.loads((inputs / "audit_release.json").read_text())
        archive_path = inputs / "source_payload.bin"
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != release["archive_sha256"]:
            raise ValueError("Mounted source archive differs")
        entries = {r["path"]: r for r in json.loads((inputs / "SOURCE_FILES.json").read_text())["files"]}
        with tarfile.open(archive_path) as archive:
            members = archive.getmembers()
            if len(members) != len(entries) or {m.name for m in members} != set(entries):
                raise ValueError("Source inventory differs")
            for member in members:
                if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts or member.linkname:
                    raise ValueError("Unsafe source member")
                if hashlib.sha256(archive.extractfile(member).read()).hexdigest() != entries[member.name]["sha256"]:
                    raise ValueError("Source member differs")
            root.mkdir(parents=True, exist_ok=True)
            archive.extractall(root)
    sys.path.insert(0, str(root / "src"))
    from molgap.frozen_audit_runtime import extract_source, run_workers
    from molgap.training_reproducibility import atomic_json
    output = Path("/kaggle/working/gptrans_triplet_portability")
    if args.task:
        import traceback
        from molgap.gptrans_triplet_portability import run_worker
        directory = output / args.task
        try:
            deadline = json.loads((output / "allocation.json").read_text())["deadline_unix"]
            run_worker(inputs, one("train_shard_0002.pt", "pcqm4mv2-ogb-fixed-100k-v1").parent.parent,
                one("train_shard_0010.pt", "pcqm4mv2-ogb-fixed-500k-scnet-v1").parent.parent, directory, deadline)
        except BaseException:
            atomic_json(directory / "failure.json", dict(error=traceback.format_exc(), training_executed=False))
            raise
    else:
        extract_source(inputs, root)
        run_workers(inputs=inputs, output=output, source_root=root, entry_script=Path(__file__),
            tasks=("portability",), cap_seconds=1800, allocation_started=started, started_unix=unix)


if __name__ == "__main__":
    main()
