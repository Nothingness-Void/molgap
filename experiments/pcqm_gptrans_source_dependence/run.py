"""Thin stdlib bootstrap; frozen audit runtime owns environment/cost/processes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time


def one(name, dataset=None):
    matches=list(Path("/kaggle/input").glob(f"**/{dataset}/train/{name}")) if dataset else list(Path("/kaggle/input").rglob(name))
    if len(matches)!=1:
        raise ValueError(f"Expected one frozen input {name}: {matches}")
    return matches[0]


def main():
    started,unix=time.perf_counter(),time.time()
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=("sources",))
    args=p.parse_args()
    inputs,source=one("audit_release.json").parent,Path("/kaggle/working/source_audit_code")
    if not args.task:
        release=json.loads((inputs/"audit_release.json").read_text())
        archive=inputs/"source_payload.bin"
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=release["archive_sha256"]:
            raise ValueError("Source archive differs")
        entries={r["path"]:r for r in json.loads((inputs/"SOURCE_FILES.json").read_text())["files"]}
        with tarfile.open(archive) as bundle:
            members=bundle.getmembers()
            if len(members)!=len(entries) or {m.name for m in members}!=set(entries):
                raise ValueError("Source inventory differs")
            for m in members:
                if not m.isfile() or Path(m.name).is_absolute() or ".." in Path(m.name).parts or m.linkname or hashlib.sha256(bundle.extractfile(m).read()).hexdigest()!=entries[m.name]["sha256"]:
                    raise ValueError("Unsafe/changed source member")
            source.mkdir(parents=True,exist_ok=True)
            bundle.extractall(source)
    sys.path.insert(0,str(source/"src"))
    from molgap.frozen_audit_runtime import extract_source,run_workers
    from molgap.training_reproducibility import atomic_json
    output=Path("/kaggle/working/gptrans_source_dependence")
    if args.task:
        from molgap.gptrans_source_dependence import run_worker
        import traceback
        try:
            deadline=json.loads((output/"allocation.json").read_text())["deadline_unix"]
            run_worker(inputs,one("train_shard_0002.pt","pcqm4mv2-ogb-fixed-100k-v1").parent.parent,
                one("train_shard_0010.pt","pcqm4mv2-ogb-fixed-500k-scnet-v1").parent.parent,output/args.task,deadline)
        except BaseException:
            atomic_json(output/args.task/"failure.json",dict(error=traceback.format_exc(),training_executed=False))
            raise
    else:
        extract_source(inputs,source)
        run_workers(inputs=inputs,output=output,source_root=source,entry_script=Path(__file__),
            tasks=("sources",),cap_seconds=1800,allocation_started=started,started_unix=unix)


if __name__=="__main__":
    main()
