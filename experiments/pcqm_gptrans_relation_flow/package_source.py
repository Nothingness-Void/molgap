"""Package committed source and an existing untrained initialization; no models."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from molgap.constants import REPO_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if hashlib.sha256(args.initial_state.read_bytes()).hexdigest() != "9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c":
        raise RuntimeError("Frozen untrained-state bytes changed")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments/pcqm_gptrans_relation_flow"], cwd=REPO_ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("Commit source and protocol before packaging")
    args.output.mkdir(parents=True)
    files = subprocess.check_output(["git", "ls-files", "-z", "src"], cwd=REPO_ROOT).decode().split("\0")
    inventory = []
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for relative in sorted(filter(None, files)):
            source = REPO_ROOT / relative
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            archive.add(source, arcname=relative)
            inventory.append({"path": relative, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    shutil.copyfile(args.output / "source.tar.gz", args.output / "source_payload.bin")
    shutil.copyfile(args.initial_state, args.output / "initial_state.pt")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    sha = hashlib.sha256((args.output / "source_payload.bin").read_bytes()).hexdigest()
    (args.output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (args.output / "SOURCE_ARCHIVE_SHA256.txt").write_text(sha + "\n", encoding="utf-8")
    (args.output / "SOURCE_FILES.json").write_text(json.dumps({"files": inventory}, indent=2), encoding="utf-8")
    (args.output / "dataset-metadata.json").write_text(json.dumps({"title": "MolGap GPTrans Relation Flow Source", "id": "kaseichou/molgap-gptrans-relation-flow-source", "licenses": [{"name": "other"}], "isPrivate": True}, indent=2), encoding="utf-8")
    print(json.dumps({"source_commit": commit, "archive_sha256": sha, "files": len(inventory)}))


if __name__ == "__main__":
    main()
