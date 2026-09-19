"""Build a private, immutable source dataset for the Kaggle screen."""
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
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "src"], cwd=REPO_ROOT
    ).decode().split("\0")
    inventory = []
    archive_path = args.output / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for relative in sorted(filter(None, files)):
            source = REPO_ROOT / relative
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            archive.add(source, arcname=relative)
            inventory.append(
                {"path": relative, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            )
    shutil.copyfile(archive_path, args.output / "source_payload.bin")
    shutil.copyfile(args.initial_state, args.output / "initial_state.pt")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    sha = hashlib.sha256((args.output / "source_payload.bin").read_bytes()).hexdigest()
    (args.output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (args.output / "SOURCE_ARCHIVE_SHA256.txt").write_text(sha + "\n", encoding="utf-8")
    (args.output / "SOURCE_FILES.json").write_text(
        json.dumps({"files": inventory}, indent=2), encoding="utf-8"
    )
    metadata = {
        "title": "MolGap GPTrans Flow Ablation Source",
        "id": args.dataset_id,
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    (args.output / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"source_commit": commit, "archive_sha256": sha, "files": len(inventory)}))


if __name__ == "__main__":
    main()
