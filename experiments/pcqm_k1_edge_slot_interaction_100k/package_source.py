"""Package committed source only; no model import or local execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from molgap.constants import REPO_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    experiment = "experiments/pcqm_k1_edge_slot_interaction_100k"
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", experiment],
        cwd=REPO_ROOT, text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Commit protocol/source before packaging")
    args.output.mkdir(parents=True)
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "src"], cwd=REPO_ROOT
    ).decode().split("\0")
    inventory = []
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for name in sorted(filter(None, files)):
            source = REPO_ROOT / name
            archive.add(source, arcname=name)
            inventory.append({"path": name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    shutil.copyfile(args.output / "source.tar.gz", args.output / "source_payload.bin")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    digest = hashlib.sha256((args.output / "source_payload.bin").read_bytes()).hexdigest()
    (args.output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (args.output / "SOURCE_ARCHIVE_SHA256.txt").write_text(digest + "\n", encoding="utf-8")
    (args.output / "SOURCE_FILES.json").write_text(json.dumps({"files": inventory}, indent=2), encoding="utf-8")
    (args.output / "dataset-metadata.json").write_text(json.dumps({
        "id": "kaseichou/molgap-k1-edge-slot-interaction-source",
        "title": "MolGap K1 Edge Slot Interaction Source",
        "licenses": [{"name": "other"}], "isPrivate": True,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"source_commit": commit, "source_archive_sha256": digest, "files": len(inventory)}))


if __name__ == "__main__":
    main()
