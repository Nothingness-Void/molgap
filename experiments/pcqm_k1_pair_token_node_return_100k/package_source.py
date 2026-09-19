"""Package committed reusable source for one private Kaggle dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from molgap.constants import REPO_ROOT


EXPERIMENT = "experiments/pcqm_k1_pair_token_node_return_100k"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-id", default="nvoid912/molgap-k1-node-adaptive-pairtoken-source"
    )
    args = parser.parse_args()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", EXPERIMENT],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status:
        raise RuntimeError("Commit source and protocol before packaging")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    raw = subprocess.check_output(["git", "ls-files", "-z", "--", "src"], cwd=REPO_ROOT)
    files = sorted(
        REPO_ROOT / item.decode("utf-8")
        for item in raw.split(b"\0")
        if item
    )
    inventory = []
    archive_path = output / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for source in files:
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            relative = source.relative_to(REPO_ROOT).as_posix()
            archive.add(source, arcname=relative, recursive=False)
            inventory.append(
                {"path": relative, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            )
            expanded = output / relative
            expanded.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, expanded)
    shutil.copyfile(archive_path, output / "source_payload.bin")
    digest = hashlib.sha256((output / "source_payload.bin").read_bytes()).hexdigest()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    (output / "SOURCE_FILES.json").write_text(
        json.dumps({"files": inventory}, indent=2) + "\n", encoding="utf-8"
    )
    (output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "SOURCE_ARCHIVE_SHA256.txt").write_text(digest + "\n", encoding="utf-8")
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": args.dataset_id,
                "title": "MolGap K1 Node Adaptive PairToken Source",
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"source_commit": commit, "source_archive_sha256": digest}))


if __name__ == "__main__":
    main()

