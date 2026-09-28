"""Thin source-publishing CLI; shared bundle code owns the archive format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from molgap.constants import REPO_ROOT
from molgap.v4_bundle import build_v4_source_bundle


DATASET_ID = "kaseichou/molgap-motif-hierarchy-source"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "src/molgap"], cwd=REPO_ROOT,
    ).split(b"\0")
    paths = [item.decode("utf-8") for item in tracked if item.endswith(b".py")]
    bundle = build_v4_source_bundle(
        repo_root=REPO_ROOT, relative_paths=paths,
        output_dir=output, source_commit=commit,
    )
    shutil.copyfile(output / "source.tar.gz", output / "source_payload.bin")
    (output / "dataset-metadata.json").write_text(json.dumps({
        "id": DATASET_ID, "title": "MolGap Motif Hierarchy CPU Source",
        "licenses": [{"name": "other"}], "isPrivate": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "dataset_id": DATASET_ID,
        "source_commit": commit,
        "source_archive_sha256": bundle["archive_sha256"],
        "source_file_count": bundle["file_count"],
    }, indent=2))


if __name__ == "__main__":
    main()

