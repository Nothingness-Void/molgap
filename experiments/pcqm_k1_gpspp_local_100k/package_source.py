"""Package committed source for the private GPSPP-local K1 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from molgap.constants import REPO_ROOT


EXPERIMENT = "experiments/pcqm_k1_gpspp_local_100k"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", EXPERIMENT],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Commit source and protocol before packaging")
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "src"], cwd=REPO_ROOT
    )
    files = sorted(
        (
            REPO_ROOT / item.decode("utf-8")
            for item in raw.split(b"\0")
            if item
        ),
        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),
    )
    files = [
        path
        for path in files
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
        and not any(part.endswith(".egg-info") for part in path.parts)
    ]
    if not files:
        raise RuntimeError("Tracked source inventory is empty")
    args.output.mkdir(parents=True)
    inventory = []
    archive = args.output / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for source in files:
            relative = source.relative_to(REPO_ROOT).as_posix()
            handle.add(source, arcname=relative, recursive=False)
            inventory.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
            expanded = args.output / relative
            expanded.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, expanded)
    payload = args.output / "source_payload.bin"
    shutil.copyfile(archive, payload)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    archive_sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    (args.output / "SOURCE_COMMIT.txt").write_text(commit + "\n")
    (args.output / "SOURCE_ARCHIVE_SHA256.txt").write_text(
        archive_sha + "\n"
    )
    (args.output / "SOURCE_FILES.json").write_text(
        json.dumps({"files": inventory}, indent=2) + "\n"
    )
    (args.output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": "kaseichou/molgap-k1-gpspp-local-source",
                "title": "MolGap K1 GPSPP Local Source",
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "source_commit": commit,
                "source_archive_sha256": archive_sha,
                "files": len(files),
            }
        )
    )


if __name__ == "__main__":
    main()
