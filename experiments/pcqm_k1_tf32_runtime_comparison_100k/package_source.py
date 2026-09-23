"""Package one committed source tree without copying unrelated worktree edits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from molgap.constants import REPO_ROOT


EXPERIMENT = "experiments/pcqm_k1_tf32_runtime_comparison_100k"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", EXPERIMENT],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status:
        raise RuntimeError("Commit source and experiment before packaging")
    output.mkdir(parents=True)
    paths = ["src", f"{EXPERIMENT}/run.py"]
    archive_path = output / "source.tar.gz"
    with archive_path.open("wb") as destination:
        subprocess.run(
            ["git", "archive", "--format=tar.gz", commit, *paths],
            cwd=REPO_ROOT,
            stdout=destination,
            check=True,
        )
    inventory = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Missing archive payload: {member.name}")
            inventory.append(
                {"path": member.name, "sha256": hashlib.sha256(stream.read()).hexdigest()}
            )
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "SOURCE_ARCHIVE_SHA256.txt").write_text(digest + "\n", encoding="utf-8")
    (output / "SOURCE_FILES.json").write_text(
        json.dumps({"files": inventory}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"source_commit": commit, "source_archive_sha256": digest, "files": len(inventory)}))


if __name__ == "__main__":
    main()
