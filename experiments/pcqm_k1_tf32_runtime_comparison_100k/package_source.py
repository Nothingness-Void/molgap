"""Package one committed source tree without copying unrelated worktree edits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

from molgap.constants import REPO_ROOT


EXPERIMENT = "experiments/pcqm_k1_tf32_runtime_comparison_100k"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    commit = args.source_commit
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Source commit must be a full Git SHA-1")
    subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=REPO_ROOT, check=True)
    subprocess.run(
        ["git", "diff", "--exit-code", commit, "HEAD", "--", "src", f"{EXPERIMENT}/run.py"],
        cwd=REPO_ROOT,
        check=True,
    )
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
    (output / "SOURCE_COMMIT.txt").write_bytes((commit + "\n").encode("ascii"))
    (output / "SOURCE_ARCHIVE_SHA256.txt").write_bytes((digest + "\n").encode("ascii"))
    (output / "SOURCE_FILES.json").write_bytes(
        (json.dumps({"files": inventory}, indent=2) + "\n").encode("utf-8")
    )
    print(json.dumps({"source_commit": commit, "source_archive_sha256": digest, "files": len(inventory)}))


if __name__ == "__main__":
    main()
