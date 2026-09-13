from __future__ import annotations

import hashlib
import json
import argparse
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiments/pcqm_ogb_rich_edgegps9_100k_v4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".codex_tmp/ogb-rich-edgegps9-v4-source",
    )
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    ).strip()
    if changed:
        raise RuntimeError("Commit all tracked source changes before packaging")
    listed = subprocess.check_output(
        ["git", "ls-files", "-z", "src/molgap", EXPERIMENT], cwd=ROOT
    ).split(b"\0")
    files = sorted(
        value.decode("utf-8")
        for value in listed
        if value and (ROOT / value.decode("utf-8")).is_file()
    )
    if not files:
        raise RuntimeError("No tracked source files were found")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / "source.tar.gz"
    inventory = []
    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        for relative in files:
            path = ROOT / relative
            archive.add(path, arcname=relative, recursive=False)
            inventory.append({"path": relative, "sha256": sha256(path)})
    archive_sha = sha256(archive_path)
    (out / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (out / "SOURCE_ARCHIVE_SHA256.txt").write_text(
        archive_sha + "\n", encoding="utf-8"
    )
    (out / "SOURCE_FILES.json").write_text(
        json.dumps({"files": inventory}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"source_commit": commit, "archive": str(archive_path), "archive_sha256": archive_sha, "file_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
