"""Package committed MolGap sources for the private K1-variant jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from molgap.constants import REPO_ROOT


def tracked_source_files() -> list[Path]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status:
        raise RuntimeError("Source packaging requires a clean committed src tree")
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "src"], cwd=REPO_ROOT
    )
    paths = [REPO_ROOT / item.decode("utf-8") for item in raw.split(b"\0") if item]
    selected = [
        path
        for path in paths
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
        and not any(part.endswith(".egg-info") for part in path.parts)
    ]
    if not selected:
        raise RuntimeError("Tracked source inventory is empty")
    return sorted(selected, key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def write_archive(path: Path, files: list[Path]) -> str:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            relative = source.relative_to(REPO_ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-id",
        default="kaseichou/molgap-pcqm-k1-v4-variants-source",
    )
    parser.add_argument(
        "--title",
        default="MolGap PCQM K1 V4 Variants Source",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    files = tracked_source_files()
    archive_sha = write_archive(output / "src.zip", files)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    (output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "SOURCE_ARCHIVE_SHA256.txt").write_text(
        archive_sha + "\n", encoding="utf-8"
    )
    (output / "SOURCE_FILES.json").write_text(
        json.dumps(
            [path.relative_to(REPO_ROOT).as_posix() for path in files], indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": args.title,
                "id": args.dataset_id,
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
