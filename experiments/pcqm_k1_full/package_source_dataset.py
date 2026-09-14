"""Package the frozen K1 scale source for a private Kaggle dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from molgap.constants import REPO_ROOT


EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def tracked_source_files() -> list[Path]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status:
        raise RuntimeError("Source packaging requires a clean, committed src tree")
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "src"], cwd=REPO_ROOT
    )
    paths = [REPO_ROOT / value.decode("utf-8") for value in raw.split(b"\0") if value]
    selected = [
        path
        for path in paths
        if not EXCLUDED_PARTS.intersection(path.parts)
        and path.suffix not in EXCLUDED_SUFFIXES
        and not any(part.endswith(".egg-info") for part in path.parts)
    ]
    if not selected or any(not path.is_file() for path in selected):
        raise RuntimeError("Tracked source inventory is empty or incomplete")
    return sorted(selected, key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def write_reproducible_archive(path: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(
            files, key=lambda item: item.relative_to(REPO_ROOT).as_posix()
        ):
            relative = source.relative_to(REPO_ROOT).as_posix()
            data = source.read_bytes()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-sha256")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    files = tracked_source_files()
    source_sha256 = write_reproducible_archive(output / "src.zip", files)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    (output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "SOURCE_ARCHIVE_SHA256.txt").write_text(
        source_sha256 + "\n", encoding="utf-8"
    )
    (output / "SOURCE_FILES.json").write_text(
        json.dumps(
            [path.relative_to(REPO_ROOT).as_posix() for path in files], indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    if args.cache_sha256:
        (output / "CACHE_AGGREGATE_SHA256.txt").write_text(
            args.cache_sha256 + "\n", encoding="utf-8"
        )
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "MolGap PCQM K1 Scale500K Source",
                "id": "kaseichou/molgap-pcqm-k1-scale500k-source",
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
