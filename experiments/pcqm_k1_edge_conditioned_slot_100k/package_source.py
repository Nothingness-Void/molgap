"""Package committed source plus expanded files for a private Kaggle job."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from molgap.constants import REPO_ROOT


def tracked_source_files() -> list[Path]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "experiments/pcqm_k1_edge_conditioned_slot_100k"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status:
        raise RuntimeError("Commit source and experiment protocol before packaging")
    raw = subprocess.check_output(["git", "ls-files", "-z", "--", "src"], cwd=REPO_ROOT)
    files = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        path = REPO_ROOT / item.decode("utf-8")
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
            and not any(part.endswith(".egg-info") for part in path.parts)
        ):
            files.append(path)
    files.sort(key=lambda path: path.relative_to(REPO_ROOT).as_posix())
    if not files:
        raise RuntimeError("Tracked source inventory is empty")
    return files


def build_payload(output: Path, files: list[Path]) -> str:
    inventory = []
    archive_path = output / "source.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for source in files:
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
    (output / "SOURCE_FILES.json").write_text(
        json.dumps({"files": inventory}, indent=2) + "\n", encoding="utf-8"
    )
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", default="kaseichou/molgap-k1-edge-conditioned-slot-source")
    parser.add_argument("--title", default="MolGap K1 Edge Conditioned Slot Source")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    files = tracked_source_files()
    output.mkdir(parents=True)
    digest = build_payload(output, files)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    (output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "SOURCE_ARCHIVE_SHA256.txt").write_text(digest + "\n", encoding="utf-8")
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {"id": args.dataset_id, "title": args.title, "licenses": [{"name": "other"}], "isPrivate": True},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"source_commit": commit, "source_archive_sha256": digest, "files": len(files)}))


if __name__ == "__main__":
    main()
