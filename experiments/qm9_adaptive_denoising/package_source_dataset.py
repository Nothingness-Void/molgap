"""Package the frozen adaptive-denoising source as a private Kaggle dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from molgap.constants import REPO_ROOT


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-id", default="kaseichou/molgap-qm9-adaptive-denoising-source"
    )
    args = parser.parse_args()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("Refusing to package a dirty or untracked source tree")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite source package: {output}")
    (output / "src").mkdir(parents=True)
    shutil.copytree(
        REPO_ROOT / "src" / "molgap",
        output / "src" / "molgap",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    (output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "SOURCE_TREE_SHA256.txt").write_text(
        tree_sha256(output / "src" / "molgap") + "\n", encoding="utf-8"
    )
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "MolGap QM9 Adaptive Denoising Source",
                "id": args.dataset_id,
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
