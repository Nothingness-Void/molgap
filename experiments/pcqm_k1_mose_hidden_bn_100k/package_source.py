"""Build the private Kaggle1 source dataset for the hidden-BN screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-id",
        default="nothingnessvoid/molgap-pcqm-k1-mose-hidden-bn-source",
    )
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Commit source before packaging")
    args.output.mkdir(parents=True, exist_ok=True)
    included = [
        "src/molgap",
        "pyproject.toml",
        "experiments/pcqm_k1_mose_hidden_bn_100k/training_contract.json",
    ]
    with tempfile.TemporaryDirectory(prefix="molgap-mose-hidden-bn-") as temporary:
        stage = Path(temporary) / "payload"
        stage.mkdir()
        for relative in included:
            source = ROOT / relative
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(
                    source,
                    target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            else:
                shutil.copy2(source, target)
        archive = args.output / "source_payload.bin"
        with tarfile.open(archive, "w:gz") as bundle:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    bundle.add(path, arcname=path.relative_to(stage).as_posix())
    files = []
    with tarfile.open(args.output / "source_payload.bin", "r:gz") as bundle:
        for member in sorted(bundle.getmembers(), key=lambda item: item.name):
            if member.isfile():
                handle = bundle.extractfile(member)
                files.append(
                    {"path": member.name, "sha256": hashlib.sha256(handle.read()).hexdigest()}
                )
    digest = sha256(args.output / "source_payload.bin")
    (args.output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (args.output / "SOURCE_ARCHIVE_SHA256.txt").write_text(digest + "\n", encoding="utf-8")
    (args.output / "SOURCE_FILES.json").write_text(
        json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "MolGap PCQM K1 MoSE Hidden BN Source",
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
