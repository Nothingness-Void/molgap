"""Package the label-sealed K1 shadow source for private Kaggle use."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from molgap.constants import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-id", default="kaseichou/molgap-pcqm-k1-shadow-source-v2"
    )
    args = parser.parse_args()
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
    (output / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "MolGap PCQM K1 Shadow Source",
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
