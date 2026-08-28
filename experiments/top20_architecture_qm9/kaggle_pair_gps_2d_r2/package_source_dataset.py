"""Package the pinned MolGap source used by the two-stage Kaggle2 run."""
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
    args = parser.parse_args()
    root = REPO_ROOT
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing package: {output}")
    (output / "src").mkdir(parents=True)
    shutil.copytree(root / "src" / "molgap", output / "src" / "molgap")
    (output / "experiments" / "qm9_architecture").mkdir(parents=True)
    shutil.copy2(
        root / "experiments" / "qm9_architecture" / "qm9_screen.py",
        output / "experiments" / "qm9_architecture" / "qm9_screen.py",
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    (output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    metadata = {
        "title": "MolGap PairGPS R2 Source",
        "id": "kaseichou/molgap-pairgps-r2-source",
        "licenses": [{"name": "other"}],
    }
    (output / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
