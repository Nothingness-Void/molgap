"""Package pinned MolGap source for the conditional EdgeState R4 screen."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
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
        "title": "MolGap EdgeState Readout R4 Source",
        "id": "kaseichou/molgap-edge-state-readout-r4-source",
        "licenses": [{"name": "other"}],
    }
    (output / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
