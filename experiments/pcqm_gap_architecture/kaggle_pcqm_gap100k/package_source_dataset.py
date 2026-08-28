"""Package pinned MolGap source for the official PCQM Gap 100K screen."""
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
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    (output / "PCQM_GAP100K_SOURCE_COMMIT.txt").write_text(
        commit + "\n", encoding="utf-8"
    )
    metadata = {
        "title": "MolGap Official PCQM Gap100K Source",
        "id": "kaseichou/molgap-pcqm-gap100k-r1-source",
        "licenses": [{"name": "other"}],
    }
    (output / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()

