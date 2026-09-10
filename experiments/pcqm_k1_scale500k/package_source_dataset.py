"""Package the frozen K1 scale source for a private Kaggle dataset."""
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
    parser.add_argument("--cache-sha256")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.make_archive(str(output / "src"), "zip", REPO_ROOT, "src")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    (output / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
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
