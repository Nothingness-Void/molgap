"""Stage the committed, explicit source allowlist for Kaggle2 P0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from molgap.constants import REPO_ROOT
from molgap.v4_bundle import build_v4_source_bundle


SOURCE = "src/molgap/gptrans_path_real_preflight.py"
DATASET = "kaseichou/molgap-gptrans-real-path-source-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()
    package = build_v4_source_bundle(
        repo_root=REPO_ROOT, relative_paths=[SOURCE], output_dir=args.output,
        source_commit=commit,
    )
    shutil.copyfile(args.output / "source.tar.gz", args.output / "source_payload.bin")
    (args.output / "dataset-metadata.json").write_text(json.dumps({
        "title": "MolGap GPTrans Real Path Preflight Source V1",
        "id": DATASET,
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }, indent=2), encoding="utf-8")
    print(json.dumps(package, sort_keys=True))


if __name__ == "__main__":
    main()
