from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_ogb_rich_gps_v4 import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform-id", default="scnet-kunshan")
    args = parser.parse_args()
    result = run_training(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        source_archive=args.source_archive,
        source_archive_sha256=args.source_archive_sha256,
        source_commit=args.source_commit,
        output=args.output,
        platform_id=args.platform_id,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
