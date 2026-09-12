"""Thin CLI for the GPTrans-T 100K V4 reference."""
from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pcqm_gptrans_v4 import run_preflight, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset-root", type=Path, required=True)
    common.add_argument("--manifest", type=Path, required=True)
    common.add_argument("--source-archive", type=Path, required=True)
    common.add_argument("--source-archive-sha256", required=True)
    common.add_argument("--source-commit", required=True)
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--platform-id", required=True)
    subparsers.add_parser("preflight", parents=[common])
    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = run_preflight(
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            source_archive=args.source_archive,
            source_archive_sha256=args.source_archive_sha256,
            source_commit=args.source_commit,
            output=args.output,
            platform_id=args.platform_id,
        )
    else:
        result = run_training(
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            preflight_path=args.preflight,
            source_archive=args.source_archive,
            source_archive_sha256=args.source_archive_sha256,
            source_commit=args.source_commit,
            output=args.output,
            platform_id=args.platform_id,
        )
    print(result, flush=True)


if __name__ == "__main__":
    main()
