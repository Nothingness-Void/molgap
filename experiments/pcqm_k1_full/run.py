"""Thin CLI for the frozen K1 full-role preflight and resumable training."""
from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pcqm_k1_full_runner import run_preflight, train_full


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--dataset-root", type=Path, required=True)
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--platform-id", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--dataset-root", type=Path, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--preflight", type=Path, required=True)
    train.add_argument("--source-commit", required=True)
    train.add_argument("--source-archive", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--max-wall-seconds", type=int)

    args = parser.parse_args()
    if args.command == "preflight":
        run_preflight(
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            output=args.output,
            platform_id=args.platform_id,
        )
        return
    train_full(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        output=args.output,
        preflight_path=args.preflight,
        source_commit=args.source_commit,
        source_archive=args.source_archive,
        resume=args.resume,
        max_wall_seconds=args.max_wall_seconds,
    )


if __name__ == "__main__":
    main()
