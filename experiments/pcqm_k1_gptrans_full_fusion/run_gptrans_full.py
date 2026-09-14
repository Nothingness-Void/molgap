"""CLI for full-train GPTrans-T runtime calibration and resumable training."""
from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pcqm_gptrans_full_runner import run_preflight, train_full


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--dataset-root", type=Path, required=True)
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train")
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
        result = run_preflight(
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            output=args.output,
        )
    else:
        result = train_full(
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            output=args.output,
            preflight_path=args.preflight,
            source_commit=args.source_commit,
            source_archive=args.source_archive,
            resume=args.resume,
            max_wall_seconds=args.max_wall_seconds,
        )
    print(result, flush=True)


if __name__ == "__main__":
    main()
