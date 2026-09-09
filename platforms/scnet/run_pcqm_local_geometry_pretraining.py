"""SCNet entry points for the PCQM local-geometry pretraining screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_local_geometry_pretraining import (
    accept_pair,
    run_pair,
    run_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--graph-root", type=Path, required=True)
    preflight.add_argument("--output-root", type=Path, required=True)
    preflight.add_argument("--source-commit", required=True)

    train = commands.add_parser("train")
    train.add_argument("--graph-root", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--source-commit", required=True)
    train.add_argument("--replicate", type=int, choices=(1, 2), required=True)
    train.add_argument("--preflight", type=Path, required=True)

    accept = commands.add_parser("accept")
    accept.add_argument("--output-root", type=Path, required=True)
    accept.add_argument("--source-commit", required=True)
    accept.add_argument("--replicate", type=int, choices=(1, 2), required=True)

    args = parser.parse_args()
    if args.command == "preflight":
        result = run_preflight(
            args.graph_root, args.output_root, source_commit=args.source_commit
        )
    elif args.command == "train":
        result = run_pair(
            args.graph_root,
            args.output_root,
            source_commit=args.source_commit,
            replicate=args.replicate,
            preflight_path=args.preflight,
        )
    else:
        result = accept_pair(
            args.output_root,
            expected_source_commit=args.source_commit,
            expected_replicate=args.replicate,
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
