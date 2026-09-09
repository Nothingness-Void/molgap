"""Thin CLI for the QM9 charge-adapter architecture screen."""
from __future__ import annotations

import argparse
from pathlib import Path

from molgap.qm9_charge_adapter import (
    accept_cache,
    build_cache,
    run_preflight,
    run_screen,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-cache")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--shard-size", type=int, default=2_000)

    accept = commands.add_parser("accept-cache")
    accept.add_argument("--cache-root", type=Path, required=True)
    accept.add_argument("--source-commit")

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--cache-root", type=Path, required=True)
    preflight.add_argument("--output-root", type=Path, required=True)
    preflight.add_argument("--source-commit", required=True)
    preflight.add_argument("--cache-sha256", required=True)

    train = commands.add_parser("train")
    train.add_argument("--cache-root", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--source-commit", required=True)
    train.add_argument("--cache-sha256", required=True)
    train.add_argument("--preflight", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "build-cache":
        build_cache(
            args.output_root,
            source_root=args.source_root,
            source_commit=args.source_commit,
            shard_size=args.shard_size,
        )
    elif args.command == "accept-cache":
        accept_cache(
            args.cache_root, expected_source_commit=args.source_commit
        )
    elif args.command == "preflight":
        run_preflight(
            args.cache_root,
            args.output_root,
            source_commit=args.source_commit,
            cache_sha256=args.cache_sha256,
        )
    else:
        run_screen(
            args.cache_root,
            args.output_root,
            source_commit=args.source_commit,
            cache_sha256=args.cache_sha256,
            preflight_path=args.preflight,
        )


if __name__ == "__main__":
    main()
