"""Thin CLI for equal-exposure masked charge pretraining."""
from __future__ import annotations

import argparse
from pathlib import Path

from molgap.qm9_masked_charge_pretraining import run_preflight, run_screen


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train"):
        command = commands.add_parser(name)
        command.add_argument("--cache-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--parent-metrics", type=Path, required=True)
        command.add_argument("--source-commit", required=True)
        command.add_argument("--cache-sha256", required=True)
        if name == "train":
            command.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args(argv)
    common = {
        "cache_root": args.cache_root,
        "output_root": args.output_root,
        "parent_metrics": args.parent_metrics,
        "source_commit": args.source_commit,
        "cache_sha256": args.cache_sha256,
    }
    if args.command == "preflight":
        run_preflight(**common)
    else:
        run_screen(preflight_path=args.preflight, **common)


if __name__ == "__main__":
    main()
