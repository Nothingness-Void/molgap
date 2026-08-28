#!/usr/bin/env python3
"""Thin CLI for the official-train-only PCQM feature-contract screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_feature_screen import (
    FeatureScreenConfig,
    accept_schedule_screen_runs,
    accept_feature_screen_graphs,
    accept_feature_screen_runs,
    build_feature_screen_graph_shard,
    preflight_feature_screen,
    prepare_feature_screen_rows,
    train_feature_screen,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    subparsers = command.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--rows-dir", type=Path, required=True)
    prepare.add_argument("--train-rows", type=int, default=100_000)
    prepare.add_argument("--development-rows", type=int, default=10_000)
    prepare.add_argument("--seed", type=int, default=20260826)

    build = subparsers.add_parser("build")
    build.add_argument("--rows-dir", type=Path, required=True)
    build.add_argument("--graph-dir", type=Path, required=True)
    build.add_argument("--shard-index", type=int, required=True)

    accept = subparsers.add_parser("accept-graphs")
    accept.add_argument("--rows-dir", type=Path, required=True)
    accept.add_argument("--graph-dir", type=Path, required=True)
    accept.add_argument("--output", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--graph-dir", type=Path, required=True)
    preflight.add_argument("--acceptance", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--graph-dir", type=Path, required=True)
    train.add_argument("--acceptance", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--schema", choices=("legacy", "ogb"), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--max-epochs", type=int, default=40)
    train.add_argument(
        "--scheduler", choices=("cosine", "warmup_cosine"), default="cosine"
    )
    train.add_argument("--warmup-epochs", type=int, default=0)
    train.add_argument("--patience", type=int, default=7)

    aggregate = subparsers.add_parser("accept-runs")
    aggregate.add_argument("--runs-dir", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    schedule = subparsers.add_parser("accept-schedule-runs")
    schedule.add_argument("--runs-dir", type=Path, required=True)
    schedule.add_argument("--output", type=Path, required=True)
    return command


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        result = prepare_feature_screen_rows(
            args.archive,
            args.rows_dir,
            train_rows=args.train_rows,
            development_rows=args.development_rows,
            seed=args.seed,
        )
    elif args.command == "build":
        result = build_feature_screen_graph_shard(
            args.rows_dir, args.graph_dir, shard_index=args.shard_index
        )
    elif args.command == "accept-graphs":
        result = accept_feature_screen_graphs(
            args.rows_dir, args.graph_dir, args.output
        )
    elif args.command == "preflight":
        result = preflight_feature_screen(
            args.graph_dir, args.acceptance, args.output
        )
    elif args.command == "train":
        result = train_feature_screen(
            args.graph_dir,
            args.acceptance,
            args.output_dir,
            schema=args.schema,
            seed=args.seed,
            config=FeatureScreenConfig(
                max_epochs=args.max_epochs,
                scheduler=args.scheduler,
                warmup_epochs=args.warmup_epochs,
                patience=args.patience,
            ),
        )
    elif args.command == "accept-runs":
        result = accept_feature_screen_runs(args.runs_dir, args.output)
    else:
        result = accept_schedule_screen_runs(args.runs_dir, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
