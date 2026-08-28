#!/usr/bin/env python3
"""Thin CLI for the official-only PCQM4Mv2 EdgeState experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_official_edge_state import (
    OfficialEdgeStateConfig,
    accept_training_graphs,
    build_training_graph_shard,
    predict_official_tests_from_raw,
    prepare_training_rows,
    train_official_edge_state,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    subparsers = command.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--rows-dir", type=Path, required=True)
    prepare.add_argument("--source-shard-rows", type=int, default=50_000)

    build = subparsers.add_parser("build")
    build.add_argument("--rows-dir", type=Path, required=True)
    build.add_argument("--graph-dir", type=Path, required=True)
    build.add_argument("--shard-index", type=int, required=True)
    build.add_argument("--workers", type=int, default=1)
    build.add_argument("--feature-schema", choices=("legacy", "ogb"), default="legacy")

    accept = subparsers.add_parser("accept")
    accept.add_argument("--archive", type=Path, required=True)
    accept.add_argument("--rows-dir", type=Path, required=True)
    accept.add_argument("--graph-dir", type=Path, required=True)
    accept.add_argument("--output", type=Path, required=True)
    accept.add_argument("--feature-schema", choices=("legacy", "ogb"), default="legacy")

    train = subparsers.add_parser("train")
    train.add_argument("--graph-dir", type=Path, required=True)
    train.add_argument("--acceptance", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--max-epochs", type=int, default=10)
    train.add_argument("--projection-epochs", type=int, default=0)
    train.add_argument(
        "--scheduler", choices=("cosine", "warmup_cosine"), default="cosine"
    )
    train.add_argument("--warmup-epochs", type=int, default=0)
    train.add_argument("--feature-schema", choices=("legacy", "ogb"), default="legacy")
    train.add_argument("--max-projected-hours", type=float, default=12.0)
    train.add_argument("--hard-job-budget-hours", type=float, default=11.5)
    train.add_argument("--patience", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--eval-batch-size", type=int, default=512)
    train.add_argument("--seed", type=int, default=42)

    infer = subparsers.add_parser("infer-test")
    infer.add_argument("--archive", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--workers", type=int, default=6)
    infer.add_argument("--part-rows", type=int, default=20_000)
    return command


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        result = prepare_training_rows(
            args.archive,
            args.rows_dir,
            source_shard_rows=args.source_shard_rows,
        )
    elif args.command == "build":
        result = build_training_graph_shard(
            args.rows_dir,
            args.graph_dir,
            shard_index=args.shard_index,
            workers=args.workers,
            feature_schema=args.feature_schema,
        )
    elif args.command == "accept":
        result = accept_training_graphs(
            args.archive,
            args.rows_dir,
            args.graph_dir,
            args.output,
            feature_schema=args.feature_schema,
        )
    elif args.command == "train":
        result = train_official_edge_state(
            args.graph_dir,
            args.acceptance,
            args.output_dir,
            config=OfficialEdgeStateConfig(
                max_epochs=args.max_epochs,
                projection_epochs=args.projection_epochs,
                scheduler=args.scheduler,
                warmup_epochs=args.warmup_epochs,
                feature_schema=args.feature_schema,
                max_projected_training_s=args.max_projected_hours * 3600.0,
                hard_job_budget_s=args.hard_job_budget_hours * 3600.0,
                patience=args.patience,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                seed=args.seed,
            ),
        )
    else:
        result = predict_official_tests_from_raw(
            args.archive,
            args.checkpoint,
            args.output_dir,
            workers=args.workers,
            part_rows=args.part_rows,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
