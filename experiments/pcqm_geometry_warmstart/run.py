"""Thin CLI for the official-PCQM geometry warm-start experiment."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from molgap.pcqm_geometry_warmstart import (
    GeometryWarmstartConfig,
    accept_geometry_cache,
    build_geometry_shard,
    cpu_smoke,
    gpu_preflight,
    train_geometry_warmstart,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--rows-dir", type=Path, required=True)
    smoke.add_argument("--base-graph-dir", type=Path, required=True)
    smoke.add_argument("--base-acceptance", type=Path, required=True)
    smoke.add_argument("--source-checkpoint", type=Path, required=True)
    smoke.add_argument("--source-config-checkpoint", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser("build-shard")
    build.add_argument("--rows-dir", type=Path, required=True)
    build.add_argument("--base-graph-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--shard-index", type=int, required=True)
    build.add_argument("--workers", type=int, default=8)

    accept = subparsers.add_parser("accept")
    accept.add_argument("--rows-dir", type=Path, required=True)
    accept.add_argument("--base-acceptance", type=Path, required=True)
    accept.add_argument("--graph-dir", type=Path, required=True)
    accept.add_argument("--output", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--graph-dir", type=Path, required=True)
    preflight.add_argument("--acceptance", type=Path, required=True)
    preflight.add_argument("--source-checkpoint", type=Path, required=True)
    preflight.add_argument("--source-config-checkpoint", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--batches", type=int, default=64)
    preflight.add_argument("--max-epochs", type=int)

    train = subparsers.add_parser("train")
    train.add_argument("--graph-dir", type=Path, required=True)
    train.add_argument("--acceptance", type=Path, required=True)
    train.add_argument("--source-checkpoint", type=Path, required=True)
    train.add_argument("--source-config-checkpoint", type=Path, required=True)
    train.add_argument("--preflight", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--max-epochs", type=int)
    return result


def main() -> None:
    args = parser().parse_args()
    config = GeometryWarmstartConfig()
    if getattr(args, "max_epochs", None) is not None:
        if args.max_epochs <= 0:
            raise ValueError("--max-epochs must be positive")
        config = replace(config, max_epochs=args.max_epochs)
    if args.command == "smoke":
        payload = cpu_smoke(
            args.rows_dir, args.base_graph_dir, args.base_acceptance,
            args.source_checkpoint, args.source_config_checkpoint, args.output,
        )
    elif args.command == "build-shard":
        payload = build_geometry_shard(
            args.rows_dir, args.base_graph_dir, args.output_dir,
            shard_index=args.shard_index, workers=args.workers,
        )
    elif args.command == "accept":
        payload = accept_geometry_cache(
            args.rows_dir, args.base_acceptance, args.graph_dir, args.output,
        )
    elif args.command == "preflight":
        payload = gpu_preflight(
            args.graph_dir, args.acceptance, args.source_checkpoint,
            args.source_config_checkpoint, args.output,
            config=config, batches=args.batches,
        )
    else:
        payload = train_geometry_warmstart(
            args.graph_dir, args.acceptance, args.source_checkpoint,
            args.source_config_checkpoint, args.preflight, args.output_dir,
            config=config,
        )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
