"""CLI for accepted full GPTrans-T convergence continuation."""
from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pcqm_gptrans_convergence import (
    accept_convergence,
    run_preflight,
    train_convergence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    train = sub.add_parser("train")
    accept = sub.add_parser("accept")
    for command in (preflight, train):
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--valid-graph-root", type=Path, required=True)
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--source-evaluation", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    train.add_argument("--preflight", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--max-wall-seconds", type=int)
    accept.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = run_preflight(
            dataset_root=args.dataset_root, manifest_path=args.manifest,
            valid_graph_root=args.valid_graph_root, source_root=args.source_root,
            source_evaluation_path=args.source_evaluation, output=args.output,
        )
    elif args.command == "train":
        result = train_convergence(
            dataset_root=args.dataset_root, manifest_path=args.manifest,
            valid_graph_root=args.valid_graph_root, source_root=args.source_root,
            source_evaluation_path=args.source_evaluation, output=args.output,
            preflight_path=args.preflight,
            resume=args.resume, max_wall_seconds=args.max_wall_seconds,
        )
    else:
        result = accept_convergence(output=args.output)
    print(result, flush=True)


if __name__ == "__main__":
    main()
