"""SCNet entry points for the paired EdgeState local-hierarchy screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_local_hierarchy import (
    accept_local_label_cache,
    accept_paired_screen,
    build_local_label_cache,
    finalize_paired_outputs,
    run_worker,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    build_labels = commands.add_parser("build-labels")
    build_labels.add_argument("--source-csv", type=Path, required=True)
    build_labels.add_argument("--graph-root", type=Path, required=True)
    build_labels.add_argument("--output-root", type=Path, required=True)
    build_labels.add_argument("--source-commit", required=True)

    accept_labels = commands.add_parser("accept-labels")
    accept_labels.add_argument("--output-root", type=Path, required=True)
    accept_labels.add_argument("--source-commit", required=True)

    worker = commands.add_parser("worker")
    worker.add_argument("--role", choices=("scratch", "pretrained"), required=True)
    worker.add_argument("--graph-root", type=Path, required=True)
    worker.add_argument("--label-root", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--label-sha256", required=True)
    worker.add_argument("--source-commit", required=True)

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--output-root", type=Path, required=True)
    finalize.add_argument("--label-sha256", required=True)
    finalize.add_argument("--source-commit", required=True)

    accept = commands.add_parser("accept")
    accept.add_argument("--output-root", type=Path, required=True)
    accept.add_argument("--label-sha256", required=True)
    accept.add_argument("--source-commit", required=True)

    args = parser.parse_args()
    if args.command == "build-labels":
        result = build_local_label_cache(
            args.source_csv,
            args.graph_root,
            args.output_root,
            source_commit=args.source_commit,
        )
    elif args.command == "accept-labels":
        result = accept_local_label_cache(
            args.output_root, expected_source_commit=args.source_commit
        )
    elif args.command == "worker":
        result = run_worker(
            args.role,
            args.output_root,
            label_sha256=args.label_sha256,
            source_commit=args.source_commit,
            graph_root=args.graph_root,
            label_root=args.label_root,
        )
    elif args.command == "finalize":
        result = finalize_paired_outputs(
            args.output_root,
            label_sha256=args.label_sha256,
            source_commit=args.source_commit,
        )
    else:
        result = accept_paired_screen(
            args.output_root,
            expected_source_commit=args.source_commit,
            expected_label_sha256=args.label_sha256,
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
