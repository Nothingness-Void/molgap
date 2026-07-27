"""Accept one Route B SchNet branch or combine two branch reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.artifact_acceptance import (
    align_gps_embeddings_to_reference,
    accept_schnet_output,
    accept_schnet_pair,
    extract_schnet_view_for_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    branch = subparsers.add_parser("branch")
    branch.add_argument("--output-dir", type=Path, required=True)
    branch.add_argument("--split-csv", type=Path, required=True)
    branch.add_argument("--split-sha256", required=True)
    branch.add_argument("--accepted-payload", type=Path, required=True)
    branch.add_argument("--report", type=Path, required=True)
    pair = subparsers.add_parser("pair")
    pair.add_argument("--primary-report", type=Path, required=True)
    pair.add_argument("--augmented-report", type=Path, required=True)
    gps = subparsers.add_parser("gps")
    gps.add_argument("--raw-embeddings", type=Path, required=True)
    gps.add_argument("--reference-payload", type=Path, required=True)
    gps.add_argument("--accepted-payload", type=Path, required=True)
    gps.add_argument("--report", type=Path, required=True)
    gps.add_argument("--expected-dim", type=int, required=True)
    gps.add_argument("--expected-sha256", required=True)
    gps.add_argument("--name", required=True)
    view = subparsers.add_parser("schnet-view")
    view.add_argument("--checkpoint", type=Path, required=True)
    view.add_argument("--graph-cache", type=Path, required=True)
    view.add_argument("--reference-payload", type=Path, required=True)
    view.add_argument("--accepted-payload", type=Path, required=True)
    view.add_argument("--report", type=Path, required=True)
    view.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.command == "pair":
        result = accept_schnet_pair(args.primary_report, args.augmented_report)
    elif args.command == "gps":
        result = align_gps_embeddings_to_reference(
            args.raw_embeddings,
            args.reference_payload,
            args.accepted_payload,
            args.report,
            expected_dim=args.expected_dim,
            expected_sha256=args.expected_sha256,
            name=args.name,
        )
    elif args.command == "schnet-view":
        result = extract_schnet_view_for_reference(
            args.checkpoint,
            args.graph_cache,
            args.reference_payload,
            args.accepted_payload,
            args.report,
            batch_size=args.batch_size,
        )
    else:
        result = accept_schnet_output(
            args.output_dir,
            args.split_csv,
            args.accepted_payload,
            args.report,
            expected_split_sha256=args.split_sha256,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
