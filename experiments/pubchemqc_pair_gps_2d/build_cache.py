"""Thin CLI for the immutable PubChemQC pure-2D Pair-GPS cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pubchemqc_pair_gps_2d import build_pair_gps_2d_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-graph-dir", type=Path, required=True)
    parser.add_argument("--primary-acceptance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    result = build_pair_gps_2d_cache(
        manifest_path=args.manifest,
        primary_graph_dir=args.primary_graph_dir,
        primary_acceptance=args.primary_acceptance,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
