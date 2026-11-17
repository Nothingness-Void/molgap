"""Build the frozen-split 18-wide pure-2D PubChemQC-100K cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pubchemqc_pair_gps_2d_screen import build_screen_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--shard-rows", type=int, default=5_000)
    args = parser.parse_args()
    print(
        build_screen_cache(
            split_csv=args.split_csv,
            output_dir=args.output_dir,
            workers=args.workers,
            shard_rows=args.shard_rows,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
