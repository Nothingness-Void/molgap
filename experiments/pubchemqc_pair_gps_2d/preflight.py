"""Thin IMS compute-node preflight for Pair-GPS 2D."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pubchemqc_pair_gps_2d import preflight_pair_gps_2d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    result = preflight_pair_gps_2d(
        cache_dir=args.cache_dir,
        output_path=args.output,
        batch_size=args.batch_size,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
