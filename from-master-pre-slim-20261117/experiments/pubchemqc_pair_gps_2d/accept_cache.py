"""Thin CLI for independent acceptance of the pure-2D graph cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pubchemqc_pair_gps_2d import accept_pair_gps_2d_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    result = accept_pair_gps_2d_cache(
        cache_dir=args.cache_dir,
        split_seed=args.split_seed,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
