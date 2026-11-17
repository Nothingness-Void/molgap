"""Accept the frozen-split 18-wide pure-2D PubChemQC-100K cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pubchemqc_pair_gps_2d_screen import accept_screen_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        accept_screen_cache(split_csv=args.split_csv, cache_dir=args.cache_dir),
        flush=True,
    )


if __name__ == "__main__":
    main()
