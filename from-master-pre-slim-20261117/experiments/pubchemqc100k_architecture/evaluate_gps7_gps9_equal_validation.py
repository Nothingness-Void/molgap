"""Evaluate one fixed GPS7/GPS9 equal validation configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pubchemqc_pair_gps_2d_screen import (
    evaluate_gps7_gps9_equal_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gps7-dir", type=Path, required=True)
    parser.add_argument("--gps9-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_gps7_gps9_equal_validation(
        split_csv=args.split_csv.resolve(),
        cache_dir=args.cache_dir.resolve(),
        gps7_dir=args.gps7_dir.resolve(),
        gps9_dir=args.gps9_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
