"""Benchmark hardware-efficient PairGPS2D training contracts on one A100."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pubchemqc_pair_gps_2d_benchmark import (
    benchmark_pair_gps_2d_a100,
    parse_benchmark_specs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="precision:batch_size:num_workers; precision is fp32, tf32, or bf16",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        benchmark_pair_gps_2d_a100(
            split_csv=args.split_csv,
            cache_dir=args.cache_dir,
            output_path=args.output,
            specs=parse_benchmark_specs(args.config),
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
