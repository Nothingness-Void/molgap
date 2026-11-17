"""Run one validation-only trial in the fair PubChemQC-100K screen."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pubchemqc_pair_gps_2d_screen import train_screen_trial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate", choices=("gps7", "gps9", "pair_gps_2d"), required=True
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        train_screen_trial(
            split_csv=args.split_csv,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            candidate=args.candidate,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            resume=args.resume,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
