"""Thin CLI for direct pure-2D Pair-GPS B3LYP training."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pubchemqc_pair_gps_2d import train_pair_gps_2d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-every-shards", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = train_pair_gps_2d(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        checkpoint_every_shards=args.checkpoint_every_shards,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        split_seed=args.split_seed,
        resume=args.resume,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
