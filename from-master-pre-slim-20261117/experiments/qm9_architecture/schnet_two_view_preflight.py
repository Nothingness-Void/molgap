"""Thin CLI wrapper for the two-ETKDG-view SchNet preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.qm9_conformer import train_schnet_conformer_augmented


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--validation-size", type=int, default=3000)
    parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--geometry-seeds", type=int, nargs=2, default=[42, 43])
    args = parser.parse_args()
    result = train_schnet_conformer_augmented(
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        epochs=args.epochs,
        seed=args.seed,
        split_seed=args.split_seed,
        geometry_seeds=tuple(args.geometry_seeds),
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
    )
    print(json.dumps(result["metrics"]["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
