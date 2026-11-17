"""Thin CLI for the one authorized PubChemQC 100K pure-2D architecture test."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pubchemqc_direct_training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()

    model_config = {
        "in_channels": 9,
        "edge_dim": 4,
        "hidden_channels": 160,
        "num_layers": 11,
        "num_heads": 4,
        "dropout": 0.05,
        "n_targets": 3,
        "max_degree": 8,
    }
    train(
        graphs_path=args.graphs,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        model_config=model_config,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        split_seed=args.split_seed,
        resume_from=args.resume_from,
        write_embeddings=not args.no_embeddings,
    )


if __name__ == "__main__":
    main()

