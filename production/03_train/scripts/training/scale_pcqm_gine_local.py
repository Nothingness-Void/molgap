"""Scale the accepted PCQM GINE expert to a nested 1M local sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_expert import run_local_scaleup

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=1_000_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--output-name",
        default="local_scaleup_1m_v7_frozen_bn",
    )
    parser.add_argument(
        "--update-batch-norm-stats",
        action="store_true",
        help="Allow source-ordered shards to update BatchNorm running statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accepted = (
        ROOT
        / "results"
        / "kaggle"
        / "staging"
        / "molgap_pcqm_gin_v5_accepted_20260726"
    )
    output = (
        ROOT
        / "results"
        / "phase8"
        / "pcqm_gine_expert_pilot"
        / args.output_name
    )
    metrics = run_local_scaleup(
        raw_csv=ROOT
        / "data"
        / "raw"
        / "pcqm4m-v2"
        / "raw"
        / "data.csv.gz",
        accepted_valid_predictions=accepted
        / "pcqm_official_valid_5k_predictions.csv",
        initial_best=ROOT
        / "results"
        / "phase8"
        / "pcqm_gine_expert_pilot"
        / "local_continuation_v6"
        / "pcqm_gine_best.pt",
        cache_dir=ROOT
        / "data"
        / "cache"
        / "phase8"
        / f"pcqm_gine_{args.train_rows}_nested_seed42_43",
        output_dir=output,
        total_train_rows=args.train_rows,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        freeze_batch_norm_stats=not args.update_batch_norm_stats,
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
