"""Continue the accepted PCQM GINE v5 specialist on the local GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.constants import EXPERIMENTS_DIR, PLATFORMS_DIR, RAW_DIR, REPO_ROOT
from molgap.pcqm_expert import run_local_continuation

CACHE_ROOT = REPO_ROOT / "data" / "cache" / "phase8"
EXPERIMENT_RESULTS = EXPERIMENTS_DIR / "pcqm_gine_expert" / "results"
ACCEPTED = (
    PLATFORMS_DIR
    / "_records"
    / "kaggle"
    / "staging"
    / "molgap_pcqm_gin_v5_accepted_20260726"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=RAW_DIR / "pcqm4m-v2" / "raw" / "data.csv.gz",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_ROOT / "pcqm_gine_250k_seed42",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_RESULTS / "local_continuation_v6",
    )
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_local_continuation(
        raw_csv=args.raw_csv,
        accepted_valid_predictions=(
            ACCEPTED / "pcqm_official_valid_5k_predictions.csv"
        ),
        resume_last=ACCEPTED / "pcqm_gine_last.pt",
        resume_best=ACCEPTED / "pcqm_gine_best.pt",
        resume_log=ACCEPTED / "pcqm_gine_train_log.csv",
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_epoch=args.max_epoch,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
