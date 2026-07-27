"""Build aligned PCQM 1M graphs for the Route B precision architecture."""

from __future__ import annotations

import argparse
import json

from molgap.constants import PLATFORMS_DIR, RAW_DIR, REPO_ROOT
from molgap.pcqm_route_b import build_route_b_cache

CACHE_ROOT = REPO_ROOT / "data" / "cache" / "phase8"
ACCEPTED = (
    PLATFORMS_DIR
    / "_records"
    / "kaggle"
    / "staging"
    / "molgap_pcqm_gin_v5_accepted_20260726"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shard-rows", type=int, default=5_000)
    args = parser.parse_args()
    result = build_route_b_cache(
        raw_csv=RAW_DIR / "pcqm4m-v2" / "raw" / "data.csv.gz",
        accepted_valid_predictions=ACCEPTED
        / "pcqm_official_valid_5k_predictions.csv",
        gine_cache=CACHE_ROOT / "pcqm_gine_1000000_nested_seed42_43",
        cache_dir=CACHE_ROOT / "pcqm_route_b_1m",
        total_train_rows=args.train_rows,
        workers=args.workers,
        shard_rows=args.shard_rows,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
