"""Build aligned PCQM 1M graphs for the Route B precision architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_route_b import build_route_b_cache

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shard-rows", type=int, default=5_000)
    args = parser.parse_args()
    accepted = (
        ROOT
        / "results/kaggle/staging/molgap_pcqm_gin_v5_accepted_20260726"
    )
    result = build_route_b_cache(
        raw_csv=ROOT / "data/raw/pcqm4m-v2/raw/data.csv.gz",
        accepted_valid_predictions=accepted
        / "pcqm_official_valid_5k_predictions.csv",
        gine_cache=ROOT
        / "data/cache/phase8/pcqm_gine_1000000_nested_seed42_43",
        cache_dir=ROOT / "data/cache/phase8/pcqm_route_b_1m",
        total_train_rows=args.train_rows,
        workers=args.workers,
        shard_rows=args.shard_rows,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
