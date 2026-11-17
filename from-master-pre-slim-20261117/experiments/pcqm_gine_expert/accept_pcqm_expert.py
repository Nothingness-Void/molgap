"""Validate a completed local PCQM GINE expert run."""

from __future__ import annotations

import argparse
import json

from molgap.constants import EXPERIMENTS_DIR, REPO_ROOT
from molgap.pcqm_expert import accept_pcqm_expert_artifacts

CACHE_ROOT = REPO_ROOT / "data" / "cache" / "phase8"
EXPERIMENT_RESULTS = EXPERIMENTS_DIR / "pcqm_gine_expert" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default="local_scaleup_1m_v7_frozen_bn",
    )
    parser.add_argument("--train-rows", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = accept_pcqm_expert_artifacts(
        EXPERIMENT_RESULTS / args.run_name,
        CACHE_ROOT / f"pcqm_gine_{args.train_rows}_nested_seed42_43",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
