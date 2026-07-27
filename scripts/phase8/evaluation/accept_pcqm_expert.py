"""Validate a completed local PCQM GINE expert run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_expert import accept_pcqm_expert_artifacts

ROOT = Path(__file__).resolve().parents[3]


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
        ROOT
        / "results"
        / "phase8"
        / "pcqm_gine_expert_pilot"
        / args.run_name,
        ROOT
        / "data"
        / "cache"
        / "phase8"
        / f"pcqm_gine_{args.train_rows}_nested_seed42_43",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
