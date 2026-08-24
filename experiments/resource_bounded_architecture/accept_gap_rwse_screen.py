"""Accept the Gap-only and normalized-RWSE Kaggle screens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.architecture_screen_acceptance import accept_gap_rwse_screen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--gap-only-dir", type=Path, required=True)
    parser.add_argument("--normalized-dir", type=Path, required=True)
    parser.add_argument("--base-graph", type=Path, required=True)
    parser.add_argument("--rwse-graph", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = accept_gap_rwse_screen(
        baseline_dir=args.baseline_dir,
        gap_only_dir=args.gap_only_dir,
        normalized_dir=args.normalized_dir,
        base_graph=args.base_graph,
        rwse_graph=args.rwse_graph,
        split_csv=args.split_csv,
        output_path=args.output,
    )
    print(json.dumps(report["normalized_rwse_gate"], sort_keys=True))


if __name__ == "__main__":
    main()
