"""Accept the one-seed gated Structural GPS feasibility screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.architecture_screen_acceptance import (
    accept_gated_structural_feasibility,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--base-graph", type=Path, required=True)
    parser.add_argument("--rwse-graph", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-training-time-s", type=float, default=4_500.0)
    args = parser.parse_args()
    report = accept_gated_structural_feasibility(
        baseline_dir=args.baseline_dir,
        candidate_dir=args.candidate_dir,
        base_graph=args.base_graph,
        rwse_graph=args.rwse_graph,
        split_csv=args.split_csv,
        output_path=args.output,
        max_training_time_s=args.max_training_time_s,
    )
    print(json.dumps(report["feasibility_gate"], sort_keys=True))


if __name__ == "__main__":
    main()
