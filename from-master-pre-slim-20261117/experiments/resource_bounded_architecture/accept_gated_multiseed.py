"""Accept independently packaged gated Structural GPS seeds 42/43/44."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.architecture_screen_acceptance import accept_gated_structural_multiseed


def _candidate(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("candidate must use SEED=PATH")
    return int(seed_text), Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=_candidate, action="append", required=True)
    parser.add_argument("--base-graph", type=Path, required=True)
    parser.add_argument("--rwse-graph", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-eV", type=float, default=0.001)
    parser.add_argument("--max-training-time-s", type=float, default=4_500.0)
    args = parser.parse_args()
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        parser.error("candidate seeds must be unique")
    report = accept_gated_structural_multiseed(
        baseline_dir=args.baseline_dir,
        candidate_dirs=candidates,
        base_graph=args.base_graph,
        rwse_graph=args.rwse_graph,
        split_csv=args.split_csv,
        output_path=args.output,
        gate_eV=args.gate_eV,
        max_training_time_s=args.max_training_time_s,
    )
    print(json.dumps(report["selection_gate"], sort_keys=True))


if __name__ == "__main__":
    main()
