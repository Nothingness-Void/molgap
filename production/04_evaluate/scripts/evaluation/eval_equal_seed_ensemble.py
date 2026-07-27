"""Evaluate an identity-aligned equal-weight seed ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.ensemble_evaluation import atomic_json, evaluate_equal_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path, nargs="+", required=True)
    parser.add_argument("--pcqm", type=Path, nargs="+", required=True)
    parser.add_argument("--model-hint", default="repaired_2m_d_gps7_seed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_equal_ensemble(
        args.common,
        args.pcqm,
        model_hint=args.model_hint,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
