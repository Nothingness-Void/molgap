"""Thin CLI for a frozen PairToken assignment audit."""
from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pcqm_k1_pair_selection_diagnostic import run_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pair-payload", type=Path, required=True)
    parser.add_argument("--k1-payload", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_diagnostic(**vars(args))
    print(result["format"], result["complete"], result["development_rows"])


if __name__ == "__main__":
    main()
