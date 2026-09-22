"""Thin no-inference CLI for frozen PairToken diagnostic acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_k1_pair_selection_acceptance import accept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = accept(args.output_root)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
