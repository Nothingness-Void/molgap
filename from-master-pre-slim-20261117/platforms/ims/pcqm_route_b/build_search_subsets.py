"""Thin IMS adapter for Route B nested search subset construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_route_b_search import build_nested_subsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = build_nested_subsets(
        source_root=args.root / "inputs" / "pcqm_route_b_1m",
        output_root=args.root / "search" / "subsets",
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
