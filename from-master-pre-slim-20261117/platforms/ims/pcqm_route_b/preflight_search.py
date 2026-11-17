"""Thin IMS adapter for the Route B hyperparameter-search CUDA preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_route_b_search import preflight_search


WARM_STARTS = {
    "gps9": "gps9.pt",
    "gps11_160": "gps11_160.pt",
    "primary_schnet": "primary_schnet.pt",
    "augmented_schnet": "augmented_schnet.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = preflight_search(
        subset_root=args.root / "search" / "subsets",
        warm_starts={
            name: args.root / "warmstarts" / filename
            for name, filename in WARM_STARTS.items()
        },
        output_path=args.root / "search" / "preflight.json",
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
