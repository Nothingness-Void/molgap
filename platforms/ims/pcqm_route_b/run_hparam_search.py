"""Thin IMS adapter for one Route B encoder search stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_route_b_search import run_search_stage


WARM_STARTS = {
    "gps9": "gps9.pt",
    "gps11_160": "gps11_160.pt",
    "primary_schnet": "primary_schnet.pt",
    "augmented_schnet": "augmented_schnet.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--encoder", choices=sorted(WARM_STARTS), required=True)
    parser.add_argument(
        "--stage",
        choices=("50k", "100k", "100k_confirm"),
        required=True,
    )
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    result = run_search_stage(
        encoder_name=args.encoder,
        stage=args.stage,
        subset_root=args.root / "search" / "subsets",
        warm_start=args.root / "warmstarts" / WARM_STARTS[args.encoder],
        output_root=args.root / "search" / "outputs" / args.stage,
        top_k=args.top_k,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
