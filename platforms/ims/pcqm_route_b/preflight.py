"""Thin IMS adapter for the four Route B CUDA preflights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_route_b_training import CONFIGS, preflight_encoder


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
    reports = {}
    for name, warm_name in WARM_STARTS.items():
        reports[name] = preflight_encoder(
            config=CONFIGS[name],
            root=args.root / "inputs" / "pcqm_route_b_1m",
            warm_start=args.root / "warmstarts" / warm_name,
            output_path=args.root / "outputs" / "preflight" / name,
        )
        print(json.dumps(reports[name], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

