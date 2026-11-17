"""Thin IMS adapter for completed Route B encoder acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_route_b_acceptance import accept_pcqm_route_b_encoders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    outputs = args.root / "outputs"
    report = accept_pcqm_route_b_encoders(
        {
            name: outputs / name
            for name in (
                "gps9",
                "gps11_160",
                "primary_schnet",
                "augmented_schnet",
            )
        },
        args.root / "logs" / "encoder_acceptance.json",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

