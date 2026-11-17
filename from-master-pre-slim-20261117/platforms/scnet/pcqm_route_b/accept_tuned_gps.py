"""Thin SCNet adapter for one completed tuned Route B GPS encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_route_b_acceptance import accept_pcqm_route_b_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--encoder", choices=("gps9", "gps11_160"), required=True)
    args = parser.parse_args()
    output_dir = args.root / "outputs_tuned" / args.encoder
    report = accept_pcqm_route_b_encoder(
        args.encoder,
        output_dir,
        output_dir / "single_encoder_acceptance.json",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
