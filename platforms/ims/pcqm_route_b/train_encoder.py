"""Thin IMS adapter for one resumable Route B encoder continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_route_b_training import CONFIGS, train_encoder


WARM_STARTS = {
    "gps9": "gps9.pt",
    "gps11_160": "gps11_160.pt",
    "primary_schnet": "primary_schnet.pt",
    "augmented_schnet": "augmented_schnet.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--encoder", choices=sorted(CONFIGS), required=True)
    args = parser.parse_args()
    config = CONFIGS[args.encoder]
    graph_root = args.root / "inputs" / "pcqm_route_b_1m"
    roots = {config.modality: graph_root}
    if config.augmented:
        roots["secondary"] = graph_root
    result = train_encoder(
        config=config,
        roots=roots,
        warm_start=args.root / "warmstarts" / WARM_STARTS[args.encoder],
        output_dir=args.root / "outputs" / args.encoder,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

