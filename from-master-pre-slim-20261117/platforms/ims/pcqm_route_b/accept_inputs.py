"""Thin IMS adapter for transferred Route B graph acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_route_b_training import atomic_json, verify_graph_view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    graph_root = args.root / "inputs" / "pcqm_route_b_1m"
    report = {
        "format": "molgap-pcqm-route-b-ims-input-acceptance-v1",
        "status": "complete",
        "views": {
            modality: verify_graph_view(graph_root, modality)
            for modality in ("gps", "primary", "secondary")
        },
    }
    atomic_json(args.root / "logs" / "input_acceptance.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

