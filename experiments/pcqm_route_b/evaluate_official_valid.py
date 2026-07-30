"""Thin CLI for the frozen PCQM Route B official-validation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_route_b_evaluation import (
    OfficialValidConfig,
    evaluate_official_valid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument("--graph-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    metrics = evaluate_official_valid(
        labels_path=args.labels.resolve(),
        payload_dir=args.payload_dir.resolve(),
        fusion_dir=args.fusion_dir.resolve(),
        graph_cache_dir=args.graph_cache_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        config=OfficialValidConfig(workers=args.workers),
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
