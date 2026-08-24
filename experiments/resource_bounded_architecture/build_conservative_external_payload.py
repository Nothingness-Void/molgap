"""Build the compact Colab external-evaluation payload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from molgap.conservative_fusion_payload import build_external_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-gps-predictions", type=Path, required=True)
    parser.add_argument("--accepted-external-predictions", type=Path, required=True)
    parser.add_argument("--dense-gate", type=Path, action="append", required=True)
    parser.add_argument("--graph-cache", type=Path, required=True)
    parser.add_argument("--primary-checkpoint", type=Path, required=True)
    parser.add_argument("--augmented-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    report = build_external_payload(
        three_gps_predictions=args.three_gps_predictions,
        accepted_external_predictions=args.accepted_external_predictions,
        dense_gate_paths=args.dense_gate,
        graph_cache=args.graph_cache,
        primary_checkpoint=args.primary_checkpoint,
        augmented_checkpoint=args.augmented_checkpoint,
        output_path=args.output,
        manifest_path=args.manifest,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps({key: report[key] for key in ("status", "rows", "scope_rows")}))


if __name__ == "__main__":
    main()
