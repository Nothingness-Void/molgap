"""Thin IMS adapter for a consolidated Colab Fusion payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_route_b_fusion import (
    ENCODER_NAMES,
    export_consolidated_fusion_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = export_consolidated_fusion_payload(
        encoder_dirs={
            name: args.encoder_root / name for name in ENCODER_NAMES
        },
        output_dir=args.output_dir,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
