"""Thin IMS adapter for the development-only PCQM Route B fusion screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_route_b_fusion import (
    ENCODER_NAMES,
    FusionConfig,
    preflight_fusion,
    train_fusion_screen,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    encoder_dirs = {
        name: args.encoder_root / name for name in ENCODER_NAMES
    }
    if args.preflight:
        result = preflight_fusion(
            encoder_dirs=encoder_dirs,
            output_path=args.output_dir / "preflight.json",
            config=FusionConfig(),
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    result = train_fusion_screen(
        encoder_dirs=encoder_dirs,
        output_dir=args.output_dir,
        config=FusionConfig(),
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
