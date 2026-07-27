"""Train one recoverable Route B frozen-embedding fusion candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.route_b_fusion import (
    CANDIDATES,
    FUSION_MODES,
    train_route_b_fusion,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--gps7", type=Path)
    parser.add_argument("--gps9", type=Path, required=True)
    parser.add_argument("--gps11-160", dest="gps11_160", type=Path)
    parser.add_argument("--schnet-primary", type=Path, required=True)
    parser.add_argument("--schnet-augmented", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusion-mode", choices=FUSION_MODES, default="gated")
    parser.add_argument("--correction-scale-eV", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    paths = {
        name: path
        for name, path in {
            "gps7": args.gps7,
            "gps9": args.gps9,
            "gps11_160": args.gps11_160,
            "schnet_primary": args.schnet_primary,
            "schnet_augmented": args.schnet_augmented,
        }.items()
        if path is not None
    }
    print(
        json.dumps(
            train_route_b_fusion(
                paths,
                args.candidate,
                args.out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden=args.hidden,
                seed=args.seed,
                resume=args.resume,
                fusion_mode=args.fusion_mode,
                correction_scale_eV=args.correction_scale_eV,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
