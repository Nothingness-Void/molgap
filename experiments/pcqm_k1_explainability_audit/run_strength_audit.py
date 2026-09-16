"""Thin CLI for the frozen K1 Round-2 layer-6 strength audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.pcqm_k1_strength_audit import run_strength_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-payload", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = run_strength_audit(
        args.cache_root,
        args.checkpoint,
        args.reference_payload,
        args.output_root,
        batch_size=args.batch_size,
    )
    print(result["format"], result["complete"], result["development_rows"])


if __name__ == "__main__":
    main()
