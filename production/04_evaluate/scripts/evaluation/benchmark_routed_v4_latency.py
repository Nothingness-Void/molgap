"""Measure local warm end-to-end latency for the registered routed v4 model."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.constants import EVALUATE_DIR
from molgap.inference_benchmark import (
    DEFAULT_SMILES,
    benchmark_routed_v4,
    write_benchmark_artifacts,
)


def _smiles_source(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_SMILES)
    return path.read_text(encoding="utf-8").splitlines()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="Torch device, default: CUDA when available")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--batch-2d", type=int, default=256)
    parser.add_argument("--batch-3d", type=int, default=128)
    parser.add_argument("--smiles-file", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            EVALUATE_DIR
            / "project_freeze"
            / "inference_latency"
            / "routed_gps7_gps9_schnet_500k_v4_local.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark_routed_v4(
        device=args.device,
        batch_sizes=args.batch_sizes,
        repeats=args.repeats,
        warmups=args.warmups,
        batch_2d=args.batch_2d,
        batch_3d=args.batch_3d,
        smiles_source=_smiles_source(args.smiles_file),
    )
    output_json, output_markdown = write_benchmark_artifacts(result, args.output)
    print(output_json)
    print(output_markdown)


if __name__ == "__main__":
    main()
