"""Stage aligned FP16 dual-GPS prefixes for 2M-2D plus 1M-3D fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.multi2d import stage_dual_gps_embedding_prefixes


def parse_expert(value: str) -> tuple[str, tuple[Path, Path]]:
    if "=" not in value or "," not in value:
        raise argparse.ArgumentTypeError("Expected NAME=GPS7_EMB,GPS9_EMB")
    name, paths = value.split("=", 1)
    gps7, gps9 = paths.split(",", 1)
    return name, (Path(gps7), Path(gps9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", action="append", type=parse_expert, required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = stage_dual_gps_embedding_prefixes(
        dict(args.expert), rows=args.rows, out_dir=args.out_dir, report_path=args.report
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
