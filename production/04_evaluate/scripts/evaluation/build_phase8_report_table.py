"""Generate the human-readable Phase 8 comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

from molgap.constants import EVALUATE_DIR, REPO_ROOT
from molgap.phase8_reporting import build_comparison_rows, write_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVALUATE_DIR / "overview" / "phase8_reporting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    rows = build_comparison_rows(REPO_ROOT)
    write_comparison(
        rows,
        out_dir / "model_comparison.csv",
        out_dir / "model_comparison.md",
    )
    print(out_dir / "model_comparison.md")


if __name__ == "__main__":
    main()
