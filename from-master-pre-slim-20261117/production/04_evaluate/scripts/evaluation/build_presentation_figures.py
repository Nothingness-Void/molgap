"""Thin CLI for the frozen presentation figure package."""
from __future__ import annotations

import argparse
from pathlib import Path

from molgap.constants import EVALUATE_DIR
from molgap.presentation_figures import build_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(EVALUATE_DIR / "project_freeze" / "figures"))
    parser.add_argument("--theme", choices=("light", "dark"), default="light")
    args = parser.parse_args()
    freeze_dir = EVALUATE_DIR / "project_freeze"
    output_dir = Path(args.output)
    build_all(
        evidence_path=freeze_dir / "presentation_evidence" / "presentation_evidence.json",
        latency_dir=freeze_dir / "inference_latency",
        output_dir=output_dir,
        source_dir=freeze_dir / "figures" / "source",
        theme=args.theme,
    )
    print(f"Wrote {args.theme} presentation figures to {args.output}")


if __name__ == "__main__":
    main()
