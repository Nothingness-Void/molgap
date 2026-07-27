"""Freeze repaired-2M scaffold folds and prepare, but do not submit, OOF jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.oof_planning import build_oof_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=Path("data/raw/phase8_repaired_2m.csv")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/repaired_2m_scaling/results/repaired_2m_manifest.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/repaired_2m_scaling/results/gps7_gps9_oof"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            build_oof_plan(
                args.csv,
                args.manifest,
                args.out_dir,
                n_folds=args.folds,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
