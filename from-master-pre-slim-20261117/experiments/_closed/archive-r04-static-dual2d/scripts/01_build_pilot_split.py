"""Build the 30k dual-2D static candidate architecture pilot from existing expansion500k caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.pubchemqc import sha256_file
from molgap.utils import ensure_dirs
from molgap.archive.phase8_r04_static_dual2d.data import PILOT_COUNTS, scaffold_split_balanced, select_balanced_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_dir = RESULTS_DIR / "phase8" / "archive" / "archive-r04-static-dual2d"
    ensure_dirs(out_dir)
    expansion_path = RAW_DIR / "phase8_expansion_500k.csv"
    topup_paths = [
        RAW_DIR / "phase8_v3_topup_balanced_200k.csv",
        RAW_DIR / "phase8_v3_topup_general_60k.csv",
    ]
    expansion = pd.read_csv(expansion_path)
    topup = pd.concat([pd.read_csv(path) for path in topup_paths], ignore_index=True)
    pilot = select_balanced_pilot(expansion, topup, seed=args.seed)
    pilot = scaffold_split_balanced(pilot, workers=args.workers, seed=args.seed)
    out_path = out_dir / "pilot_30k.parquet"
    pilot.to_parquet(out_path, index=False)
    split_scaffolds = {
        name: set(part.scaffold) for name, part in pilot.groupby("split")
    }
    overlaps = {
        f"{left}_{right}": len(split_scaffolds[left] & split_scaffolds[right])
        for i, left in enumerate(split_scaffolds)
        for right in list(split_scaffolds)[i + 1:]
    }
    manifest = {
        "purpose": "architecture feasibility only; not a final 700k result",
        "weights_initialization": "random; no old checkpoints",
        "source_pool": str(expansion_path),
        "source_pool_sha256": sha256_file(expansion_path),
        "sampling_target": PILOT_COUNTS,
        "counts": {
            "total": len(pilot),
            "by_source": pilot.sampling_source.value_counts().to_dict(),
            "by_split": pilot.split.value_counts().to_dict(),
            "source_by_split": pd.crosstab(
                pilot.split, pilot.sampling_source
            ).to_dict(orient="index"),
        },
        "unique": {
            "cid": int(pilot.cid.nunique()),
            "canonical_smiles": int(pilot.canonical_smiles.nunique()),
            "scaffold": int(pilot.scaffold.nunique()),
        },
        "scaffold_overlap": overlaps,
        "label_consistency_max_abs_eV": float(
            (pilot.gap - (pilot.lumo - pilot.homo)).abs().max()
        ),
        "seed": args.seed,
        "output_sha256": sha256_file(out_path),
    }
    (out_dir / "pilot_split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=2), flush=True)
    print(json.dumps({"scaffold_overlap": overlaps}, indent=2), flush=True)


if __name__ == "__main__":
    main()
