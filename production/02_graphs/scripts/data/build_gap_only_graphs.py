"""Build a resumable 2D graph cache for a Gap-only table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from molgap.gap_specialization import build_gap_graph_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smiles-column", default="canonical_smiles")
    parser.add_argument("--gap-column", default="homolumogap")
    parser.add_argument("--shard-size", type=int, default=10_000)
    args = parser.parse_args()
    table = (
        pd.read_parquet(args.input)
        if args.input.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(args.input)
    )
    report = build_gap_graph_cache(
        table,
        smiles_column=args.smiles_column,
        gap_column=args.gap_column,
        out_path=args.out,
        progress_path=args.out.with_suffix(".progress"),
        shard_dir=args.out.with_name(f"{args.out.stem}_shards"),
        shard_size=args.shard_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
