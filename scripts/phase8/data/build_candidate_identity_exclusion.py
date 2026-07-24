"""Build a deduplicated CID/SMILES exclusion CSV from a selection report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()
    report = json.loads(args.selection_report.read_text(encoding="utf-8"))
    frames = []
    for raw_path in report["candidate_csvs"]:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, usecols=["cid", "canonical_smiles"], dtype="string"))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["cid", "canonical_smiles"])
    combined = combined.drop_duplicates(subset=["cid"], keep="first")
    combined = combined.drop_duplicates(subset=["canonical_smiles"], keep="first")
    if len(combined) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows:,} unique rows, got {len(combined):,}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    combined.to_csv(temporary, index=False)
    os.replace(temporary, args.out)
    print(json.dumps({"out": str(args.out), "rows": len(combined)}, indent=2))


if __name__ == "__main__":
    main()
