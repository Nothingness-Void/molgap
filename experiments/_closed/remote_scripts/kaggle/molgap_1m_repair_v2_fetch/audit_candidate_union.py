"""Count the strict CID/SMILES union of ordered candidate-pool CSV sources."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def key(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, action="append", required=True,
                    help="Ordered source CSV; earlier sources win a duplicate tie")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    seen_cids: set[str] = set()
    seen_smiles: set[str] = set()
    sources: list[dict] = []
    for path in args.csv:
        total = 0
        new_rows = 0
        duplicates = 0
        for chunk in pd.read_csv(path, usecols=lambda col: col in {"cid", "canonical_smiles"},
                                 dtype={"cid": "string", "canonical_smiles": "string"}, chunksize=100_000):
            for row in chunk.itertuples(index=False):
                cid = key(getattr(row, "cid", None))
                smiles = key(getattr(row, "canonical_smiles", None))
                duplicate = (cid is not None and cid in seen_cids) or (smiles is not None and smiles in seen_smiles)
                total += 1
                if duplicate:
                    duplicates += 1
                    continue
                if cid is not None:
                    seen_cids.add(cid)
                if smiles is not None:
                    seen_smiles.add(smiles)
                new_rows += 1
        sources.append({"csv": str(path), "rows": total, "new_unique_rows": new_rows, "overlap_rows": duplicates})

    result = {"unique_rows": len(seen_smiles), "sources": sources}
    atomic_json(args.out_json, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
