"""Recover, deduplicate, and scaffold-seal a partial residual fetch round."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from molgap.utils import scaffold_split_key


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sealed-target", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV shards in {args.input_dir}")
    parts = []
    for path in paths:
        part = pd.read_csv(path)
        part["source_shard"] = path.name
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    raw_rows = len(frame)
    frame = frame.drop_duplicates("cid", keep="first")
    frame = frame.drop_duplicates("canonical_smiles", keep="first").reset_index(drop=True)
    if frame[["cid", "canonical_smiles", "homo", "lumo", "gap"]].isna().any().any():
        raise ValueError("Recovered rows contain missing identity or target values")

    frame["scaffold"] = [scaffold_split_key(smiles) for smiles in frame.canonical_smiles]
    scaffold_sizes = frame.groupby("scaffold").size().to_dict()
    ordered = sorted(scaffold_sizes, key=lambda value: stable_key(str(value), args.seed))
    sealed_scaffolds: set[str] = set()
    sealed_rows = 0
    for scaffold in ordered:
        size = int(scaffold_sizes[scaffold])
        if sealed_rows >= args.sealed_target:
            break
        if sealed_rows and abs((sealed_rows + size) - args.sealed_target) > abs(sealed_rows - args.sealed_target):
            continue
        sealed_scaffolds.add(scaffold)
        sealed_rows += size

    sealed_mask = frame.scaffold.isin(sealed_scaffolds)
    sealed = frame.loc[sealed_mask].reset_index(drop=True)
    development = frame.loc[~sealed_mask].reset_index(drop=True)
    if set(sealed.scaffold) & set(development.scaffold):
        raise AssertionError("Scaffold leakage between development and sealed rows")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(frame, args.out_dir / "residual_target_round01_recovered.csv")
    atomic_csv(development, args.out_dir / "residual_target_round01_development.csv")
    atomic_csv(sealed, args.out_dir / "residual_target_round01_sealed.csv")
    report = {
        "source_shards": [str(path) for path in paths],
        "raw_rows": raw_rows,
        "deduplicated_rows": int(len(frame)),
        "duplicate_rows_removed": int(raw_rows - len(frame)),
        "development_rows": int(len(development)),
        "sealed_rows": int(len(sealed)),
        "development_unique_scaffolds": int(development.scaffold.nunique()),
        "sealed_unique_scaffolds": int(sealed.scaffold.nunique()),
        "scaffold_overlap": 0,
        "bucket_counts_all": {str(k): int(v) for k, v in frame.bucket.value_counts().items()},
        "bucket_counts_development": {str(k): int(v) for k, v in development.bucket.value_counts().items()},
        "bucket_counts_sealed": {str(k): int(v) for k, v in sealed.bucket.value_counts().items()},
    }
    atomic_json(report, args.out_dir / "recovery_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
