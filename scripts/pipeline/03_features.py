"""
03_features.py — generate molecular features for MolGap.

Input:
  data/processed/pubchemqc_chon_mw200_300_clean.csv

Output:
  data/processed/features_morgan2048_desc.csv

Features:
  - Morgan fingerprint / ECFP4, radius=2, 2048 bits
  - RDKit 2D descriptors via Descriptors.CalcMolDescriptors

The output keeps metadata and target columns so downstream scripts can train and
write traceable prediction tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors

from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import (
    METADATA_COLS,
    PROCESSED_DIR,
    TARGET_COLS,
    build_feature_rows_parallel,
    ensure_dirs,
)


DEFAULT_INPUT = PROCESSED_DIR / "pubchemqc_chon_mw200_300_clean.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "features_morgan2048_desc.csv"
ALL_DESC_NAMES = [name for name, _ in Descriptors._descList]


def build_features(
    input_path: Path,
    output_path: Path,
    radius: int = 2,
    n_bits: int = 2048,
    max_missing_ratio: float = 0.5,
) -> pd.DataFrame:
    """Build Morgan + descriptor features and save a CSV."""
    print(f"Loading clean data: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  clean rows: {len(df)}")

    required = METADATA_COLS + TARGET_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    smiles_list = df["canonical_smiles"].tolist()
    print(f"  generating features with multiprocessing...")
    results = build_feature_rows_parallel(smiles_list, radius=radius, n_bits=n_bits)
    failed = len(df) - len(results)

    rows = []
    for orig_idx, generated in results:
        feature_row = {c: df.iloc[orig_idx][c] for c in required}
        feature_row.update(generated)
        rows.append(feature_row)

    feat = pd.DataFrame(rows)
    if feat.empty:
        raise RuntimeError("No valid feature rows were generated.")

    feature_cols = [c for c in feat.columns if c not in required]
    feat[feature_cols] = feat[feature_cols].replace([np.inf, -np.inf], np.nan)

    desc_cols = [c for c in feature_cols if c.startswith("desc_")]
    original_desc_count = len(desc_cols)

    missing_ratio = feat[desc_cols].isna().mean() if desc_cols else pd.Series(dtype=float)
    high_missing_cols = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
    if high_missing_cols:
        feat = feat.drop(columns=high_missing_cols)
        print(f"  dropped high-missing descriptor columns: {len(high_missing_cols)}")

    feature_cols = [c for c in feat.columns if c not in required]
    constant_cols = []
    for col in tqdm(feature_cols, desc="Drop constant features", unit="col", mininterval=0.5):
        if feat[col].dropna().nunique() <= 1:
            constant_cols.append(col)
    if constant_cols:
        feat = feat.drop(columns=constant_cols)
        print(f"  dropped constant feature columns: {len(constant_cols)}")

    feature_cols = [c for c in feat.columns if c not in required]
    nan_before = int(feat[feature_cols].isna().sum().sum())
    for col in tqdm(feature_cols, desc="Fill missing features", unit="col", mininterval=0.5):
        if feat[col].isna().any():
            median = feat[col].median()
            if pd.isna(median):
                median = 0.0
            feat[col] = feat[col].fillna(median)
    nan_after = int(feat[feature_cols].isna().sum().sum())

    ensure_dirs(output_path.parent)
    feat.to_csv(output_path, index=False, encoding="utf-8")

    print("\n=== FEATURE SUMMARY ===")
    print(f"input rows             : {len(df)}")
    print(f"feature rows           : {len(feat)}")
    print(f"failed molecules       : {failed}")
    print(f"Morgan bits requested  : {n_bits}")
    print(f"RDKit descriptors raw  : {original_desc_count}")
    print(f"final feature columns  : {len(feature_cols)}")
    print(f"NaN filled             : {nan_before} -> {nan_after}")
    print(f"output                 : {output_path}")
    return feat


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MolGap molecular features")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--max-missing-ratio", type=float, default=0.5)
    args = parser.parse_args()

    build_features(
        input_path=args.input,
        output_path=args.output,
        radius=args.radius,
        n_bits=args.n_bits,
        max_missing_ratio=args.max_missing_ratio,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
