"""
02_clean.py — clean the slim PubChemQC CSV for MolGap.

Input:
  data/raw/pubchemqc_chon_mw200_300.csv

Output:
  data/processed/pubchemqc_chon_mw200_300_clean.csv

This step validates numeric targets, parses SMILES with RDKit, creates canonical
SMILES, removes duplicates, and checks the internal relation gap = lumo - homo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import PROCESSED_DIR, RAW_DIR, canonicalize_smiles, ensure_dirs


DEFAULT_INPUT = RAW_DIR / "pubchemqc_chon_mw200_300.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "pubchemqc_chon_mw200_300_clean.csv"
REQUIRED_COLS = ["cid", "mw", "formula", "smiles", "homo", "lumo", "gap"]
NUMERIC_COLS = ["cid", "mw", "homo", "lumo", "gap"]


def clean_pubchemqc(input_path: Path, output_path: Path, gap_tol: float = 1e-6) -> pd.DataFrame:
    """Clean a slim PubChemQC CSV and write the processed result."""
    print(f"Loading raw data: {input_path}")
    df = pd.read_csv(input_path)
    raw_n = len(df)
    print(f"  raw rows: {raw_n}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLS].copy()

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["cid", "mw", "homo", "lumo", "gap", "smiles", "formula"])
    print(f"  removed missing required values: {before - len(df)}")

    before = len(df)
    df = df[df["gap"] > 0].copy()
    print(f"  removed non-positive gap rows: {before - len(df)}")

    df["gap_residual"] = ((df["lumo"] - df["homo"]) - df["gap"]).abs()
    before = len(df)
    df = df[df["gap_residual"] <= gap_tol].copy()
    print(f"  removed inconsistent gap rows (tol={gap_tol:g}): {before - len(df)}")

    print("Canonicalizing SMILES with RDKit...")
    df["canonical_smiles"] = df["smiles"].progress_map(canonicalize_smiles)
    before = len(df)
    df = df.dropna(subset=["canonical_smiles"]).copy()
    print(f"  removed invalid SMILES rows: {before - len(df)}")

    before = len(df)
    df = df.drop_duplicates(subset=["canonical_smiles"], keep="first").copy()
    print(f"  removed duplicate canonical SMILES rows: {before - len(df)}")

    df["cid"] = df["cid"].astype(int)
    df = df.sort_values("cid").reset_index(drop=True)

    output_cols = [
        "cid",
        "mw",
        "formula",
        "smiles",
        "canonical_smiles",
        "homo",
        "lumo",
        "gap",
        "gap_residual",
    ]
    ensure_dirs(output_path.parent)
    df[output_cols].to_csv(output_path, index=False, encoding="utf-8")

    print("\n=== CLEAN SUMMARY ===")
    print(f"raw rows       : {raw_n}")
    print(f"clean rows     : {len(df)}")
    print(f"removed total  : {raw_n - len(df)}")
    print(f"output         : {output_path}")
    return df[output_cols]


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean MolGap PubChemQC slim CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gap-tol", type=float, default=1e-6)
    args = parser.parse_args()

    clean_pubchemqc(args.input, args.output, args.gap_tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
