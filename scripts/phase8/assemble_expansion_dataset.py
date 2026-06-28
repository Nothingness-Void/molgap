"""
Assemble a larger Phase 8 training CSV by appending targeted top-up rows.

This is for the v2 -> v3 expansion path:

  base 300k + targeted new rows -> 500k mixed training set

Unlike assemble_replacement_dataset.py, this script does not remove base rows.
It keeps the replay set intact to avoid catastrophic forgetting, drops duplicate
canonical SMILES/CIDs, and appends up to the requested final size.

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/assemble_expansion_dataset.py `
    --base-csv data/raw/phase8_replacement_300k.csv `
    --candidate-csv data/raw/phase8_v3_topup_200k.csv `
    --out-csv data/raw/phase8_expansion_500k.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.utils import canonicalize_smiles, ensure_dirs

BASE_CSV = RAW_DIR / "phase8_replacement_300k.csv"
OUT_CSV = RAW_DIR / "phase8_expansion_500k.csv"
REPORT_JSON = RESULTS_DIR / "phase8" / "expansion_500k_report.json"
REPORT_MD = RESULTS_DIR / "phase8" / "expansion_500k_report.md"
TRAIN_COLS = ["cid", "mw", "formula", "smiles", "homo", "lumo", "gap", "canonical_smiles"]
TARGET_COLS = ["homo", "lumo", "gap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, default=BASE_CSV)
    parser.add_argument("--candidate-csv", type=Path, action="append", required=True)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--target-size", type=int, default=500_000)
    return parser.parse_args()


def load_base(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "canonical_smiles" not in df:
        df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
    for col in ["mw", *TARGET_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["canonical_smiles", "smiles", *TARGET_COLS])
    df = df[df["gap"] > 0].drop_duplicates("canonical_smiles").reset_index(drop=True)
    return df[TRAIN_COLS].copy()


def load_candidates(paths: list[Path], base: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        if "canonical_smiles" not in df:
            df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=TRAIN_COLS + ["bucket", "source_file"])

    cand = pd.concat(frames, ignore_index=True, sort=False)
    for col in ["mw", *TARGET_COLS]:
        cand[col] = pd.to_numeric(cand[col], errors="coerce")
    cand["cid_num"] = pd.to_numeric(cand.get("cid"), errors="coerce")
    base_smiles = set(base["canonical_smiles"].dropna().astype(str))
    base_cids = set(pd.to_numeric(base["cid"], errors="coerce").dropna().astype(int))
    cand = cand.dropna(subset=["canonical_smiles", "smiles", "mw", *TARGET_COLS])
    cand = cand[cand["gap"] > 0].copy()
    cand = cand[~cand["canonical_smiles"].astype(str).isin(base_smiles)].copy()
    cand = cand[~cand["cid_num"].dropna().astype("Int64").isin(base_cids).reindex(cand.index, fill_value=False)]
    cand = cand.drop_duplicates("canonical_smiles").reset_index(drop=True)
    for col in TRAIN_COLS:
        if col not in cand:
            cand[col] = pd.NA
    if "bucket" not in cand:
        cand["bucket"] = "targeted"
    return cand


def summarize(df: pd.DataFrame) -> dict:
    out = {"n": int(len(df))}
    for col in ["mw", *TARGET_COLS]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        out[col] = {
            "mean": float(vals.mean()),
            "p1": float(vals.quantile(0.01)),
            "p50": float(vals.quantile(0.50)),
            "p99": float(vals.quantile(0.99)),
        }
    if "bucket" in df:
        out["bucket_counts"] = {str(k): int(v) for k, v in df["bucket"].value_counts().items()}
    return out


def write_md(report: dict, path: Path) -> None:
    lines = [
        "# Phase 8 Expansion 500K Dataset Report",
        "",
        f"- base csv: `{report['base_csv']}`",
        f"- output csv: `{report['out_csv']}`",
        f"- base rows: {report['base']['n']:,}",
        f"- usable candidate rows: {report['candidate_pool']['n']:,}",
        f"- appended rows: {report['appended']['n']:,}",
        f"- final rows: {report['expanded']['n']:,}",
        "",
        "## Appended Buckets",
        "",
        "| bucket | n |",
        "|---|---:|",
    ]
    for bucket, n in report["appended"].get("bucket_counts", {}).items():
        lines.append(f"| `{bucket}` | {n:,} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base = load_base(args.base_csv)
    candidates = load_candidates(args.candidate_csv, base)
    need = max(0, args.target_size - len(base))
    appended = candidates.head(need).copy()
    expanded = pd.concat([base, appended[TRAIN_COLS]], ignore_index=True, sort=False)
    expanded = expanded.drop_duplicates("canonical_smiles").reset_index(drop=True)
    if len(expanded) != len(base) + len(appended):
        raise RuntimeError("Unexpected duplicate after expansion assembly")

    ensure_dirs(args.out_csv.parent, args.report_json.parent, args.report_md.parent)
    expanded[TRAIN_COLS].to_csv(args.out_csv, index=False, encoding="utf-8")
    report = {
        "base_csv": str(args.base_csv),
        "candidate_csvs": [str(p) for p in args.candidate_csv],
        "out_csv": str(args.out_csv),
        "target_size": int(args.target_size),
        "base": summarize(base),
        "candidate_pool": summarize(candidates),
        "appended": summarize(appended),
        "expanded": summarize(expanded),
        "complete": bool(len(expanded) == args.target_size),
        "missing_rows": int(max(0, args.target_size - len(expanded))),
    }
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_md(report, args.report_md)
    print(f"Base rows: {len(base):,}")
    print(f"Usable candidates: {len(candidates):,}")
    print(f"Appended: {len(appended):,}")
    print(f"Expanded rows: {len(expanded):,}")
    if len(expanded) < args.target_size:
        print(f"WARNING: short by {args.target_size - len(expanded):,} rows")
    print(f"Saved {args.out_csv}")
    print(f"Saved {args.report_md}")


if __name__ == "__main__":
    main()
