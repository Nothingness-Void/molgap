"""Analyze GW Δ-model residuals by chemistry strata.

Input is the scaffold-test predictions CSV emitted by train_delta.py. The script
does not retrain; it identifies where the v3 Δ model still leaves error so a
future per-stratum model has a concrete target rather than guessing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.metrics import mean_absolute_error, r2_score

TARGETS = ("homo", "lumo", "gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def add_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for smi in df["smiles"].astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({"mw": np.nan, "rot_bonds": np.nan, "arom_rings": np.nan, "tpsa": np.nan})
            continue
        rows.append({
            "mw": Descriptors.MolWt(mol),
            "rot_bonds": Lipinski.NumRotatableBonds(mol),
            "arom_rings": Lipinski.NumAromaticRings(mol),
            "tpsa": Descriptors.TPSA(mol),
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def metrics(sub: pd.DataFrame, pred_prefix: str) -> dict:
    row = {"n": int(len(sub))}
    for target in TARGETS:
        y = sub[f"gw_{target}"].to_numpy(dtype=np.float64)
        p = sub[f"{pred_prefix}_{target}"].to_numpy(dtype=np.float64)
        row[f"{target}_mae"] = float(mean_absolute_error(y, p))
        row[f"{target}_r2"] = float(r2_score(y, p)) if len(sub) > 1 else None
    row["avg_mae"] = float(np.mean([row[f"{t}_mae"] for t in TARGETS]))
    return row


def bucket_rows(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    gap = df["pred_gap"]
    mw = df["mw"]
    rot = df["rot_bonds"]
    arom = df["arom_rings"]
    return [
        ("all", pd.Series(True, index=df.index)),
        ("pred_gap_lt6", gap < 6.0),
        ("pred_gap_6_8", (gap >= 6.0) & (gap < 8.0)),
        ("pred_gap_ge8", gap >= 8.0),
        ("mw_lt400", mw < 400),
        ("mw_400_700", (mw >= 400) & (mw < 700)),
        ("mw_ge700", mw >= 700),
        ("rot_ge8", rot >= 8),
        ("arom_ge4", arom >= 4),
        ("flexible_large", (rot >= 8) & (mw >= 500)),
        ("large_aromatic", (mw >= 500) & (arom >= 4)),
    ]


def main() -> None:
    args = parse_args()
    df = add_descriptors(pd.read_csv(args.predictions))
    rows = []
    for bucket, mask in bucket_rows(df):
        sub = df.loc[mask.fillna(False)].copy()
        if len(sub) < 10:
            continue
        lgbm = metrics(sub, "gw_pred_lgbm_delta")
        const = metrics(sub, "gw_pred_const")
        raw = metrics(sub, "gw_pred_raw")
        rows.append({
            "bucket": bucket,
            "n": int(len(sub)),
            "raw_avg_mae": raw["avg_mae"],
            "const_avg_mae": const["avg_mae"],
            "lgbm_avg_mae": lgbm["avg_mae"],
            "lgbm_gap_mae": lgbm["gap_mae"],
            "delta_lgbm_minus_const_avg": lgbm["avg_mae"] - const["avg_mae"],
            "delta_lgbm_minus_const_gap": lgbm["gap_mae"] - const["gap_mae"],
        })

    out = {"predictions": str(args.predictions), "rows": rows}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# V3 GW Delta Strata Analysis",
        "",
        f"Input: `{args.predictions}`",
        "",
        "| bucket | n | raw avg MAE | const avg MAE | LGBM avg MAE | LGBM Gap MAE | Δ avg vs const | Δ Gap vs const |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['bucket']} | {r['n']} | {r['raw_avg_mae']:.3f} | "
            f"{r['const_avg_mae']:.3f} | {r['lgbm_avg_mae']:.3f} | "
            f"{r['lgbm_gap_mae']:.3f} | {r['delta_lgbm_minus_const_avg']:+.3f} | "
            f"{r['delta_lgbm_minus_const_gap']:+.3f} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {args.out_json}")
    print(f"Saved {args.out_md}")


if __name__ == "__main__":
    main()
