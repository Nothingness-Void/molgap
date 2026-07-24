"""Residual analysis of the expansion500k hybrid on the common-eval set.

This script reads the common-eval prediction CSV and writes three reproducible
artifacts:

- markdown summary
- structured JSON
- worst Gap offenders CSV

The key question is whether the remaining expansion500k error is broad and
uniform, or concentrated in a small chemistry/geometry tail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
except Exception:  # pragma: no cover - keeps the analysis readable if RDKit is absent
    Chem = None
    rdMolDescriptors = None


DEFAULT_CSV = Path("results/phase8/full_expansion500k_common_eval_predictions.csv")
DEFAULT_MD = Path("results/phase8/residual_analysis_expansion500k.md")
DEFAULT_JSON = Path("results/phase8/residual_analysis_expansion500k.json")
DEFAULT_WORST = Path("results/phase8/residual_analysis_expansion500k_worst.csv")
PRED_V3 = "expansion500k_full_hybrid"
MODELS = {
    "Phase 7": "phase7_full_hybrid",
    "replacement300k": "replacement300k_full_hybrid",
    "expansion500k": PRED_V3,
}
TARGETS = ("homo", "lumo", "gap")


def _float(v: Any, ndigits: int = 6) -> float:
    return round(float(v), ndigits)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def _format_cell(v: Any) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4f}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def _md_table(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    rows = [[_format_cell(v) for v in row] for row in df.to_numpy()]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def _add_rdkit_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    smiles = df["canonical_smiles"].fillna("").astype(str)
    upper = smiles.str.upper()
    df["has_S"] = upper.str.contains("S")
    df["has_Cl"] = upper.str.contains("CL")
    df["n_S"] = upper.str.count("S")
    df["n_Cl"] = upper.str.count("CL")
    df["n_N"] = upper.str.count("N")
    df["n_O"] = upper.str.count("O")
    df["n_rotatable"] = np.nan
    df["n_aromatic_rings"] = np.nan

    if Chem is None or rdMolDescriptors is None:
        return df

    rot, arom, counts = [], [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rot.append(np.nan)
            arom.append(np.nan)
            counts.append({})
            continue
        rot.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        arom.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        atom_counts: dict[str, int] = {}
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            atom_counts[sym] = atom_counts.get(sym, 0) + 1
        counts.append(atom_counts)

    df["n_rotatable"] = rot
    df["n_aromatic_rings"] = arom
    for sym in ("S", "Cl", "N", "O"):
        df[f"n_{sym}"] = [c.get(sym, 0) for c in counts]
    df["has_S"] = df["n_S"] > 0
    df["has_Cl"] = df["n_Cl"] > 0
    return df


def _target_metrics(df: pd.DataFrame, model_col: str, target: str) -> dict[str, float]:
    y_true = df[target].to_numpy()
    y_pred = df[f"{model_col}_{target}"].to_numpy()
    return {"mae": _float(_mae(y_true, y_pred)), "bias": _float(_bias(y_true, y_pred))}


def _overall_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket in sorted(df["eval_set"].unique()):
        sub = df[df["eval_set"] == bucket]
        row: dict[str, Any] = {"bucket": bucket, "n": len(sub)}
        for target in TARGETS:
            metric = _target_metrics(sub, PRED_V3, target)
            row[f"{target}_mae"] = _float(metric["mae"], 4)
            row[f"{target}_bias"] = _float(metric["bias"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _bin_v3(df: pd.DataFrame, col: str, bins: list[float], labels: list[str]) -> pd.DataFrame:
    cats = pd.cut(df[col], bins=bins, labels=labels)
    y_true = df["gap"].to_numpy()
    y_pred = df[f"{PRED_V3}_gap"].to_numpy()
    err = np.abs(y_true - y_pred)
    rows = []
    for label in labels:
        mask = (cats == label).to_numpy()
        if not mask.sum():
            continue
        rows.append({
            "bin": label,
            "n": int(mask.sum()),
            "mae": _float(err[mask].mean(), 4),
            "bias": _float((y_pred[mask] - y_true[mask]).mean(), 4),
        })
    return pd.DataFrame(rows)


def _bin_model_comparison(
    df: pd.DataFrame,
    col: str,
    bins: list[float],
    labels: list[str],
) -> pd.DataFrame:
    cats = pd.cut(df[col], bins=bins, labels=labels)
    rows = []
    for label in labels:
        mask = (cats == label).to_numpy()
        if not mask.sum():
            continue
        row: dict[str, Any] = {"bin": label, "n": int(mask.sum())}
        y_true = df.loc[mask, "gap"].to_numpy()
        for name, model_col in MODELS.items():
            y_pred = df.loc[mask, f"{model_col}_gap"].to_numpy()
            row[f"{name}_gap_mae"] = _float(_mae(y_true, y_pred), 4)
        row["v3_minus_v2"] = _float(row["expansion500k_gap_mae"] - row["replacement300k_gap_mae"], 4)
        row["v3_minus_p7"] = _float(row["expansion500k_gap_mae"] - row["Phase 7_gap_mae"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _flag_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_true = df["gap"].to_numpy()
    y_pred = df[f"{PRED_V3}_gap"].to_numpy()
    err = np.abs(y_true - y_pred)
    for flag in ("has_S", "has_Cl"):
        for value in (True, False):
            mask = (df[flag] == value).to_numpy()
            if not mask.sum():
                continue
            rows.append({
                "flag": flag,
                "value": value,
                "n": int(mask.sum()),
                "gap_mae": _float(err[mask].mean(), 4),
            })
    return pd.DataFrame(rows)


def _error_concentration(df: pd.DataFrame) -> dict[str, float | int]:
    err = np.abs(df["gap"].to_numpy() - df[f"{PRED_V3}_gap"].to_numpy())
    k = max(1, len(err) // 10)
    top = np.sort(err)[::-1][:k].sum()
    return {
        "n": int(len(err)),
        "worst_10pct_n": int(k),
        "worst_10pct_error_share_pct": _float(100 * top / err.sum(), 2),
        "median": _float(np.median(err), 4),
        "mean": _float(err.mean(), 4),
        "p90": _float(np.percentile(err, 90), 4),
        "p99": _float(np.percentile(err, 99), 4),
    }


def _train_distribution() -> list[dict[str, Any]]:
    rows = []
    for path in (
        Path("data/raw/phase8_replacement_300k.csv"),
        Path("data/raw/phase8_expansion_500k.csv"),
    ):
        if not path.exists():
            continue
        d = pd.read_csv(path, usecols=["gap", "mw"])
        rows.append({
            "csv": str(path),
            "n": int(len(d)),
            "gap_lt_3_n": int((d["gap"] < 3).sum()),
            "gap_lt_3_pct": _float(100 * (d["gap"] < 3).mean(), 2),
            "mw_gt_800_n": int((d["mw"] > 800).sum()),
            "mw_gt_800_pct": _float(100 * (d["mw"] > 800).mean(), 2),
            "gap_p01": _float(d["gap"].quantile(0.01), 3),
            "gap_p05": _float(d["gap"].quantile(0.05), 3),
            "gap_p50": _float(d["gap"].quantile(0.50), 3),
            "gap_p99": _float(d["gap"].quantile(0.99), 3),
        })
    return rows


def _worst_offenders(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.copy()
    df["gap_abserr"] = np.abs(df["gap"] - df[f"{PRED_V3}_gap"])
    cols = [
        "eval_set", "cid", "mw", "gap", f"{PRED_V3}_gap", "gap_abserr",
        "n_rotatable", "n_aromatic_rings", "n_S", "n_Cl", "n_N", "n_O",
        "canonical_smiles",
    ]
    cols = [c for c in cols if c in df.columns]
    return df.nlargest(n, "gap_abserr")[cols].copy()


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def _build_markdown(
    args: argparse.Namespace,
    n_rows: int,
    bucket_counts: dict[str, int],
    overall: pd.DataFrame,
    gap_bins_v3: pd.DataFrame,
    mw_bins_v3: pd.DataFrame,
    gap_bins_cmp: pd.DataFrame,
    mw_bins_cmp: pd.DataFrame,
    flags: pd.DataFrame,
    concentration: dict[str, Any],
    train_dist: list[dict[str, Any]],
    worst: pd.DataFrame,
) -> str:
    train_df = pd.DataFrame(train_dist)
    worst_preview = worst.head(15).copy()
    for col in ("mw", "gap", f"{PRED_V3}_gap", "gap_abserr"):
        if col in worst_preview.columns:
            worst_preview[col] = worst_preview[col].round(2)

    return f"""# Phase 8 — Residual analysis of the expansion500k hybrid (v3)

Date: 2026-06-30

Script: `scripts/phase8/archive/legacy/data_coverage/residual_analysis_expansion500k.py`
Input: `{args.csv}` ({n_rows} molecules; buckets: {bucket_counts}).

## Why this analysis

This checks whether the remaining B3LYP-surrogate residual is broad and uniform,
or concentrated in a small chemistry/geometry tail. It also compares the same
bins across Phase 7, replacement300k, and expansion500k so the conclusion does
not rely on v3 in isolation.

## Headline numbers for v3

{_md_table(overall)}

Gap bias is near-zero at the bucket level. There is no simple global offset to
correct.

## Error concentration

- median Gap abs-err: **{concentration['median']:.4f}** eV
- mean Gap abs-err: **{concentration['mean']:.4f}** eV
- p90 / p99: **{concentration['p90']:.4f} / {concentration['p99']:.4f}** eV
- worst 10% of molecules ({concentration['worst_10pct_n']}/{concentration['n']})
  hold **{concentration['worst_10pct_error_share_pct']:.1f}%** of total Gap
  absolute error

The residual is tail-heavy rather than uniform.

## V3 residual bins

### By true Gap

{_md_table(gap_bins_v3)}

### By molecular weight

{_md_table(mw_bins_v3)}

### By heteroatom presence

{_md_table(flags)}

S/Cl flags are flat, so the earlier low-S/Cl coverage issue is no longer the
dominant bottleneck.

## P7 vs v2 vs v3 by hard bins

### Gap bins

{_md_table(gap_bins_cmp)}

### Molecular-weight bins

{_md_table(mw_bins_cmp)}

Important nuance: v3 already improves the hard bins substantially. For example,
Gap 2-3 eV improves from replacement300k to expansion500k, and MW>800 also
improves. The remaining tail is therefore not evidence that targeted expansion
failed; it is evidence that the next B3LYP-only round has lower expected ROI.

## Training-set coverage check

{_md_table(train_df) if not train_df.empty else 'Local training CSVs not available.'}

Expansion500k increased both low-gap and very-large molecule coverage versus
replacement300k. The remaining residual persists despite that broader coverage.

## Worst Gap offenders

Full table: `{args.worst_csv}`.

{_md_table(worst_preview)}

The worst rows are enriched for narrow-gap, very large, flexible, or otherwise
structurally difficult molecules. They are not exclusively large molecules, so
the actionable diagnosis is the intersection of low Gap, size/flexibility, and
geometry sensitivity rather than molecular weight alone.

## Conclusion

Expansion500k is a real improvement over replacement300k in the exact bins where
the model is still weakest. The remaining error is concentrated in a small tail,
especially Gap <3 eV and MW>800/flexible structures. That tail overlaps two known
limits:

1. narrow-gap / charge-transfer chemistry, where B3LYP labels themselves are the
   method ceiling and GW Delta-learning is the right next accuracy lever;
2. very large flexible molecules, where ETKDG geometry quality can become a
   separate edge case.

So the conservative decision is: keep v3 as the B3LYP surrogate default, stop
head-swap/longer-B3LYP-training loops by default, and move the main work to
Phase 9/10 re-validation against v3. Another B3LYP targeted top-up is not
impossible, but it is now lower ROI than GW Delta-learning.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual analysis for expansion500k common eval")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--worst-csv", type=Path, default=DEFAULT_WORST)
    parser.add_argument("--worst-n", type=int, default=50)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    df = _add_rdkit_features(df)
    bucket_counts = {k: int(v) for k, v in df["eval_set"].value_counts().items()}

    gap_bins = [-1, 1, 2, 3, 4, 5, 6, 100]
    gap_labels = ["<1", "1-2", "2-3", "3-4", "4-5", "5-6", ">6"]
    mw_bins = [0, 300, 400, 500, 600, 800, 1e9]
    mw_labels = ["<300", "300-400", "400-500", "500-600", "600-800", ">800"]

    overall = _overall_by_bucket(df)
    gap_bins_v3 = _bin_v3(df, "gap", gap_bins, gap_labels)
    mw_bins_v3 = _bin_v3(df, "mw", mw_bins, mw_labels)
    gap_bins_cmp = _bin_model_comparison(df, "gap", gap_bins, gap_labels)
    mw_bins_cmp = _bin_model_comparison(df, "mw", mw_bins, mw_labels)
    flags = _flag_report(df)
    concentration = _error_concentration(df)
    train_dist = _train_distribution()
    worst = _worst_offenders(df, args.worst_n)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.worst_csv.parent.mkdir(parents=True, exist_ok=True)

    worst.to_csv(args.worst_csv, index=False, encoding="utf-8")
    payload = {
        "input_csv": str(args.csv),
        "n": int(len(df)),
        "bucket_counts": bucket_counts,
        "overall_v3": _records(overall),
        "gap_bins_v3": _records(gap_bins_v3),
        "mw_bins_v3": _records(mw_bins_v3),
        "gap_bins_model_comparison": _records(gap_bins_cmp),
        "mw_bins_model_comparison": _records(mw_bins_cmp),
        "heteroatom_flags": _records(flags),
        "error_concentration": concentration,
        "train_distribution": train_dist,
        "worst_csv": str(args.worst_csv),
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(
        _build_markdown(
            args,
            len(df),
            bucket_counts,
            overall,
            gap_bins_v3,
            mw_bins_v3,
            gap_bins_cmp,
            mw_bins_cmp,
            flags,
            concentration,
            train_dist,
            worst,
        ),
        encoding="utf-8",
    )

    print(f"Rows: {len(df)} buckets={bucket_counts}")
    print(f"Markdown -> {args.out_md}")
    print(f"JSON -> {args.out_json}")
    print(f"Worst offenders -> {args.worst_csv}")
    print(
        "Gap MAE all: "
        + " ".join(
            f"{name}={_mae(df['gap'].to_numpy(), df[f'{col}_gap'].to_numpy()):.4f}"
            for name, col in MODELS.items()
        )
    )
    print(
        f"Tail: worst 10% holds {concentration['worst_10pct_error_share_pct']:.1f}% "
        f"of Gap abs-error; median={concentration['median']:.4f}, "
        f"p90={concentration['p90']:.4f}"
    )


if __name__ == "__main__":
    main()
