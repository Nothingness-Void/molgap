"""
Phase 8 v2 audit: summarize what replacement300k fixed and what remains hard.

Inputs are already-generated prediction CSVs; this script does not run model
inference. It compares Phase 7 hybrid vs replacement300k hybrid on:
  - the full common eval (OOD1000 + P8 targeted hard);
  - the PCQM4Mv2 valid proxy used for the P8.1 coverage diagnostic.

Outputs:
  results/phase8/v2_error_mode_analysis.json
  results/phase8/v2_error_mode_analysis.md
  results/phase8/v2_common_eval_remaining_worst.csv
  results/phase8/v2_pcqm_proxy_remaining_worst.csv

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/data_coverage/analyze_v2_error_modes.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from molgap.constants import RESULTS_DIR, TARGET_COLS

PHASE8_DIR = RESULTS_DIR / "phase8"
COMMON_CSV = PHASE8_DIR / "full_replacement_common_eval_predictions.csv"
PCQM_CSV = PHASE8_DIR / "pcqm4mv2_proxy_p7_vs_p8_predictions.csv"
OUT_JSON = PHASE8_DIR / "v2_error_mode_analysis.json"
OUT_MD = PHASE8_DIR / "v2_error_mode_analysis.md"
OUT_COMMON_WORST = PHASE8_DIR / "v2_common_eval_remaining_worst.csv"
OUT_PCQM_WORST = PHASE8_DIR / "v2_pcqm_proxy_remaining_worst.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-csv", type=Path, default=COMMON_CSV)
    parser.add_argument("--pcqm-csv", type=Path, default=PCQM_CSV)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    parser.add_argument("--common-worst-out", type=Path, default=OUT_COMMON_WORST)
    parser.add_argument("--pcqm-worst-out", type=Path, default=OUT_PCQM_WORST)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def safe_mol(smiles: object):
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            Chem.RemoveStereochemistry(mol)
        except Exception:
            pass
    return mol


def scaffold_smiles(mol) -> str:
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except RuntimeError:
        return ""


def molecule_features(smiles: object) -> dict:
    mol = safe_mol(smiles)
    if mol is None:
        return {
            "valid_rdkit": False,
            "rdkit_mw": np.nan,
            "heavy_atoms": np.nan,
            "fragments": np.nan,
            "elements": "",
            "hetero_atoms": np.nan,
            "rotatable_bonds": np.nan,
            "ring_count": np.nan,
            "aromatic_rings": np.nan,
            "conjugated_bonds": np.nan,
            "frac_csp3": np.nan,
            "tpsa": np.nan,
            "formal_charge": np.nan,
            "scaffold": "",
        }
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    elements = sorted({atom.GetSymbol() for atom in atoms})
    radical_electrons = int(sum(atom.GetNumRadicalElectrons() for atom in atoms))
    return {
        "valid_rdkit": True,
        "rdkit_mw": float(Descriptors.MolWt(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "fragments": int(len(Chem.GetMolFrags(mol))),
        "elements": ",".join(elements),
        "hetero_atoms": int(sum(1 for atom in atoms if atom.GetSymbol() not in {"C", "H"})),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "conjugated_bonds": int(sum(1 for bond in bonds if bond.GetIsConjugated())),
        "frac_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in atoms)),
        "radical_electrons": radical_electrons,
        "scaffold": scaffold_smiles(mol),
    }


def add_features(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> pd.DataFrame:
    features = [molecule_features(smi) for smi in df[smiles_col]]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(features)], axis=1)


def label_flags(row: pd.Series) -> str:
    flags: list[str] = []
    if row.get("fragments", 0) > 1:
        flags.append("salt_or_multifragment")
    if abs(row.get("formal_charge", 0)) > 0:
        flags.append("charged")
    if row.get("radical_electrons", 0) > 0:
        flags.append("radical_or_open_shell")
    if row.get("rotatable_bonds", 0) >= 8:
        flags.append("flexible")
    if row.get("aromatic_rings", 0) >= 4 or row.get("conjugated_bonds", 0) >= 18:
        flags.append("large_conjugated")
    if row.get("gap_true", row.get("gap", np.nan)) < 3.0:
        flags.append("narrow_gap")
    if row.get("gap_true", row.get("gap", np.nan)) > 5.5:
        flags.append("wide_gap")
    elements = str(row.get("elements", ""))
    if "S" in elements:
        flags.append("sulfur")
    if "Cl" in elements:
        flags.append("chlorinated")
    if "F" in elements:
        flags.append("fluorinated")
    return "; ".join(flags) if flags else "ordinary"


def summarize_numeric(series: pd.Series) -> dict:
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def common_eval(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    for target in TARGET_COLS:
        p7 = f"phase7_full_hybrid_{target}"
        p8 = f"replacement300k_full_hybrid_{target}"
        df[f"p7_{target}_abs"] = (df[p7] - df[target]).abs()
        df[f"p8_{target}_abs"] = (df[p8] - df[target]).abs()
        df[f"delta_{target}_abs"] = df[f"p8_{target}_abs"] - df[f"p7_{target}_abs"]
    df["p7_avg_abs"] = df[[f"p7_{t}_abs" for t in TARGET_COLS]].mean(axis=1)
    df["p8_avg_abs"] = df[[f"p8_{t}_abs" for t in TARGET_COLS]].mean(axis=1)
    df["delta_avg_abs"] = df["p8_avg_abs"] - df["p7_avg_abs"]
    df = add_features(df)
    df["flags"] = df.apply(label_flags, axis=1)

    summary = {
        "n": int(len(df)),
        "overall": {
            "p7_avg_mae": float(df["p7_avg_abs"].mean()),
            "p8_avg_mae": float(df["p8_avg_abs"].mean()),
            "delta_avg_mae": float(df["delta_avg_abs"].mean()),
            "p7_gap_mae": float(df["p7_gap_abs"].mean()),
            "p8_gap_mae": float(df["p8_gap_abs"].mean()),
            "delta_gap_mae": float(df["delta_gap_abs"].mean()),
        },
        "by_eval_set": {},
        "improvement_counts": {
            "p8_better_avg_abs": int((df["delta_avg_abs"] < 0).sum()),
            "p8_worse_avg_abs": int((df["delta_avg_abs"] > 0).sum()),
            "p8_better_gap_abs": int((df["delta_gap_abs"] < 0).sum()),
            "p8_worse_gap_abs": int((df["delta_gap_abs"] > 0).sum()),
        },
        "p8_avg_abs_distribution": summarize_numeric(df["p8_avg_abs"]),
        "p8_gap_abs_distribution": summarize_numeric(df["p8_gap_abs"]),
        "remaining_worst_flags": flag_counts(df.nlargest(100, "p8_avg_abs")),
        "largest_improvement_flags": flag_counts(df.nsmallest(100, "delta_avg_abs")),
        "largest_regression_flags": flag_counts(df.nlargest(100, "delta_avg_abs")),
    }
    for eval_set, group in df.groupby("eval_set"):
        summary["by_eval_set"][eval_set] = {
            "n": int(len(group)),
            "p7_avg_mae": float(group["p7_avg_abs"].mean()),
            "p8_avg_mae": float(group["p8_avg_abs"].mean()),
            "delta_avg_mae": float(group["delta_avg_abs"].mean()),
            "p7_gap_mae": float(group["p7_gap_abs"].mean()),
            "p8_gap_mae": float(group["p8_gap_abs"].mean()),
            "delta_gap_mae": float(group["delta_gap_abs"].mean()),
        }
    return df, summary


def pcqm_proxy(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    df["delta_gap_abs"] = df["p8_gap_abs_err"] - df["p7_gap_abs_err"]
    df = add_features(df)
    df["flags"] = df.apply(label_flags, axis=1)
    bins = [0.0, 0.3, 0.4, 0.5, 0.6, 1.01]
    df["p7_sim_bin"] = pd.cut(
        df["p7_train_max_sim"], bins=bins, right=False, include_lowest=True
    ).astype(str)
    summary = {
        "n": int(len(df)),
        "overall": {
            "p7_gap_mae": float(df["p7_gap_abs_err"].mean()),
            "p8_gap_mae": float(df["p8_gap_abs_err"].mean()),
            "delta_gap_mae": float(df["delta_gap_abs"].mean()),
            "p8_better_gap_abs": int((df["delta_gap_abs"] < 0).sum()),
            "p8_worse_gap_abs": int((df["delta_gap_abs"] > 0).sum()),
        },
        "by_p7_sim_bin": {},
        "p8_gap_abs_distribution": summarize_numeric(df["p8_gap_abs_err"]),
        "remaining_worst_flags": flag_counts(df.nlargest(100, "p8_gap_abs_err")),
        "largest_improvement_flags": flag_counts(df.nsmallest(100, "delta_gap_abs")),
        "largest_regression_flags": flag_counts(df.nlargest(100, "delta_gap_abs")),
    }
    for sim_bin, group in df.groupby("p7_sim_bin", observed=True):
        summary["by_p7_sim_bin"][sim_bin] = {
            "n": int(len(group)),
            "p7_gap_mae": float(group["p7_gap_abs_err"].mean()),
            "p8_gap_mae": float(group["p8_gap_abs_err"].mean()),
            "delta_gap_mae": float(group["delta_gap_abs"].mean()),
        }
    return df, summary


def flag_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for flags in df.get("flags", []):
        for flag in str(flags).split("; "):
            if flag:
                counts[flag] += 1
    return dict(counts.most_common())


def md_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.5f}")
            else:
                vals.append(str(val).replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def top_rows_for_md(df: pd.DataFrame, sort_col: str, n: int, kind: str) -> list[dict]:
    rows = []
    for _, row in df.sort_values(sort_col, ascending=False).head(n).iterrows():
        if kind == "common":
            rows.append({
                "eval_set": row["eval_set"],
                "cid": row.get("cid", ""),
                "gap": row["gap"],
                "p8_avg_abs": row["p8_avg_abs"],
                "p8_gap_abs": row["p8_gap_abs"],
                "delta_avg_abs": row["delta_avg_abs"],
                "flags": row["flags"],
                "scaffold": row["scaffold"][:80],
            })
        else:
            rows.append({
                "pcqm_idx": row["pcqm_idx"],
                "gap_true": row["gap_true"],
                "p7_sim": row["p7_train_max_sim"],
                "p8_gap_abs": row["p8_gap_abs_err"],
                "delta_gap_abs": row["delta_gap_abs"],
                "flags": row["flags"],
                "scaffold": row["scaffold"][:80],
            })
    return rows


def write_markdown(
    common_df: pd.DataFrame,
    pcqm_df: pd.DataFrame,
    summary: dict,
    out: Path,
    top_n: int,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 8 v2 Error Mode Analysis")
    lines.append("")
    lines.append("This is a model-selection audit, not a new training run.")
    lines.append("")
    lines.append("## Decision-Relevant Summary")
    lines.append("")
    c = summary["common_eval"]["overall"]
    p = summary["pcqm_proxy"]["overall"]
    lines.append(
        f"- Common eval avg MAE: P7 {c['p7_avg_mae']:.5f} -> "
        f"P8 {c['p8_avg_mae']:.5f} ({c['delta_avg_mae']:+.5f})."
    )
    lines.append(
        f"- Common eval Gap MAE: P7 {c['p7_gap_mae']:.5f} -> "
        f"P8 {c['p8_gap_mae']:.5f} ({c['delta_gap_mae']:+.5f})."
    )
    lines.append(
        f"- PCQM proxy Gap MAE: P7 {p['p7_gap_mae']:.5f} -> "
        f"P8 {p['p8_gap_mae']:.5f} ({p['delta_gap_mae']:+.5f})."
    )
    lines.append(
        "- Main interpretation: replacement300k is a v2 B3LYP-base upgrade for "
        "low-coverage chemistry, while high-similarity chemistry is essentially tied."
    )
    lines.append("")

    lines.append("## Common Eval By Slice")
    lines.append("")
    rows = []
    for eval_set, row in summary["common_eval"]["by_eval_set"].items():
        rows.append({"eval_set": eval_set, **row})
    lines.append(md_table(rows, [
        "eval_set", "n", "p7_avg_mae", "p8_avg_mae", "delta_avg_mae",
        "p7_gap_mae", "p8_gap_mae", "delta_gap_mae",
    ]))
    lines.append("")

    lines.append("## PCQM Proxy By P7 Similarity")
    lines.append("")
    rows = []
    for sim_bin, row in summary["pcqm_proxy"]["by_p7_sim_bin"].items():
        rows.append({"sim_bin": sim_bin, **row})
    lines.append(md_table(rows, ["sim_bin", "n", "p7_gap_mae", "p8_gap_mae", "delta_gap_mae"]))
    lines.append("")

    lines.append("## Remaining Worst Common-Eval Molecules")
    lines.append("")
    lines.append(md_table(top_rows_for_md(common_df, "p8_avg_abs", top_n, "common"), [
        "eval_set", "cid", "gap", "p8_avg_abs", "p8_gap_abs", "delta_avg_abs",
        "flags", "scaffold",
    ]))
    lines.append("")

    lines.append("## Remaining Worst PCQM Proxy Molecules")
    lines.append("")
    lines.append(md_table(top_rows_for_md(pcqm_df, "p8_gap_abs_err", top_n, "pcqm"), [
        "pcqm_idx", "gap_true", "p7_sim", "p8_gap_abs", "delta_gap_abs",
        "flags", "scaffold",
    ]))
    lines.append("")

    lines.append("## Top-100 Flag Counts")
    lines.append("")
    lines.append("Common eval remaining worst:")
    for flag, count in summary["common_eval"]["remaining_worst_flags"].items():
        lines.append(f"- {flag}: {count}")
    lines.append("")
    lines.append("PCQM proxy remaining worst:")
    for flag, count in summary["pcqm_proxy"]["remaining_worst_flags"].items():
        lines.append(f"- {flag}: {count}")
    lines.append("")
    lines.append("Largest PCQM proxy improvements:")
    for flag, count in summary["pcqm_proxy"]["largest_improvement_flags"].items():
        lines.append(f"- {flag}: {count}")
    lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    common_df, common_summary = common_eval(args.common_csv)
    pcqm_df, pcqm_summary = pcqm_proxy(args.pcqm_csv)

    common_worst = common_df.sort_values("p8_avg_abs", ascending=False).head(200)
    pcqm_worst = pcqm_df.sort_values("p8_gap_abs_err", ascending=False).head(200)
    common_worst.to_csv(args.common_worst_out, index=False)
    pcqm_worst.to_csv(args.pcqm_worst_out, index=False)

    summary = {
        "inputs": {
            "common_csv": str(args.common_csv),
            "pcqm_csv": str(args.pcqm_csv),
        },
        "outputs": {
            "common_worst_csv": str(args.common_worst_out),
            "pcqm_worst_csv": str(args.pcqm_worst_out),
            "markdown": str(args.md_out),
        },
        "common_eval": common_summary,
        "pcqm_proxy": pcqm_summary,
    }
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(common_df, pcqm_df, summary, args.md_out, args.top_n)
    print(f"Saved {args.json_out}")
    print(f"Saved {args.md_out}")
    print(f"Saved {args.common_worst_out}")
    print(f"Saved {args.pcqm_worst_out}")


if __name__ == "__main__":
    main()
