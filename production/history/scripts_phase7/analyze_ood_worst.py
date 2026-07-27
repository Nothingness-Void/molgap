"""
Analyze worst OOD-1000 molecules for the Phase 7 Hybrid / MoE comparison.

Input:
  results/phase7/moe_experiment/ood_moe_e4_predictions.csv

Outputs:
  results/phase7/moe_experiment/ood_worst_hybrid.csv
  results/phase7/moe_experiment/ood_worst_hybrid_top20.md
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from molgap.constants import RESULTS_DIR, TARGET_COLS

PRED_CSV = RESULTS_DIR / "phase7" / "moe_experiment" / "ood_moe_e4_predictions.csv"
OUT_CSV = RESULTS_DIR / "phase7" / "moe_experiment" / "ood_worst_hybrid.csv"
OUT_MD = RESULTS_DIR / "phase7" / "moe_experiment" / "ood_worst_hybrid_top20.md"
MODELS = ["2d", "3d", "hybrid", "moe"]


def safe_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.RemoveStereochemistry(mol)
    return mol


def scaffold_smiles(mol) -> str:
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except RuntimeError:
        return ""


def molecule_features(smiles: str) -> dict:
    mol = safe_mol(smiles)
    if mol is None:
        return {
            "valid_rdkit": False,
            "fragments": smiles.count(".") + 1,
            "scaffold": "",
        }
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    elements = sorted({a.GetSymbol() for a in atoms})
    hetero_atoms = sum(1 for a in atoms if a.GetSymbol() not in {"C", "H"})
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    rotors = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    fragments = len(Chem.GetMolFrags(mol))
    conjugated_bonds = sum(1 for b in bonds if b.GetIsConjugated())
    formal_charge = sum(a.GetFormalCharge() for a in atoms)
    return {
        "valid_rdkit": True,
        "rdkit_mw": Descriptors.MolWt(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "fragments": fragments,
        "elements": "".join(elements),
        "hetero_atoms": hetero_atoms,
        "rotatable_bonds": rotors,
        "ring_count": ring_count,
        "aromatic_rings": aromatic_rings,
        "conjugated_bonds": conjugated_bonds,
        "frac_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "formal_charge": formal_charge,
        "scaffold": scaffold_smiles(mol),
    }


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    for model in MODELS:
        abs_cols = []
        signed_cols = []
        for target in TARGET_COLS:
            signed = f"{target}_{model}_err"
            absolute = f"{target}_{model}_abs"
            df[signed] = df[f"{target}_{model}"] - df[target]
            df[absolute] = df[signed].abs()
            signed_cols.append(signed)
            abs_cols.append(absolute)
        df[f"{model}_avg_abs"] = df[abs_cols].mean(axis=1)
        df[f"{model}_max_abs"] = df[abs_cols].max(axis=1)
        df[f"{model}_gap_consistency_abs"] = (
            df[f"gap_{model}"] - (df[f"lumo_{model}"] - df[f"homo_{model}"])
        ).abs()
    avg_cols = [f"{m}_avg_abs" for m in MODELS]
    df["best_model"] = df[avg_cols].idxmin(axis=1).str.replace("_avg_abs", "", regex=False)
    df["best_avg_abs"] = df[avg_cols].min(axis=1)
    df["hybrid_minus_best_abs"] = df["hybrid_avg_abs"] - df["best_avg_abs"]
    df["hybrid_minus_2d_abs"] = df["hybrid_avg_abs"] - df["2d_avg_abs"]
    df["hybrid_minus_3d_abs"] = df["hybrid_avg_abs"] - df["3d_avg_abs"]
    return df


def label_row(row: pd.Series) -> str:
    flags = []
    if row["fragments"] > 1:
        flags.append("multi_fragment/salt")
    if row["gap"] < 3.0:
        flags.append("narrow_gap")
    elif row["gap"] > 5.5:
        flags.append("wide_gap")
    if row["rotatable_bonds"] >= 8:
        flags.append("flexible")
    if row["aromatic_rings"] >= 3 or row["conjugated_bonds"] >= 16:
        flags.append("large_conjugated")
    if "Cl" in str(row["formula"]):
        flags.append("chlorinated")
    if row["hybrid_minus_best_abs"] > 0.03:
        flags.append("fusion_not_best")
    if row["best_avg_abs"] > 0.20:
        flags.append("all_models_hard")
    return "; ".join(flags) if flags else "ordinary"


def markdown_table(df: pd.DataFrame, columns: list[str], n: int) -> str:
    view = df.head(n)[columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.4f}")
    headers = list(view.columns)
    rows = [[str(value) for value in row] for row in view.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def small_markdown_table(df: pd.DataFrame) -> str:
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.4f}")
    view.insert(0, "quantile", [str(i) for i in view.index])
    return markdown_table(view, list(view.columns), len(view))


def summarize(df: pd.DataFrame, top: pd.DataFrame, top_n: int) -> str:
    lines = []
    lines.append("# OOD Worst Molecule Analysis")
    lines.append("")
    lines.append(f"Rows analyzed: {len(df)}")
    lines.append(f"Primary sort: Hybrid average absolute error over {TARGET_COLS}")
    lines.append("")

    lines.append("## Top Worst Hybrid Molecules")
    lines.append("")
    columns = [
        "cid", "formula", "mw", "gap", "hybrid_avg_abs", "homo_hybrid_abs",
        "lumo_hybrid_abs", "gap_hybrid_abs", "best_model", "best_avg_abs",
        "rotatable_bonds", "aromatic_rings", "fragments", "flags", "smiles",
    ]
    lines.append(markdown_table(top, columns, top_n))
    lines.append("")

    lines.append("## Pattern Counts In Top Set")
    lines.append("")
    flag_counts = Counter()
    for flags in top["flags"]:
        for flag in str(flags).split("; "):
            flag_counts[flag] += 1
    for flag, count in flag_counts.most_common():
        lines.append(f"- {flag}: {count}/{len(top)}")
    lines.append("")

    lines.append("## Best Model Among Top Worst")
    lines.append("")
    for model, count in top["best_model"].value_counts().items():
        lines.append(f"- {model}: {count}/{len(top)}")
    lines.append("")

    lines.append("## Overall Model-Winner Counts")
    lines.append("")
    for model, count in df["best_model"].value_counts().items():
        lines.append(f"- {model}: {count}/{len(df)}")
    lines.append("")
    lines.append(
        f"Hybrid loses to the best available model by >0.03 eV on "
        f"{int((df['hybrid_minus_best_abs'] > 0.03).sum())}/{len(df)} molecules."
    )
    lines.append(
        f"Hybrid loses by >0.10 eV on "
        f"{int((df['hybrid_minus_best_abs'] > 0.10).sum())}/{len(df)} molecules."
    )
    lines.append("")

    lines.append("## Overall Flag Aggregates")
    lines.append("")
    flag_rows = []
    flags = [
        "all_models_hard", "fusion_not_best", "flexible", "large_conjugated",
        "wide_gap", "narrow_gap", "multi_fragment/salt", "chlorinated",
    ]
    for flag in flags:
        mask = df["flags"].str.contains(flag, regex=False)
        if mask.any():
            flag_rows.append({
                "flag": flag,
                "n": int(mask.sum()),
                "mean_hybrid_avg_abs": df.loc[mask, "hybrid_avg_abs"].mean(),
                "mean_gap_abs": df.loc[mask, "gap_hybrid_abs"].mean(),
            })
    flag_df = pd.DataFrame(flag_rows).sort_values("mean_hybrid_avg_abs", ascending=False)
    lines.append(markdown_table(flag_df, list(flag_df.columns), len(flag_df)))
    lines.append("")

    lines.append("## Top Gap Errors")
    lines.append("")
    gap_top = df.sort_values("gap_hybrid_abs", ascending=False).head(10)
    gap_cols = [
        "cid", "formula", "gap", "gap_hybrid", "gap_hybrid_abs",
        "hybrid_avg_abs", "best_model", "best_avg_abs", "flags", "smiles",
    ]
    lines.append(markdown_table(gap_top, gap_cols, len(gap_top)))
    lines.append("")

    lines.append("## Biggest Fusion Misses")
    lines.append("")
    miss_top = df.sort_values("hybrid_minus_best_abs", ascending=False).head(10)
    miss_cols = [
        "cid", "formula", "hybrid_avg_abs", "best_model", "best_avg_abs",
        "hybrid_minus_best_abs", "flags", "smiles",
    ]
    lines.append(markdown_table(miss_top, miss_cols, len(miss_top)))
    lines.append("")

    lines.append("## Error Quantiles")
    lines.append("")
    q = df[["hybrid_avg_abs", "gap_hybrid_abs", "hybrid_minus_best_abs"]].quantile(
        [0.5, 0.75, 0.9, 0.95, 0.99]
    )
    lines.append(small_markdown_table(q))
    lines.append("")

    lines.append("## Scaffold Repeats In Top Set")
    lines.append("")
    scaffold_counts = top["scaffold"].replace("", "(none)").value_counts().head(10)
    for scaffold, count in scaffold_counts.items():
        lines.append(f"- `{scaffold}`: {count}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", type=str, default=str(PRED_CSV))
    parser.add_argument("--out-csv", type=str, default=str(OUT_CSV))
    parser.add_argument("--out-md", type=str, default=str(OUT_MD))
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.warning")
    df = pd.read_csv(args.pred_csv)
    df = add_error_columns(df)
    feat = pd.DataFrame([molecule_features(s) for s in df["smiles"]])
    df = pd.concat([df, feat], axis=1)
    df["flags"] = df.apply(label_row, axis=1)
    df = df.sort_values("hybrid_avg_abs", ascending=False).reset_index(drop=True)
    top = df.head(args.top_n)

    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    md = summarize(df, top, args.top_n)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print(f"\nSaved CSV: {args.out_csv}")
    print(f"Saved report: {args.out_md}")


if __name__ == "__main__":
    main()
