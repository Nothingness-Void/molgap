"""Turn fixed external residuals into interpretable data-acquisition targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from molgap.router import router_descriptor_row
from molgap.utils import scaffold_split_key


TARGETS = ("homo", "lumo", "gap")
AMIDE = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")


def molecule_features(smiles: str) -> dict[str, object]:
    row: dict[str, object] = router_descriptor_row(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        row.update({"scaffold": "INVALID", "amide_bonds": np.nan, "macrocycle": np.nan})
        return row
    row["scaffold"] = scaffold_split_key(smiles)
    row["amide_bonds"] = float(len(mol.GetSubstructMatches(AMIDE)))
    row["macrocycle"] = float(any(len(ring) >= 8 for ring in mol.GetRingInfo().AtomRings()))
    row["bridgeheads"] = float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    return row


def summarize(group: pd.DataFrame, family: str, bucket: str) -> dict[str, object]:
    base = group.base_gap_abs_error.to_numpy()
    delta = group.candidate_minus_base_gap_error.to_numpy()
    return {
        "family": family,
        "bucket": bucket,
        "n": int(len(group)),
        "base_gap_mae_eV": float(base.mean()),
        "base_gap_p90_eV": float(np.quantile(base, 0.9)),
        "candidate_minus_base_gap_eV": float(delta.mean()),
        "candidate_win_rate": float((delta < 0).mean()),
        "share_of_base_gap_error": float(base.sum()),
    }


def add_bucket_rows(rows: list[dict], frame: pd.DataFrame, family: str, labels: pd.Series) -> None:
    for bucket, group in frame.groupby(labels, observed=True):
        if len(group) >= 20:
            rows.append(summarize(group, family, str(bucket)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    required = {"eval_set", "cid", "smiles", *TARGETS}
    for target in TARGETS:
        required.update({f"original_1m_{target}", f"additive_1p5m_{target}"})
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    descriptors = pd.DataFrame([molecule_features(smiles) for smiles in frame.smiles])
    frame = pd.concat([frame.reset_index(drop=True), descriptors], axis=1)
    for target in TARGETS:
        frame[f"base_{target}_abs_error"] = (frame[f"original_1m_{target}"] - frame[target]).abs()
        frame[f"candidate_{target}_abs_error"] = (frame[f"additive_1p5m_{target}"] - frame[target]).abs()
        frame[f"candidate_minus_base_{target}_error"] = (
            frame[f"candidate_{target}_abs_error"] - frame[f"base_{target}_abs_error"]
        )

    rows: list[dict] = []
    add_bucket_rows(rows, frame, "eval_set", frame.eval_set)
    add_bucket_rows(rows, frame, "true_gap", pd.cut(frame.gap, [-np.inf, 2.5, 3.2, 4.0, 5.5, np.inf]))
    add_bucket_rows(rows, frame, "mw", pd.cut(frame.mw, [-np.inf, 300, 500, 700, np.inf]))
    add_bucket_rows(rows, frame, "aromatic_rings", pd.cut(frame.aromatic_rings, [-np.inf, 0, 2, 4, np.inf]))
    add_bucket_rows(rows, frame, "rotatable_bonds", pd.cut(frame.rotatable_bonds, [-np.inf, 3, 7, np.inf]))
    add_bucket_rows(rows, frame, "fraction_csp3", pd.cut(frame.fraction_csp3, [-np.inf, 0.1, 0.4, 0.7, np.inf]))
    add_bucket_rows(rows, frame, "amide_bonds", pd.cut(frame.amide_bonds, [-np.inf, 0, 2, 5, np.inf]))
    for column in ("n_N", "n_O", "n_S", "n_F", "n_Cl", "macrocycle"):
        labels = frame[column].gt(0).map({True: f"has_{column}", False: f"no_{column}"})
        add_bucket_rows(rows, frame, column, labels)

    strata = pd.DataFrame(rows)
    strata["error_contribution_fraction"] = (
        strata.share_of_base_gap_error / frame.base_gap_abs_error.sum()
    )
    strata = strata.sort_values(
        ["base_gap_mae_eV", "error_contribution_fraction"], ascending=False
    ).reset_index(drop=True)

    scaffold = frame.groupby("scaffold", dropna=False).agg(
        n=("cid", "size"),
        base_gap_mae_eV=("base_gap_abs_error", "mean"),
        candidate_minus_base_gap_eV=("candidate_minus_base_gap_error", "mean"),
        example_smiles=("smiles", "first"),
    )
    scaffold = scaffold.loc[scaffold.n >= 2].sort_values("base_gap_mae_eV", ascending=False).reset_index()
    worst = frame.sort_values("base_gap_abs_error", ascending=False).head(200).copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "residual_descriptors.csv", index=False)
    strata.to_csv(args.out_dir / "residual_strata.csv", index=False)
    scaffold.to_csv(args.out_dir / "worst_scaffolds.csv", index=False)
    worst.to_csv(args.out_dir / "worst_200_molecules.csv", index=False)

    top = strata.loc[strata.n >= 40].head(20)
    summary = {
        "source": str(args.predictions),
        "n": int(len(frame)),
        "development_set_warning": (
            "These external rows are now used for acquisition design and cannot remain a sealed acceptance set."
        ),
        "base_gap_mae_eV": float(frame.base_gap_abs_error.mean()),
        "candidate_gap_mae_eV": float(frame.candidate_gap_abs_error.mean()),
        "top_strata_min_n40": top.to_dict("records"),
    }
    (args.out_dir / "residual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Original 1M Residual Acquisition Analysis",
        "",
        "This analysis uses the fixed common/OOD/P8-hard predictions to design",
        "future acquisition. These 1,977 rows are therefore development evidence",
        "from this point onward; promotion requires a new scaffold-disjoint sealed set.",
        "",
        "## Highest-error structural strata (minimum 40 molecules)",
        "",
        "| family | bucket | n | original 1M Gap MAE | 1.5M minus 1M | candidate win rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in top.itertuples():
        report.append(
            f"| {row.family} | {row.bucket} | {row.n} | {row.base_gap_mae_eV:.5f} | "
            f"{row.candidate_minus_base_gap_eV:+.5f} | {row.candidate_win_rate:.3f} |"
        )
    report.extend([
        "",
        "## Outputs",
        "",
        "- `worst_200_molecules.csv`: molecule-level acquisition seeds.",
        "- `worst_scaffolds.csv`: repeated high-error scaffolds.",
        "- `residual_strata.csv`: all interpretable descriptor strata.",
        "- `residual_descriptors.csv`: complete descriptor-enriched residual table.",
    ])
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
