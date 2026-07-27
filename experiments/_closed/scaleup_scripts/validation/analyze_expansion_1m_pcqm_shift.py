"""Explain where the 1M continuation changes PCQM4Mv2 public-valid error.

This is a read-only paired analysis of the completed local 5K PCQM4Mv2
public-valid check.  It intentionally uses only the official Gap label because
PCQM4Mv2 does not provide HOMO/LUMO targets in this split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


CONTINUOUS = ("heavy_atoms", "mol_wt", "rings", "aromatic_rings", "rotatable_bonds")
ELEMENTS = ("N", "O", "S", "F", "Cl", "Br", "P", "Si")


def descriptors(smiles: str) -> dict[str, float | int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES in completed prediction file: {smiles!r}")
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    result: dict[str, float | int] = {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mol_wt": round(float(Descriptors.MolWt(mol)), 5),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
    }
    for element in ELEMENTS:
        result[f"has_{element}"] = int(element in atoms)
    return result


def bootstrap_ci(delta: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    if len(delta) < 2:
        return float("nan"), float("nan")
    samples = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def summarize(group: pd.DataFrame, label: str, rng: np.random.Generator, draws: int) -> dict:
    delta = group["abs_error_delta_gap"].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_ci(delta, rng, draws)
    return {
        "bucket": label,
        "n": int(len(group)),
        "routed_v4_gap_mae_eV": float(group["routed_v4_abs_error_gap"].mean()),
        "candidate_1m_gap_mae_eV": float(group["candidate_1m_abs_error_gap"].mean()),
        "candidate_minus_v4_gap_mae_eV": float(delta.mean()),
        "ci95_low_eV": ci_low,
        "ci95_high_eV": ci_high,
        "candidate_win_rate": float((delta < 0.0).mean()),
    }


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# 1M PCQM4Mv2 Shift Analysis",
        "",
        "Read-only structural stratification of the completed 4,981-molecule paired",
        "PCQM4Mv2 public-valid check. Positive delta means the 1M candidate has",
        "higher Gap absolute error than routed-v4. This is diagnostic evidence, not",
        "a replacement for the official OGB hidden evaluation.",
        "",
        "## Overall",
        "",
        "| n | routed-v4 Gap MAE | 1M Gap MAE | 1M minus v4 | 95% CI | 1M win rate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    overall = summary["overall"]
    lines.append(
        f"| {overall['n']} | {overall['routed_v4_gap_mae_eV']:.6f} | "
        f"{overall['candidate_1m_gap_mae_eV']:.6f} | "
        f"{overall['candidate_minus_v4_gap_mae_eV']:+.6f} | "
        f"[{overall['ci95_low_eV']:+.6f}, {overall['ci95_high_eV']:+.6f}] | "
        f"{overall['candidate_win_rate']:.3f} |"
    )
    for section, rows in summary["strata"].items():
        lines.extend(["", f"## {section}", "", "| bucket | n | v4 MAE | 1M MAE | delta | 95% CI | 1M win rate |", "|---|---:|---:|---:|---:|---:|---:|"])
        for row in rows:
            lines.append(
                f"| {row['bucket']} | {row['n']} | {row['routed_v4_gap_mae_eV']:.6f} | "
                f"{row['candidate_1m_gap_mae_eV']:.6f} | {row['candidate_minus_v4_gap_mae_eV']:+.6f} | "
                f"[{row['ci95_low_eV']:+.6f}, {row['ci95_high_eV']:+.6f}] | {row['candidate_win_rate']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    required = {"smiles", "gap", "routed_v4_gap", "candidate_1m_gap", "abs_error_delta_gap"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")
    descriptor_frame = pd.DataFrame([descriptors(smiles) for smiles in frame["smiles"]])
    frame = pd.concat([frame.reset_index(drop=True), descriptor_frame], axis=1)
    frame["routed_v4_abs_error_gap"] = (frame["routed_v4_gap"] - frame["gap"]).abs()
    frame["candidate_1m_abs_error_gap"] = (frame["candidate_1m_gap"] - frame["gap"]).abs()
    calculated_delta = frame["candidate_1m_abs_error_gap"] - frame["routed_v4_abs_error_gap"]
    if not np.allclose(frame["abs_error_delta_gap"], calculated_delta, atol=1e-6):
        raise ValueError("Stored paired error deltas do not match predictions")

    rng = np.random.default_rng(20260718)
    strata: dict[str, list[dict]] = {}
    for column in CONTINUOUS:
        labels = pd.qcut(frame[column], q=4, duplicates="drop")
        rows = []
        for interval, group in frame.groupby(labels, observed=True):
            rows.append(summarize(group, f"{column} {interval}", rng, args.bootstrap_draws))
        strata[f"{column} quartiles"] = rows
    element_rows = []
    for element in ELEMENTS:
        column = f"has_{element}"
        for value, label in ((1, f"contains {element}"), (0, f"no {element}")):
            group = frame.loc[frame[column] == value]
            if len(group) >= 20:
                element_rows.append(summarize(group, label, rng, args.bootstrap_draws))
    strata["Element presence"] = element_rows

    summary = {
        "source": str(args.predictions),
        "n": int(len(frame)),
        "target": "PCQM4Mv2 homolumogap only",
        "bootstrap_draws": args.bootstrap_draws,
        "overall": summarize(frame, "all", rng, args.bootstrap_draws),
        "strata": strata,
    }
    for path in (args.out_csv, args.out_json, args.out_report):
        path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.out_report, summary)
    print(json.dumps(summary["overall"], indent=2))
    print(f"Rows -> {args.out_csv}")
    print(f"Summary -> {args.out_json}")
    print(f"Report -> {args.out_report}")


if __name__ == "__main__":
    main()
