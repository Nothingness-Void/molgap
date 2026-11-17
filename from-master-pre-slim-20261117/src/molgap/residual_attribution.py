"""Paired residual attribution for molecular-property model comparisons."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


TARGETS = ("homo", "lumo", "gap")


def molecular_descriptors(smiles_values: Sequence[object]) -> pd.DataFrame:
    rows = []
    for value in smiles_values:
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            rows.append(
                {
                    "heavy_atoms": np.nan,
                    "mw": np.nan,
                    "rings": np.nan,
                    "aromatic_rings": np.nan,
                    "rotatable_bonds": np.nan,
                    "hetero_atoms": np.nan,
                    "formal_charge": np.nan,
                    "fragments": np.nan,
                    "aromatic_atom_fraction": np.nan,
                    "radical_electrons": np.nan,
                    "has_s": np.nan,
                    "has_f": np.nan,
                    "has_cl": np.nan,
                }
            )
            continue
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        heavy_atoms = mol.GetNumHeavyAtoms()
        rows.append(
            {
                "heavy_atoms": heavy_atoms,
                "mw": Descriptors.MolWt(mol),
                "rings": Lipinski.RingCount(mol),
                "aromatic_rings": Lipinski.NumAromaticRings(mol),
                "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
                "hetero_atoms": Lipinski.NumHeteroatoms(mol),
                "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
                "fragments": len(Chem.GetMolFrags(mol)),
                "aromatic_atom_fraction": (
                    sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) / heavy_atoms
                    if heavy_atoms
                    else 0.0
                ),
                "radical_electrons": sum(
                    atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()
                ),
                "has_s": int(16 in atomic_numbers),
                "has_f": int(9 in atomic_numbers),
                "has_cl": int(17 in atomic_numbers),
            }
        )
    return pd.DataFrame(rows)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _metric_block(
    frame: pd.DataFrame,
    baseline: str,
    candidate: str,
    targets: Sequence[str],
) -> dict:
    required = [
        *targets,
        *(f"{baseline}_{target}" for target in targets),
        *(f"{candidate}_{target}" for target in targets),
    ]
    frame = frame.dropna(subset=required)
    if frame.empty:
        raise ValueError(f"No aligned finite rows for {baseline} vs {candidate}")
    baseline_abs, candidate_abs, disagreement = [], [], []
    signed_baseline, signed_candidate = [], []
    target_metrics = {}
    for target in targets:
        truth = frame[target].to_numpy(np.float64)
        base_prediction = frame[f"{baseline}_{target}"].to_numpy(np.float64)
        candidate_prediction = frame[f"{candidate}_{target}"].to_numpy(np.float64)
        base_error = np.abs(base_prediction - truth)
        candidate_error = np.abs(candidate_prediction - truth)
        delta = candidate_error - base_error
        target_metrics[target] = {
            "baseline_mae_eV": float(base_error.mean()),
            "candidate_mae_eV": float(candidate_error.mean()),
            "delta_mae_eV": float(delta.mean()),
            "candidate_win_rate": float((delta < 0).mean()),
            "baseline_signed_bias_eV": float((base_prediction - truth).mean()),
            "candidate_signed_bias_eV": float((candidate_prediction - truth).mean()),
            "absolute_residual_correlation": _correlation(base_error, candidate_error),
            "signed_residual_correlation": _correlation(
                base_prediction - truth, candidate_prediction - truth
            ),
            "disagreement_delta_correlation": _correlation(
                np.abs(candidate_prediction - base_prediction), delta
            ),
            "oracle_mae_eV": float(np.minimum(base_error, candidate_error).mean()),
        }
        baseline_abs.append(base_error)
        candidate_abs.append(candidate_error)
        disagreement.append(np.abs(candidate_prediction - base_prediction))
        signed_baseline.append(base_prediction - truth)
        signed_candidate.append(candidate_prediction - truth)
    baseline_row = np.mean(baseline_abs, axis=0)
    candidate_row = np.mean(candidate_abs, axis=0)
    delta_row = candidate_row - baseline_row
    disagreement_row = np.mean(disagreement, axis=0)
    return {
        "targets": target_metrics,
        "average": {
            "baseline_mae_eV": float(baseline_row.mean()),
            "candidate_mae_eV": float(candidate_row.mean()),
            "delta_mae_eV": float(delta_row.mean()),
            "candidate_win_rate": float((delta_row < 0).mean()),
            "absolute_residual_correlation": _correlation(
                baseline_row, candidate_row
            ),
            "disagreement_delta_correlation": _correlation(
                disagreement_row, delta_row
            ),
            "oracle_mae_eV": float(np.minimum(baseline_row, candidate_row).mean()),
            "top_disagreement_quartile_delta_mae_eV": float(
                delta_row[
                    disagreement_row >= np.quantile(disagreement_row, 0.75)
                ].mean()
            ),
        },
    }


def _fixed_strata(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "true_gap": pd.cut(
            frame["gap"], [-np.inf, 2, 4, 6, 8, np.inf], right=False
        ).astype(str),
        "heavy_atoms": pd.cut(
            frame["heavy_atoms"], [-np.inf, 15, 25, 35, 50, np.inf], right=False
        ).astype(str),
        "mw": pd.cut(
            frame["mw"], [-np.inf, 200, 350, 500, 700, np.inf], right=False
        ).astype(str),
        "aromatic_rings": pd.cut(
            frame["aromatic_rings"], [-np.inf, 1, 2, 4, np.inf], right=False
        ).astype(str),
        "rotatable_bonds": pd.cut(
            frame["rotatable_bonds"], [-np.inf, 2, 5, 10, np.inf], right=False
        ).astype(str),
        "fragments": frame["fragments"].fillna(-1).astype(int).astype(str),
    }


def analyze_comparison(
    frame: pd.DataFrame,
    *,
    baseline: str,
    candidates: Sequence[str],
    targets: Sequence[str] = TARGETS,
) -> tuple[dict, pd.DataFrame]:
    """Attribute paired candidate-minus-baseline errors across molecular strata."""
    required = {"smiles", *targets}
    for model in (baseline, *candidates):
        required.update(f"{model}_{target}" for target in targets)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}")
    enriched = pd.concat(
        [frame.reset_index(drop=True), molecular_descriptors(frame.smiles)],
        axis=1,
    )
    scopes = {"all": np.ones(len(enriched), dtype=bool)}
    if "eval_set" in enriched:
        scopes.update(
            {
                str(scope): enriched.eval_set.eq(scope).to_numpy()
                for scope in enriched.eval_set.dropna().unique()
            }
        )
    report = {
        "rows": len(enriched),
        "baseline": baseline,
        "candidates": {},
    }
    strata_rows = []
    strata = _fixed_strata(enriched)
    for candidate in candidates:
        candidate_report = {"scopes": {}}
        for scope, mask in scopes.items():
            candidate_report["scopes"][scope] = _metric_block(
                enriched.loc[mask], baseline, candidate, targets
            )
        for descriptor, labels in strata.items():
            for label in sorted(labels.unique()):
                mask = labels.eq(label).to_numpy()
                if mask.sum() < 20:
                    continue
                metrics = _metric_block(
                    enriched.loc[mask], baseline, candidate, targets
                )
                strata_rows.append(
                    {
                        "candidate": candidate,
                        "descriptor": descriptor,
                        "stratum": label,
                        "n": int(mask.sum()),
                        "average_delta_mae_eV": metrics["average"]["delta_mae_eV"],
                        "gap_delta_mae_eV": metrics["targets"]["gap"]["delta_mae_eV"],
                        "average_win_rate": metrics["average"]["candidate_win_rate"],
                    }
                )
        report["candidates"][candidate] = candidate_report
    return report, pd.DataFrame(strata_rows)
