"""G0 audit for ring/conjugation features against stored routed-v4 residuals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from molgap.archive.phase8_r06_structural_gps_adapter.structural_features import structural_summary_from_mol
from molgap.utils import safe_mol


PHASE8 = Path("results/phase8")
ARCHIVE_DIR = PHASE8 / "archive" / "archive-r06-structural-gps-adapter"
DEFAULT_INPUT = PHASE8 / "gps_arch_dualgps_common_eval_predictions.csv"
DEFAULT_OUT_DIR = ARCHIVE_DIR
FEATURE_COLUMNS = (
    "has_ring",
    "atom_in_ring_fraction",
    "smallest_ring_size",
    "max_ring_membership_count",
    "has_fused_ring_atom",
    "fused_ring_atom_fraction",
    "ring_bond_fraction",
    "has_conjugated_bond",
    "conjugated_bond_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-prefix", default="expansion500k_full_hybrid")
    parser.add_argument("--dual-prefix", default="expansion500k_dualgps_hybrid")
    parser.add_argument("--route-threshold", type=float, default=4.0)
    return parser.parse_args()


def routed_gap_prediction(frame: pd.DataFrame, base_prefix: str, dual_prefix: str, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    base_col = f"{base_prefix}_gap"
    dual_col = f"{dual_prefix}_gap"
    missing = [col for col in ("gap", base_col, dual_col) if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    base = frame[base_col].to_numpy(dtype=np.float64)
    dual = frame[dual_col].to_numpy(dtype=np.float64)
    route = base < threshold
    return np.where(route, dual, base), route


def add_structural_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    smiles_col = "canonical_smiles" if "canonical_smiles" in frame.columns else "smiles"
    if smiles_col not in frame.columns:
        raise KeyError("Need canonical_smiles or smiles column")

    records: list[dict[str, float]] = []
    invalid = 0
    for smiles in frame[smiles_col].fillna("").astype(str):
        mol = safe_mol(smiles)
        if mol is None:
            invalid += 1
            records.append({name: np.nan for name in FEATURE_COLUMNS})
            continue
        records.append(structural_summary_from_mol(mol))

    features = pd.DataFrame(records, index=frame.index)
    result = pd.concat([frame.copy(), features], axis=1)
    finite = np.isfinite(features.to_numpy(dtype=np.float64)).all(axis=1)
    return result, {
        "input_rows": int(len(frame)),
        "valid_smiles": int(len(frame) - invalid),
        "invalid_smiles": int(invalid),
        "finite_feature_rows": int(finite.sum()),
    }


def decile_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for decile, subset in frame.groupby("gap_error_decile", observed=True):
        row: dict[str, float | int] = {
            "error_decile": int(decile),
            "n": int(len(subset)),
            "gap_mae": float(subset["routed_v4_gap_abs_error"].mean()),
        }
        for name in FEATURE_COLUMNS:
            row[name] = float(subset[name].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("error_decile").reset_index(drop=True)


def enrichment_table(frame: pd.DataFrame) -> pd.DataFrame:
    overall = frame.loc[:, FEATURE_COLUMNS].mean(numeric_only=True)
    top = frame.loc[frame["gap_error_decile"] == 10, FEATURE_COLUMNS].mean(numeric_only=True)
    rows = []
    for name in FEATURE_COLUMNS:
        rows.append({
            "feature": name,
            "overall_mean": float(overall[name]),
            "top_decile_mean": float(top[name]),
            "top_minus_overall": float(top[name] - overall[name]),
            "top_over_overall": float(top[name] / overall[name]) if overall[name] > 0 else None,
        })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    headers = [str(col) for col in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            formatted.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        rows.append(formatted)
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def build_report(args: argparse.Namespace, coverage: dict[str, int], error: dict[str, float], deciles: pd.DataFrame, enrichment: pd.DataFrame) -> str:
    enriched = enrichment.loc[enrichment["top_minus_overall"] > 0, "feature"].tolist()
    return f"""# Phase 8 Structural GPS Adapter -- G0 Coverage Audit

Input: `{args.input}`

This is an analysis-only audit. The routed-v4 Gap prediction is reconstructed
from the stored v3 and dual-GPS predictions using its fixed `{args.route_threshold:g}` eV
route rule; no checkpoint or fitted model was used.

## Data contract

- rows: {coverage['input_rows']}
- valid SMILES: {coverage['valid_smiles']}
- invalid SMILES: {coverage['invalid_smiles']}
- finite structural-feature rows: {coverage['finite_feature_rows']}
- routed-v4 Gap MAE: {error['mae']:.6f} eV
- routed rows: {error['route_n']} / {coverage['input_rows']}

## Feature prevalence by routed-v4 Gap-error decile

Decile 10 is the highest absolute Gap-error group.

{markdown_table(deciles)}

## Highest-decile enrichment

Positive `top_minus_overall` means a structural feature is more prevalent in
the worst routed-v4 Gap decile than in the evaluated population.

{markdown_table(enrichment)}

## G0 result

Features with positive highest-decile enrichment: {", ".join(enriched) if enriched else "none"}.

This result alone does not authorize fitting. G1 remains conditional on a
chemically interpretable enrichment rather than a feature-frequency artifact,
as specified in `pre_registration.md`.
"""


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    frame = pd.read_csv(args.input)
    frame, coverage = add_structural_features(frame)
    pred, route = routed_gap_prediction(frame, args.base_prefix, args.dual_prefix, args.route_threshold)
    frame["routed_v4_gap_prediction"] = pred
    frame["routed_v4_route"] = route
    frame["routed_v4_gap_abs_error"] = np.abs(frame["gap"].to_numpy(dtype=np.float64) - pred)
    frame["gap_error_decile"] = pd.qcut(
        frame["routed_v4_gap_abs_error"].rank(method="first"), q=10, labels=False
    ).astype(int) + 1

    deciles = decile_table(frame)
    enrichment = enrichment_table(frame)
    error = {
        "mae": float(frame["routed_v4_gap_abs_error"].mean()),
        "route_n": int(route.sum()),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "g0_rows.csv", index=False, encoding="utf-8")
    metrics = {
        "input": str(args.input),
        "base_prefix": args.base_prefix,
        "dual_prefix": args.dual_prefix,
        "route_threshold_eV": args.route_threshold,
        "coverage": coverage,
        "routed_v4_gap": error,
        "deciles": deciles.to_dict(orient="records"),
        "top_decile_enrichment": enrichment.to_dict(orient="records"),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.out_dir / "report.md").write_text(
        build_report(args, coverage, error, deciles, enrichment), encoding="utf-8"
    )
    print(f"Rows: {coverage['input_rows']} valid={coverage['valid_smiles']} finite={coverage['finite_feature_rows']}")
    print(f"Routed-v4 Gap MAE: {error['mae']:.6f} eV; routed={error['route_n']}")
    print(f"Metrics -> {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
