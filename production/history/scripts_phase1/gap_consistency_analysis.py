"""
11_gap_consistency_analysis.py — compare gap prediction strategies.

The model directly predicts HOMO, LUMO, and gap, but physically gap = LUMO - HOMO.
This script checks whether direct gap prediction, orbital-derived gap, or a blend
is best on existing random/scaffold LightGBM prediction files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from molgap.utils import RESULTS_DIR, ensure_dirs


DEFAULT_RANDOM = RESULTS_DIR / "phase1" / "baseline" / "test_predictions_lightgbm.csv"
DEFAULT_SCAFFOLD = RESULTS_DIR / "phase1" / "scaffold" / "test_predictions_lightgbm_scaffold.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "phase1" / "gap_consistency"


def gap_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_true - y_pred
    return {
        "gap_mae": float(mean_absolute_error(y_true, y_pred)),
        "gap_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "gap_r2": float(r2_score(y_true, y_pred)),
        "gap_mean_signed_error": float(np.mean(residual)),
    }


def analyze_file(pred_path: Path, split_name: str, output_dir: Path, alphas: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(pred_path)
    required = ["homo_pred", "lumo_pred", "gap_true", "gap_pred"]
    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise ValueError(f"{pred_path} missing columns: {missing}")

    y_true = pred["gap_true"].values
    direct = pred["gap_pred"].values
    orbital = pred["lumo_pred"].values - pred["homo_pred"].values

    rows = []
    for strategy, values, alpha in [
        ("direct_gap", direct, np.nan),
        ("orbital_gap", orbital, 0.0),
    ]:
        row = {"split_type": split_name, "strategy": strategy, "alpha": alpha}
        row.update(gap_metrics(y_true, values))
        rows.append(row)

    for alpha in alphas:
        blended = alpha * direct + (1.0 - alpha) * orbital
        row = {"split_type": split_name, "strategy": "blend_gap", "alpha": float(alpha)}
        row.update(gap_metrics(y_true, blended))
        rows.append(row)

    comparison = pd.DataFrame(rows)
    best = comparison.sort_values("gap_mae").iloc[0]
    best_alpha = 1.0 if best["strategy"] == "direct_gap" else float(best["alpha"])
    best_blend = best_alpha * direct + (1.0 - best_alpha) * orbital

    enriched = pred.copy()
    enriched["gap_orbital_pred"] = orbital
    enriched["gap_orbital_abs_error"] = np.abs(y_true - orbital)
    enriched["gap_blend_best_alpha"] = best_alpha
    enriched["gap_blend_best_pred"] = best_blend
    enriched["gap_blend_best_abs_error"] = np.abs(y_true - best_blend)
    enriched.to_csv(output_dir / f"{split_name}_gap_predictions_with_strategies.csv", index=False, encoding="utf-8")
    return comparison, enriched


def plot_strategy(comparison: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    simple = comparison[comparison["strategy"].isin(["direct_gap", "orbital_gap"])].copy()
    blend_best = comparison[comparison["strategy"] == "blend_gap"].sort_values("gap_mae").groupby("split_type").head(1)
    blend_best = blend_best.copy()
    blend_best["strategy"] = "best_blend_gap"
    plot_df = pd.concat([simple, blend_best], ignore_index=True)

    for metric, ylabel, fname in [
        ("gap_mae", "Gap MAE (eV)", "gap_strategy_mae.png"),
        ("gap_r2", "Gap R²", "gap_strategy_r2.png"),
    ]:
        plt.figure(figsize=(7, 4.5))
        sns.barplot(data=plot_df, x="split_type", y=metric, hue="strategy")
        plt.xlabel("Split type")
        plt.ylabel(ylabel)
        plt.title(ylabel + " by strategy")
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=300)
        plt.close()


def run_gap_consistency(random_path: Path, scaffold_path: Path, output_dir: Path) -> pd.DataFrame:
    ensure_dirs(output_dir)
    alphas = np.round(np.arange(0.0, 1.0001, 0.1), 2)
    all_rows = []
    for split_name, path in [("random", random_path), ("scaffold", scaffold_path)]:
        if not path.exists():
            print(f"Skipping missing predictions: {path}")
            continue
        comp, _ = analyze_file(path, split_name, output_dir, alphas)
        all_rows.append(comp)

    comparison = pd.concat(all_rows, ignore_index=True)
    comparison.to_csv(output_dir / "gap_strategy_comparison.csv", index=False, encoding="utf-8")
    best = comparison.sort_values("gap_mae").groupby("split_type").head(1)
    best.to_csv(output_dir / "gap_strategy_best_by_split.csv", index=False, encoding="utf-8")
    plot_strategy(comparison, output_dir)

    print("\n=== GAP CONSISTENCY SUMMARY ===")
    print(best[["split_type", "strategy", "alpha", "gap_mae", "gap_rmse", "gap_r2"]])
    print(f"outputs: {output_dir}")
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze MolGap gap consistency strategies")
    parser.add_argument("--random", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--scaffold", type=Path, default=DEFAULT_SCAFFOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    run_gap_consistency(args.random, args.scaffold, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
