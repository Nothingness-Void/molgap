"""
05_analyze_results.py — interpretability and error plots for the MolGap baseline.

This script turns the trained LightGBM baseline into report-ready analysis:
  - per-target metric summary
  - parity plots
  - residual distributions
  - top-error molecule table
  - LightGBM feature-importance tables and plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import MODELS_DIR, RESULTS_DIR, TARGET_COLS, ensure_dirs, load_model_bundle


DEFAULT_MODEL = MODELS_DIR / "baseline_lightgbm.joblib"
DEFAULT_PREDICTIONS = RESULTS_DIR / "test_predictions_lightgbm.csv"
DEFAULT_COMPARISON = RESULTS_DIR / "model_comparison_baseline.csv"
DEFAULT_METRICS = RESULTS_DIR / "metrics_lightgbm_test.json"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "analysis"


TARGET_LABELS = {
    "homo": "HOMO energy (eV)",
    "lumo": "LUMO energy (eV)",
    "gap": "HOMO-LUMO gap (eV)",
}


def load_metrics(metrics_path: Path, comparison_path: Path) -> pd.DataFrame:
    """Load per-target metrics into a tidy table."""
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows = []
        for target in TARGET_COLS:
            row = {"target": target}
            row.update(metrics[target])
            rows.append(row)
        return pd.DataFrame(rows)

    comp = pd.read_csv(comparison_path)
    row = comp[(comp["model"] == "lightgbm") & (comp["split"] == "test")].iloc[0]
    return pd.DataFrame(
        [
            {
                "target": target,
                "mae": row[f"{target}_mae"],
                "rmse": row[f"{target}_rmse"],
                "r2": row[f"{target}_r2"],
            }
            for target in TARGET_COLS
        ]
    )


def plot_parity(pred: pd.DataFrame, target: str, output_dir: Path) -> None:
    true_col = f"{target}_true"
    pred_col = f"{target}_pred"
    plt.figure(figsize=(5.5, 5.5))
    sns.scatterplot(data=pred, x=true_col, y=pred_col, s=18, alpha=0.65, edgecolor=None)
    low = min(pred[true_col].min(), pred[pred_col].min())
    high = max(pred[true_col].max(), pred[pred_col].max())
    plt.plot([low, high], [low, high], "r--", linewidth=1.2, label="ideal")
    plt.xlabel(f"True {TARGET_LABELS[target]}")
    plt.ylabel(f"Predicted {TARGET_LABELS[target]}")
    plt.title(f"Parity plot: {target.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"parity_{target}.png", dpi=300)
    plt.close()


def plot_residual(pred: pd.DataFrame, target: str, output_dir: Path) -> None:
    residual_col = f"{target}_residual"
    plt.figure(figsize=(6, 4.5))
    sns.histplot(pred[residual_col], bins=40, kde=True)
    plt.axvline(0.0, color="red", linestyle="--", linewidth=1.2)
    plt.xlabel(f"Residual true - predicted {TARGET_LABELS[target]}")
    plt.ylabel("Count")
    plt.title(f"Residual distribution: {target.upper()}")
    plt.tight_layout()
    plt.savefig(output_dir / f"residual_{target}.png", dpi=300)
    plt.close()


def build_top_error_table(pred: pd.DataFrame, top_n: int) -> pd.DataFrame:
    out = pred.copy()
    error_cols = [f"{target}_abs_error" for target in TARGET_COLS]
    out["average_abs_error"] = out[error_cols].mean(axis=1)
    for target in TARGET_COLS:
        out[f"rank_{target}_error"] = out[f"{target}_abs_error"].rank(method="min", ascending=False)
    out["rank_average_error"] = out["average_abs_error"].rank(method="min", ascending=False)
    return out.sort_values("average_abs_error", ascending=False).head(top_n)


def extract_feature_importance(model_path: Path, output_dir: Path, top_n: int) -> pd.DataFrame:
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        raise TypeError("Expected baseline_lightgbm.joblib to contain MultiOutputRegressor with estimators_.")

    all_rows = []
    for target, estimator in zip(TARGET_COLS, estimators):
        gains = estimator.booster_.feature_importance(importance_type="gain")
        splits = estimator.booster_.feature_importance(importance_type="split")
        rows = pd.DataFrame(
            {
                "target": target,
                "feature": feature_cols,
                "importance_gain": gains,
                "importance_split": splits,
            }
        ).sort_values("importance_gain", ascending=False)
        all_rows.append(rows)

        top = rows.head(top_n).sort_values("importance_gain", ascending=True)
        plt.figure(figsize=(8, max(4.5, top_n * 0.28)))
        plt.barh(top["feature"], top["importance_gain"])
        plt.xlabel("LightGBM gain importance")
        plt.title(f"Top {top_n} features: {target.upper()}")
        plt.tight_layout()
        plt.savefig(output_dir / f"feature_importance_{target}.png", dpi=300)
        plt.close()

    importance = pd.concat(all_rows, ignore_index=True)
    importance.to_csv(output_dir / "feature_importance_lightgbm.csv", index=False, encoding="utf-8")
    return importance


def analyze(
    predictions_path: Path,
    model_path: Path,
    comparison_path: Path,
    metrics_path: Path,
    output_dir: Path,
    top_n: int,
) -> None:
    ensure_dirs(output_dir)
    sns.set_theme(style="whitegrid")

    print(f"Loading predictions: {predictions_path}")
    pred = pd.read_csv(predictions_path)

    metrics = load_metrics(metrics_path, comparison_path)
    metrics.to_csv(output_dir / "target_metrics_summary.csv", index=False, encoding="utf-8")

    print("Generating target plots...")
    for target in tqdm(TARGET_COLS, desc="Plot targets", unit="target"):
        plot_parity(pred, target, output_dir)
        plot_residual(pred, target, output_dir)

    top_errors = build_top_error_table(pred, top_n=top_n)
    top_errors.to_csv(output_dir / "top_errors_lightgbm.csv", index=False, encoding="utf-8")

    print("Extracting LightGBM feature importance...")
    extract_feature_importance(model_path, output_dir, top_n=top_n)

    print("\n=== ANALYSIS SUMMARY ===")
    print(metrics)
    print(f"outputs: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze MolGap LightGBM baseline results")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    analyze(args.predictions, args.model, args.comparison, args.metrics, args.output_dir, args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
