"""
06_y_randomization.py — Y-randomization validation for MolGap.

The script shuffles target values, retrains the selected regressor, and checks
whether performance collapses. This helps verify that the baseline is learning a
real structure-property relation rather than chance correlation or leakage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from molgap.utils import (
    DEFAULT_SPLIT_PATH,
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET_COLS,
    ensure_dirs,
    flatten_metrics,
    get_feature_target_arrays,
    load_split_indices_or_raise,
    regression_metrics,
    save_json,
)


DEFAULT_INPUT = PROCESSED_DIR / "features_morgan2048_desc.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "phase1" / "y_randomization"
DEFAULT_REAL_METRICS = RESULTS_DIR / "phase1" / "baseline" / "metrics_lightgbm_test.json"


def make_lightgbm(random_state: int):
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    return MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    )


def make_extratrees(random_state: int):
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)


def build_model(model_name: str, random_state: int):
    if model_name == "lightgbm":
        return make_lightgbm(random_state)
    if model_name == "extratrees":
        return make_extratrees(random_state)
    raise ValueError(f"Unsupported model: {model_name}")


def load_real_average_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    import json

    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics.get("average", {})


def plot_distribution(summary: pd.DataFrame, metric: str, real_value: float | None, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 4.5))
    sns.histplot(summary[f"average_{metric}"], bins=20, kde=True)
    if real_value is not None:
        plt.axvline(real_value, color="red", linestyle="--", linewidth=1.5, label="real model")
        plt.legend()
    plt.xlabel(f"Y-randomized average {metric.upper()}")
    plt.ylabel("Count")
    plt.title(f"Y-randomization {metric.upper()} distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_y_randomization(
    input_path: Path,
    output_dir: Path,
    model_name: str,
    n_runs: int,
    random_state: int,
    real_metrics_path: Path,
) -> pd.DataFrame:
    ensure_dirs(output_dir)
    print(f"Loading features: {input_path}")
    df = pd.read_csv(input_path)
    X, y, _ = get_feature_target_arrays(df)
    train_idx, valid_idx, test_idx = load_split_indices_or_raise(len(df), DEFAULT_SPLIT_PATH)

    train_valid_idx = np.concatenate([train_idx, valid_idx])
    X_train_valid = X[train_valid_idx]
    y_train_valid = y[train_valid_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    rows = []
    rng = np.random.default_rng(random_state)
    for run in tqdm(range(n_runs), desc="Y-randomization", unit="run"):
        shuffled = y_train_valid.copy()
        for target_idx in range(shuffled.shape[1]):
            shuffled[:, target_idx] = rng.permutation(shuffled[:, target_idx])

        model = build_model(model_name, random_state + run)
        model.fit(X_train_valid, shuffled)
        pred = model.predict(X_test)
        metrics = regression_metrics(y_test, pred)
        row = flatten_metrics(model_name, f"y_random_{run:03d}", metrics)
        row["run"] = run
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "y_randomization_summary.csv", index=False, encoding="utf-8")

    real_avg = load_real_average_metrics(real_metrics_path)
    aggregate = {
        "model": model_name,
        "n_runs": n_runs,
        "real_average": real_avg,
        "randomized_average_mean": {
            "mae": float(summary["average_mae"].mean()),
            "rmse": float(summary["average_rmse"].mean()),
            "r2": float(summary["average_r2"].mean()),
        },
        "randomized_average_std": {
            "mae": float(summary["average_mae"].std()),
            "rmse": float(summary["average_rmse"].std()),
            "r2": float(summary["average_r2"].std()),
        },
    }
    save_json(aggregate, output_dir / "y_randomization_summary.json")

    plot_distribution(summary, "r2", real_avg.get("r2"), output_dir / "y_randomization_r2_distribution.png")
    plot_distribution(summary, "mae", real_avg.get("mae"), output_dir / "y_randomization_mae_distribution.png")

    print("\n=== Y-RANDOMIZATION SUMMARY ===")
    print(f"real avg MAE: {real_avg.get('mae', 'n/a')}")
    print(f"real avg R2 : {real_avg.get('r2', 'n/a')}")
    print(f"random avg MAE mean: {summary['average_mae'].mean():.4f}")
    print(f"random avg R2  mean: {summary['average_r2'].mean():.4f}")
    print(f"outputs: {output_dir}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MolGap Y-randomization analysis")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", choices=["lightgbm", "extratrees"], default="lightgbm")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--real-metrics", type=Path, default=DEFAULT_REAL_METRICS)
    args = parser.parse_args()

    run_y_randomization(args.input, args.output_dir, args.model, args.n_runs, args.random_state, args.real_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
