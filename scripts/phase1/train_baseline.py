"""
04_train_baseline.py — train first MolGap baseline models.

Input:
  data/processed/features_morgan2048_desc.csv

Outputs:
  models/baseline_*.joblib
  results/metrics_*.json
  results/model_comparison_baseline.csv
  results/test_predictions_*.csv

The current small raw dataset is only for pipeline verification. Metrics from a
few hundred rows should not be treated as final scientific results.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import (
    METADATA_COLS,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET_COLS,
    ensure_dirs,
    flatten_metrics,
    load_or_create_split_indices,
    regression_metrics,
    save_json,
)


DEFAULT_INPUT = PROCESSED_DIR / "features_morgan2048_desc.csv"


def build_models(random_state: int = 42) -> dict:
    """Return baseline model candidates."""
    models = {
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=random_state)),
            ]
        ),
        "extratrees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "randomforest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    try:
        from lightgbm import LGBMRegressor
        from sklearn.multioutput import MultiOutputRegressor

        models["lightgbm"] = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        )
    except Exception:
        print("LightGBM unavailable; skipping lightgbm baseline.")

    return models


def make_prediction_table(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Build a traceable test prediction table."""
    out = meta.reset_index(drop=True).copy()
    for i, target in enumerate(TARGET_COLS):
        out[f"{target}_true"] = y_true[:, i]
        out[f"{target}_pred"] = y_pred[:, i]
        out[f"{target}_residual"] = y_true[:, i] - y_pred[:, i]
        out[f"{target}_abs_error"] = np.abs(y_true[:, i] - y_pred[:, i])
    return out


def train_baselines(input_path: Path, random_state: int = 42) -> pd.DataFrame:
    """Train baseline models and save metrics, predictions, and model files."""
    print(f"Loading features: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  rows: {len(df)} columns: {len(df.columns)}")

    required = METADATA_COLS + TARGET_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_cols = [c for c in df.columns if c not in required]
    if not feature_cols:
        raise ValueError("No feature columns found.")

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)

    train_idx, valid_idx, test_idx, split_msg = load_or_create_split_indices(
        len(df), random_state=random_state
    )
    print(split_msg)
    print(f"  train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]
    X_train_valid = X[np.concatenate([train_idx, valid_idx])]
    y_train_valid = y[np.concatenate([train_idx, valid_idx])]
    X_test = X[test_idx]
    y_test = y[test_idx]
    meta_test = df.loc[test_idx, METADATA_COLS]

    ensure_dirs(MODELS_DIR, RESULTS_DIR)
    models = build_models(random_state=random_state)
    comparison_rows = []
    best_name = None
    best_avg_mae = float("inf")

    for name, model in tqdm(models.items(), desc="Train baseline models", unit="model"):
        tqdm.write(f"\nTraining model: {name}")
        model.fit(X_train, y_train)

        valid_pred = model.predict(X_valid)
        valid_metrics = regression_metrics(y_valid, valid_pred)
        save_json(valid_metrics, RESULTS_DIR / "phase1" / "baseline" / f"metrics_{name}_valid.json")
        comparison_rows.append(flatten_metrics(name, "valid", valid_metrics))
        tqdm.write(
            f"  valid avg MAE={valid_metrics['average']['mae']:.4f} "
            f"avg R2={valid_metrics['average']['r2']:.4f}"
        )

        if valid_metrics["average"]["mae"] < best_avg_mae:
            best_avg_mae = valid_metrics["average"]["mae"]
            best_name = name

        # Refit on train+valid before final test evaluation.
        model.fit(X_train_valid, y_train_valid)
        test_pred = model.predict(X_test)
        test_metrics = regression_metrics(y_test, test_pred)
        save_json(test_metrics, RESULTS_DIR / "phase1" / "baseline" / f"metrics_{name}_test.json")
        comparison_rows.append(flatten_metrics(name, "test", test_metrics))
        tqdm.write(
            f"  test  avg MAE={test_metrics['average']['mae']:.4f} "
            f"avg R2={test_metrics['average']['r2']:.4f}"
        )

        pred_table = make_prediction_table(meta_test, y_test, test_pred)
        pred_table.to_csv(RESULTS_DIR / f"test_predictions_{name}.csv", index=False, encoding="utf-8")
        joblib.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
                "target_cols": TARGET_COLS,
                "metadata_cols": METADATA_COLS,
                "random_state": random_state,
            },
            MODELS_DIR / f"baseline_{name}.joblib",
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = RESULTS_DIR / "phase1" / "baseline" / "model_comparison_baseline.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")

    print("\n=== BASELINE SUMMARY ===")
    print(f"feature columns : {len(feature_cols)}")
    print(f"best valid model: {best_name} (avg MAE={best_avg_mae:.4f})")
    print(f"comparison      : {comparison_path}")
    print("NOTE: current small sample metrics are for pipeline verification only.")
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MolGap baseline models")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_baselines(args.input, random_state=args.random_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
