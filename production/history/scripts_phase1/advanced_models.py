"""
15_advanced_models.py — try advanced model strategies on the current dataset.

Strategies:
  1. Per-target LightGBM (independent tuned params per HOMO/LUMO/gap)
  2. Stacking ensemble (LightGBM + Ridge + ExtraTrees → meta Ridge)
  3. XGBoost baseline
  4. CatBoost baseline (if installed)
  5. LightGBM with DART booster

Outputs:
  results/advanced/advanced_model_comparison.csv
  results/advanced/advanced_experiment_summary.json
  models/advanced_best.joblib
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from molgap.utils import (
    METADATA_COLS,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET_COLS,
    ensure_dirs,
    load_or_create_split_indices,
    regression_metrics,
    save_json,
)

warnings.filterwarnings("ignore")

RESULTS_ADV = RESULTS_DIR / "phase1" / "advanced"


def per_target_lightgbm(X_train, y_train, X_test, y_test, seed=42):
    from lightgbm import LGBMRegressor

    per_target_params = {
        "homo": {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 40,
                 "max_depth": 10, "min_child_samples": 20, "subsample": 0.85,
                 "colsample_bytree": 0.6, "reg_alpha": 0.01, "reg_lambda": 0.01},
        "lumo": {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 50,
                 "max_depth": 10, "min_child_samples": 15, "subsample": 0.85,
                 "colsample_bytree": 0.7, "reg_alpha": 0.001, "reg_lambda": 0.001},
        "gap":  {"n_estimators": 1000, "learning_rate": 0.04, "num_leaves": 60,
                 "max_depth": 12, "min_child_samples": 10, "subsample": 0.8,
                 "colsample_bytree": 0.6, "reg_alpha": 0.005, "reg_lambda": 0.005},
    }
    preds = np.zeros_like(y_test)
    models = {}
    for i, target in enumerate(TARGET_COLS):
        params = per_target_params[target]
        m = LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **params)
        m.fit(X_train, y_train[:, i])
        preds[:, i] = m.predict(X_test)
        models[target] = m
    return regression_metrics(y_test, preds), models


def stacking_ensemble(X_train, y_train, X_test, y_test, seed=42):
    from lightgbm import LGBMRegressor

    base_estimators = [
        ("lgbm", LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                num_leaves=31, random_state=seed, n_jobs=-1, verbose=-1)),
        ("et", ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1)),
        ("ridge", Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])),
    ]
    preds = np.zeros_like(y_test)
    models = {}
    for i, target in enumerate(TARGET_COLS):
        stacker = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1,
        )
        stacker.fit(X_train, y_train[:, i])
        preds[:, i] = stacker.predict(X_test)
        models[target] = stacker
    return regression_metrics(y_test, preds), models


def xgboost_model(X_train, y_train, X_test, y_test, seed=42):
    from xgboost import XGBRegressor
    model = MultiOutputRegressor(XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=8,
        subsample=0.85, colsample_bytree=0.6,
        reg_alpha=0.01, reg_lambda=0.01,
        random_state=seed, n_jobs=-1, verbosity=0,
    ))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return regression_metrics(y_test, pred), model


def catboost_model(X_train, y_train, X_test, y_test, seed=42):
    from catboost import CatBoostRegressor
    model = MultiOutputRegressor(CatBoostRegressor(
        iterations=800, learning_rate=0.05, depth=8,
        subsample=0.85, random_seed=seed, verbose=0,
    ))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return regression_metrics(y_test, pred), model


def dart_lightgbm(X_train, y_train, X_test, y_test, seed=42):
    from lightgbm import LGBMRegressor
    model = MultiOutputRegressor(LGBMRegressor(
        boosting_type="dart", n_estimators=500, learning_rate=0.05,
        num_leaves=40, max_depth=10, subsample=0.85,
        colsample_bytree=0.6, random_state=seed, n_jobs=-1, verbose=-1,
    ))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return regression_metrics(y_test, pred), model


def tuned_lgbm_baseline(X_train, y_train, X_test, y_test, seed=42):
    from lightgbm import LGBMRegressor
    model = MultiOutputRegressor(LGBMRegressor(
        n_estimators=800, learning_rate=0.06, num_leaves=39,
        max_depth=10, min_child_samples=23, subsample=0.888,
        colsample_bytree=0.604, reg_alpha=0.00556, reg_lambda=0.00920,
        random_state=seed, n_jobs=-1, verbose=-1,
    ))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return regression_metrics(y_test, pred), model


def run(input_path: Path, seed: int = 42):
    ensure_dirs(RESULTS_ADV, MODELS_DIR)
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    required = set(METADATA_COLS + TARGET_COLS)
    feature_cols = [c for c in df.columns if c not in required]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)
    print(f"  rows={len(df)}, features={len(feature_cols)}")

    train_idx, valid_idx, test_idx, msg = load_or_create_split_indices(len(df), random_state=seed)
    tv_idx = np.concatenate([train_idx, valid_idx])
    print(f"  split: {msg}")
    print(f"  train+valid={len(tv_idx)}, test={len(test_idx)}")

    X_tv, y_tv = X[tv_idx], y[tv_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    experiments = {}
    best_name = None
    best_mae = float("inf")
    best_model = None

    strategies = [
        ("tuned_lgbm", tuned_lgbm_baseline),
        ("per_target_lgbm", per_target_lightgbm),
        ("stacking", stacking_ensemble),
        ("dart_lgbm", dart_lightgbm),
    ]

    # Try optional models
    try:
        import xgboost
        strategies.append(("xgboost", xgboost_model))
    except ImportError:
        print("XGBoost not installed, skipping.")

    try:
        import catboost
        strategies.append(("catboost", catboost_model))
    except ImportError:
        print("CatBoost not installed, skipping.")

    for name, func in strategies:
        print(f"\n--- {name} ---")
        try:
            metrics, model = func(X_tv, y_tv, X_test, y_test, seed)
            experiments[name] = metrics
            avg = metrics["average"]
            print(f"  avg MAE={avg['mae']:.4f}  R2={avg['r2']:.4f}")
            for t in TARGET_COLS:
                m = metrics[t]
                print(f"  {t:5s}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}")
            if avg["mae"] < best_mae:
                best_mae = avg["mae"]
                best_name = name
                best_model = model
        except Exception as e:
            print(f"  FAILED: {e}")

    # Save comparison
    rows = []
    for name, m in experiments.items():
        row = {"model": name}
        for target in TARGET_COLS + ["average"]:
            for metric in ["mae", "rmse", "r2"]:
                row[f"{target}_{metric}"] = m[target][metric]
        rows.append(row)
    comp = pd.DataFrame(rows).sort_values("average_mae")
    comp.to_csv(RESULTS_ADV / "advanced_model_comparison.csv", index=False)

    print(f"\n{'='*60}")
    print(f"RANKING:")
    print(comp[["model", "average_mae", "average_r2"]].to_string(index=False))
    print(f"\nBest: {best_name} (avg MAE={best_mae:.4f})")

    if best_model is not None:
        joblib.dump({"model": best_model, "name": best_name,
                     "feature_cols": feature_cols}, MODELS_DIR / "advanced_best.joblib")

    save_json(experiments, RESULTS_ADV / "advanced_experiment_summary.json")
    print(f"Results saved to {RESULTS_ADV}/")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "features_selected.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.input, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
