"""
Phase 4 Step 2: True per-target Optuna tuning.
Each target (homo, lumo, gap) gets its own LightGBM hyperparameters.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import (
    METADATA_COLS,
    RESULTS_DIR,
    TARGET_COLS,
    create_split_indices,
    ensure_dirs,
    regression_metrics,
    save_json,
)

OUT_DIR = RESULTS_DIR / "phase4"
PHASE3_FEAT = RESULTS_DIR / "phase3" / "phase3_features.csv"
PHASE3_OPT = RESULTS_DIR / "phase3" / "optimize"
SEED = 42
TRIALS_PER_TARGET = 60


def load_data():
    df = pd.read_csv(PHASE3_FEAT)
    required = set(METADATA_COLS + TARGET_COLS)
    feature_cols = [c for c in df.columns if c not in required]

    gain_path = PHASE3_OPT / "feature_gain.csv"
    if gain_path.exists():
        gain_df = pd.read_csv(gain_path)
        kept = gain_df[gain_df["total_gain"] > 0]["feature"].tolist()
        feature_cols = [c for c in kept if c in df.columns]

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)
    return X, y, feature_cols


def tune_single_target(X_tr, y_tr, X_va, y_va, target_name, n_trials, seed):
    from lightgbm import LGBMRegressor

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 5, 18),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-7, 10.0, log=True),
        }
        model = LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        return float(np.mean(np.abs(pred - y_va)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  [{target_name}] Best MAE: {study.best_value:.4f}")
    return study.best_trial.params


def main():
    from lightgbm import LGBMRegressor

    ensure_dirs(OUT_DIR)
    print("=== Phase 4 Step 2: Per-Target Optuna Tuning ===\n")

    X, y, feature_cols = load_data()
    train_idx, valid_idx, test_idx = create_split_indices(len(X), random_state=SEED)
    tv_idx = np.concatenate([train_idx, valid_idx])

    print(f"  Data: {len(X)} rows, {len(feature_cols)} features")
    print(f"  Split: train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}")
    print(f"  Trials per target: {TRIALS_PER_TARGET}\n")

    all_params = {}
    per_target_preds = []

    for i, target in enumerate(TARGET_COLS):
        print(f"\n{'='*50}")
        print(f"  Tuning {target} ({TRIALS_PER_TARGET} trials)")
        print(f"{'='*50}")

        params = tune_single_target(
            X[train_idx], y[train_idx, i],
            X[valid_idx], y[valid_idx, i],
            target, TRIALS_PER_TARGET, SEED + i
        )
        all_params[target] = params
        save_json(params, OUT_DIR / f"per_target_params_{target}.json")

        # Retrain on train+valid, predict test
        model = LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1, **params)
        model.fit(X[tv_idx], y[tv_idx, i])
        pred = model.predict(X[test_idx])
        per_target_preds.append(pred)

        mae = float(np.mean(np.abs(pred - y[test_idx, i])))
        r2 = float(1 - np.sum((pred - y[test_idx, i])**2) / np.sum((y[test_idx, i] - y[test_idx, i].mean())**2))
        print(f"  [{target}] Test MAE={mae:.4f} R2={r2:.4f}")

    # Combined metrics
    combined_pred = np.column_stack(per_target_preds)
    m = regression_metrics(y[test_idx], combined_pred)

    print(f"\n{'='*50}")
    print(f"  Per-Target Tuned Combined Result")
    print(f"{'='*50}")
    for t in TARGET_COLS:
        print(f"  {t:5s}: MAE={m[t]['mae']:.4f} R2={m[t]['r2']:.4f}")
    print(f"  avg  : MAE={m['average']['mae']:.4f} R2={m['average']['r2']:.4f}")

    save_json({
        "params": all_params,
        "metrics": m,
        "n_trials_per_target": TRIALS_PER_TARGET,
    }, OUT_DIR / "per_target_summary.json")

    print(f"\n  Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
