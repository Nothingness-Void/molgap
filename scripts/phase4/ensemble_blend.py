"""
Phase 4 Step 1: Ensemble/Blend of tuned LGBM + XGB + HistGBT.
Optimizes blend weights on validation set, evaluates on test.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

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


def main():
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor

    ensure_dirs(OUT_DIR)
    print("=== Phase 4 Step 1: Ensemble Blend ===\n")

    X, y, feature_cols = load_data()
    train_idx, valid_idx, test_idx = create_split_indices(len(X), random_state=SEED)
    tv_idx = np.concatenate([train_idx, valid_idx])

    with open(PHASE3_OPT / "best_params_lgbm.json") as f:
        lgbm_params = json.load(f)
    with open(PHASE3_OPT / "best_params_xgb.json") as f:
        xgb_params = json.load(f)

    models = {
        "lgbm": MultiOutputRegressor(LGBMRegressor(
            random_state=SEED, n_jobs=-1, verbose=-1, **lgbm_params)),
        "xgb": MultiOutputRegressor(XGBRegressor(
            random_state=SEED, n_jobs=-1, verbosity=0, tree_method="hist", **xgb_params)),
        "hgbt": MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.06, max_leaf_nodes=63,
            max_depth=10, min_samples_leaf=20, random_state=SEED)),
    }

    # Train on train only, predict valid + test
    val_preds = {}
    test_preds = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X[train_idx], y[train_idx])
        val_preds[name] = model.predict(X[valid_idx])
        test_preds[name] = model.predict(X[test_idx])

    # Individual model metrics on test (retrain on train+valid)
    individual_results = []
    retrained_test_preds = {}
    for name, model in models.items():
        print(f"  Retraining {name} on train+valid...")
        model.fit(X[tv_idx], y[tv_idx])
        pred = model.predict(X[test_idx])
        retrained_test_preds[name] = pred
        m = regression_metrics(y[test_idx], pred)
        individual_results.append({"model": name, **{
            f"{t}_{k}": m[t][k] for t in TARGET_COLS + ["average"] for k in ["mae", "r2"]
        }})
        print(f"    {name}: MAE={m['average']['mae']:.4f} R2={m['average']['r2']:.4f}")

    # Optimize blend weights on validation
    names = list(val_preds.keys())
    vp = np.stack([val_preds[n] for n in names])  # (n_models, n_valid, 3)

    def blend_mae(weights):
        w = np.array(weights)
        w = w / w.sum()
        blended = np.tensordot(w, vp, axes=([0], [0]))
        return np.mean(np.abs(blended - y[valid_idx]))

    best_w = None
    best_mae = float("inf")
    # Grid search
    for a in np.arange(0, 1.05, 0.05):
        for b in np.arange(0, 1.05 - a, 0.05):
            c = 1.0 - a - b
            if c < -0.01:
                continue
            mae = blend_mae([a, b, max(c, 0)])
            if mae < best_mae:
                best_mae = mae
                best_w = [a, b, max(c, 0)]

    # Refine with scipy
    res = minimize(blend_mae, best_w, method="Nelder-Mead",
                   options={"maxiter": 1000, "xatol": 1e-5})
    if res.fun < best_mae:
        best_w = res.x.tolist()
    w = np.array(best_w)
    w = w / w.sum()
    blend_weights = {n: float(w[i]) for i, n in enumerate(names)}
    print(f"\n  Blend weights: {blend_weights}")

    # Blend test predictions (retrained models)
    tp = np.stack([retrained_test_preds[n] for n in names])
    blended_pred = np.tensordot(np.array([blend_weights[n] for n in names]), tp, axes=([0], [0]))
    m_blend = regression_metrics(y[test_idx], blended_pred)
    print(f"  Blend: MAE={m_blend['average']['mae']:.4f} R2={m_blend['average']['r2']:.4f}")

    # Simple average
    avg_pred = tp.mean(axis=0)
    m_avg = regression_metrics(y[test_idx], avg_pred)
    print(f"  Average: MAE={m_avg['average']['mae']:.4f} R2={m_avg['average']['r2']:.4f}")

    # Ridge stacking (OOF predictions as features)
    print("\n  Training Ridge stacking...")
    oof_train = np.hstack([val_preds[n] for n in names])  # (n_valid, 3*n_models)
    oof_test = np.hstack([retrained_test_preds[n] for n in names])
    stack_model = MultiOutputRegressor(Ridge(alpha=1.0))
    stack_model.fit(oof_train, y[valid_idx])
    stack_pred = stack_model.predict(oof_test)
    m_stack = regression_metrics(y[test_idx], stack_pred)
    print(f"  Stack: MAE={m_stack['average']['mae']:.4f} R2={m_stack['average']['r2']:.4f}")

    # Save results
    results = []
    results.append({"model": "Phase3_best(LGBM)", "average_mae": 0.1596, "average_r2": 0.8853})
    for r in individual_results:
        results.append(r)
    results.append({"model": "Blend_weighted", "average_mae": m_blend["average"]["mae"],
                     "average_r2": m_blend["average"]["r2"],
                     **{f"{t}_{k}": m_blend[t][k] for t in TARGET_COLS for k in ["mae", "r2"]}})
    results.append({"model": "Blend_average", "average_mae": m_avg["average"]["mae"],
                     "average_r2": m_avg["average"]["r2"],
                     **{f"{t}_{k}": m_avg[t][k] for t in TARGET_COLS for k in ["mae", "r2"]}})
    results.append({"model": "Stack_ridge", "average_mae": m_stack["average"]["mae"],
                     "average_r2": m_stack["average"]["r2"],
                     **{f"{t}_{k}": m_stack[t][k] for t in TARGET_COLS for k in ["mae", "r2"]}})

    pd.DataFrame(results).to_csv(OUT_DIR / "ensemble_comparison.csv", index=False)
    save_json({"blend_weights": blend_weights,
               "blend_mae": m_blend["average"]["mae"],
               "blend_r2": m_blend["average"]["r2"],
               "stack_mae": m_stack["average"]["mae"],
               "stack_r2": m_stack["average"]["r2"]},
              OUT_DIR / "ensemble_summary.json")
    print(f"\n  Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
