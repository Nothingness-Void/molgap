"""
20_phase3_select_and_optimize.py — Phase 3 with feature selection then optimization.

Step 1: Load Phase 3 features (6028 dim)
Step 2: Gain-based feature selection (drop gain=0)
Step 3: Optuna tune LightGBM / XGBoost on selected features
Step 4: CatBoost, HistGBT, per-target LGBM
Step 5: Compare all models

Outputs:
  results/phase3/optimize/selected_feature_stats.json
  results/phase3/optimize/best_params_lgbm.json
  results/phase3/optimize/best_params_xgb.json
  results/phase3/optimize/model_comparison.csv
  results/phase3/optimize/optimize_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

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

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR = RESULTS_DIR / "phase3" / "optimize"
PHASE3_FEAT = RESULTS_DIR / "phase3" / "phase3_features.csv"


# ── Load & Select ─────────────────────────────────────────

def load_and_select_features(seed=42):
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    print(f"Loading: {PHASE3_FEAT}")
    df = pd.read_csv(PHASE3_FEAT)
    required = set(METADATA_COLS + TARGET_COLS)
    feature_cols = [c for c in df.columns if c not in required]
    print(f"  rows={len(df)}, raw features={len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)

    train_idx, _, _ = create_split_indices(len(X), random_state=seed)

    print("  Training quick LightGBM for feature importance...")
    model = MultiOutputRegressor(LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9,
        random_state=seed, n_jobs=-1, verbose=-1,
    ))
    model.fit(X[train_idx], y[train_idx])

    total_gain = np.zeros(len(feature_cols))
    for est in model.estimators_:
        total_gain += est.feature_importances_

    gain_df = pd.DataFrame({"feature": feature_cols, "total_gain": total_gain})
    gain_df = gain_df.sort_values("total_gain", ascending=False)

    kept = gain_df[gain_df["total_gain"] > 0]["feature"].tolist()
    dropped = len(feature_cols) - len(kept)

    by_type = {}
    for f in kept:
        prefix = f.split("_")[0]
        by_type[prefix] = by_type.get(prefix, 0) + 1

    print(f"\n=== FEATURE SELECTION ===")
    print(f"  original: {len(feature_cols)}")
    print(f"  kept:     {len(kept)}")
    print(f"  dropped:  {dropped}")
    print(f"  by type:  {by_type}")

    save_json({
        "original": len(feature_cols),
        "kept": len(kept),
        "dropped": dropped,
        "by_type": by_type,
    }, OUT_DIR / "selected_feature_stats.json")

    gain_df.to_csv(OUT_DIR / "feature_gain.csv", index=False)

    X_sel = df[kept].values.astype(np.float32)
    return X_sel, y, kept


# ── Optuna LightGBM ──────────────────────────────────────

def tune_lgbm(X, y, train_idx, valid_idx, n_trials=80, seed=42):
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[valid_idx], y[valid_idx]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "max_depth": trial.suggest_int("max_depth", 6, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        }
        model = MultiOutputRegressor(LGBMRegressor(
            random_state=seed, n_jobs=-1, verbose=-1, **params))
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        return regression_metrics(y_va, pred)["average"]["mae"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[LightGBM] Best valid MAE: {study.best_value:.4f}")
    print(f"  params: {study.best_trial.params}")
    save_json(study.best_trial.params, OUT_DIR / "best_params_lgbm.json")
    return study.best_trial.params


# ── Optuna XGBoost ────────────────────────────────────────

def tune_xgb(X, y, train_idx, valid_idx, n_trials=60, seed=42):
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[valid_idx], y[valid_idx]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-6, 5.0, log=True),
        }
        model = MultiOutputRegressor(XGBRegressor(
            random_state=seed, n_jobs=-1, verbosity=0, tree_method="hist", **params))
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        return regression_metrics(y_va, pred)["average"]["mae"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[XGBoost] Best valid MAE: {study.best_value:.4f}")
    print(f"  params: {study.best_trial.params}")
    save_json(study.best_trial.params, OUT_DIR / "best_params_xgb.json")
    return study.best_trial.params


# ── Main ──────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lgbm-trials", type=int, default=80)
    p.add_argument("--xgb-trials", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ensure_dirs(OUT_DIR)

    # Step 1-2: load + select
    X, y, feature_cols = load_and_select_features(args.seed)
    train_idx, valid_idx, test_idx = create_split_indices(len(X), random_state=args.seed)
    tv_idx = np.concatenate([train_idx, valid_idx])
    print(f"\nSplit: train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}")

    # Step 3: Optuna
    print("\n" + "="*60)
    print("1/5  Optuna LightGBM")
    print("="*60)
    lgbm_params = tune_lgbm(X, y, train_idx, valid_idx,
                            n_trials=args.lgbm_trials, seed=args.seed)

    print("\n" + "="*60)
    print("2/5  Optuna XGBoost")
    print("="*60)
    xgb_params = tune_xgb(X, y, train_idx, valid_idx,
                          n_trials=args.xgb_trials, seed=args.seed)

    # Step 4-5: All models on test
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    print("\n" + "="*60)
    print("Final test evaluation (train+valid -> test)")
    print("="*60)

    results = []

    # Phase 3 baseline reference
    results.append({
        "model": "Phase3_baseline(no_select)",
        "average_mae": 0.1706, "average_rmse": 0.2353, "average_r2": 0.8755,
        "homo_mae": 0.1448, "homo_r2": 0.8437,
        "lumo_mae": 0.1569, "lumo_r2": 0.9154,
        "gap_mae": 0.2102, "gap_r2": 0.8675,
    })

    def eval_on_test(name, make_model):
        model = make_model()
        model.fit(X[tv_idx], y[tv_idx])
        pred = model.predict(X[test_idx])
        m = regression_metrics(y[test_idx], pred)
        row = {"model": name}
        for t in TARGET_COLS + ["average"]:
            row[f"{t}_mae"] = m[t]["mae"]
            row[f"{t}_rmse"] = m[t]["rmse"]
            row[f"{t}_r2"] = m[t]["r2"]
        print(f"\n  [{name}]")
        for t in TARGET_COLS:
            print(f"    {t:5s}: MAE={m[t]['mae']:.4f}  R2={m[t]['r2']:.4f}")
        print(f"    avg  : MAE={m['average']['mae']:.4f}  R2={m['average']['r2']:.4f}")
        return row

    results.append(eval_on_test("Tuned_LGBM", lambda: MultiOutputRegressor(
        LGBMRegressor(random_state=args.seed, n_jobs=-1, verbose=-1, **lgbm_params))))

    results.append(eval_on_test("Tuned_XGB", lambda: MultiOutputRegressor(
        XGBRegressor(random_state=args.seed, n_jobs=-1, verbosity=0,
                     tree_method="hist", **xgb_params))))

    results.append(eval_on_test("CatBoost", lambda: MultiOutputRegressor(
        CatBoostRegressor(iterations=800, learning_rate=0.06, depth=8,
                          l2_leaf_reg=3, random_seed=args.seed, verbose=0,
                          thread_count=-1))))

    results.append(eval_on_test("HistGBT", lambda: MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=800, learning_rate=0.06,
                                     max_leaf_nodes=63, max_depth=10,
                                     min_samples_leaf=20, random_state=args.seed))))

    # Per-target tuned LGBM
    pt_models = []
    for i, t in enumerate(TARGET_COLS):
        m = LGBMRegressor(random_state=args.seed, n_jobs=-1, verbose=-1, **lgbm_params)
        m.fit(X[tv_idx], y[tv_idx, i])
        pt_models.append(m)
    pred_pt = np.column_stack([m.predict(X[test_idx]) for m in pt_models])
    m_pt = regression_metrics(y[test_idx], pred_pt)
    row_pt = {"model": "PerTarget_LGBM"}
    for t in TARGET_COLS + ["average"]:
        row_pt[f"{t}_mae"] = m_pt[t]["mae"]
        row_pt[f"{t}_rmse"] = m_pt[t]["rmse"]
        row_pt[f"{t}_r2"] = m_pt[t]["r2"]
    print(f"\n  [PerTarget_LGBM]")
    for t in TARGET_COLS:
        print(f"    {t:5s}: MAE={m_pt[t]['mae']:.4f}  R2={m_pt[t]['r2']:.4f}")
    print(f"    avg  : MAE={m_pt['average']['mae']:.4f}  R2={m_pt['average']['r2']:.4f}")
    results.append(row_pt)

    # Save
    comp = pd.DataFrame(results)
    comp = comp.sort_values("average_r2", ascending=False).reset_index(drop=True)
    comp.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    print("\n" + "="*60)
    print("MODEL COMPARISON (sorted by avg R2)")
    print("="*60)
    for _, r in comp.iterrows():
        print(f"  {r['model']:30s}  MAE={r['average_mae']:.4f}  R2={r['average_r2']:.4f}")

    best = comp.iloc[0]
    print(f"\nBest: {best['model']}  MAE={best['average_mae']:.4f}  R2={best['average_r2']:.4f}")

    if best["average_r2"] >= 0.9:
        print(">> TARGET R2 >= 0.9 REACHED")
    else:
        print(f">> Gap to 0.9: {0.9 - best['average_r2']:.4f}")

    save_json({
        "n_features_original": 6028,
        "n_features_selected": len(feature_cols),
        "best_model": best["model"],
        "best_avg_mae": float(best["average_mae"]),
        "best_avg_r2": float(best["average_r2"]),
        "lgbm_params": lgbm_params,
        "xgb_params": xgb_params,
        "all_results": results,
    }, OUT_DIR / "optimize_summary.json")

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    raise SystemExit(main())
