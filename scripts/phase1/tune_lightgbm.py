"""
13_tune_lightgbm.py — Optuna-based LightGBM hyperparameter tuning.

Searches over LightGBM hyperparameters using the existing random split.
Evaluates on validation set, then retrain best on train+valid and report test.
Also runs scaffold split evaluation with the best hyperparameters.

Outputs:
  results/tuning/optuna_study_summary.csv
  results/tuning/best_params.json
  results/tuning/tuned_vs_baseline_comparison.csv
  results/tuning/tuned_test_predictions_random.csv
  results/tuning/tuned_test_predictions_scaffold.csv
  models/tuned_lightgbm_random.joblib
  models/tuned_lightgbm_scaffold.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

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
    get_feature_target_arrays,
    load_or_create_split_indices,
    murcko_scaffold_smiles,
    regression_metrics,
    save_json,
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_INPUT = PROCESSED_DIR / "features_morgan2048_desc.csv"
TUNING_DIR = RESULTS_DIR / "phase1" / "tuning"


def scaffold_split_indices(
    df: pd.DataFrame, train_frac: float = 0.8, valid_frac: float = 0.1, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaffolds = df["canonical_smiles"].apply(murcko_scaffold_smiles)
    unique_scaffolds = scaffolds.unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_scaffolds)

    n = len(unique_scaffolds)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)

    train_scaffolds = set(unique_scaffolds[:n_train])
    valid_scaffolds = set(unique_scaffolds[n_train : n_train + n_valid])

    train_idx = np.where(scaffolds.isin(train_scaffolds))[0]
    valid_idx = np.where(scaffolds.isin(valid_scaffolds))[0]
    test_idx = np.where(~scaffolds.isin(train_scaffolds | valid_scaffolds))[0]
    return train_idx, valid_idx, test_idx


def make_model(params: dict, seed: int = 42) -> MultiOutputRegressor:
    return MultiOutputRegressor(
        LGBMRegressor(
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
    )


def objective(trial, X_train, y_train, X_valid, y_valid):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        "max_depth": trial.suggest_int("max_depth", 5, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
    }
    model = make_model(params)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    metrics = regression_metrics(y_valid, pred)
    return metrics["average"]["mae"]


def evaluate_and_save(
    params: dict,
    X_train, y_train,
    X_valid, y_valid,
    X_test, y_test,
    meta_test: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
    seed: int = 42,
) -> dict:
    model = make_model(params, seed)
    X_tv = np.concatenate([X_train, X_valid])
    y_tv = np.concatenate([y_train, y_valid])
    model.fit(X_tv, y_tv)

    test_pred = model.predict(X_test)
    test_metrics = regression_metrics(y_test, test_pred)

    pred_table = meta_test.reset_index(drop=True).copy()
    for i, t in enumerate(TARGET_COLS):
        pred_table[f"{t}_true"] = y_test[:, i]
        pred_table[f"{t}_pred"] = test_pred[:, i]
        pred_table[f"{t}_residual"] = y_test[:, i] - test_pred[:, i]
        pred_table[f"{t}_abs_error"] = np.abs(y_test[:, i] - test_pred[:, i])
    pred_table.to_csv(
        TUNING_DIR / f"tuned_test_predictions_{split_name}.csv",
        index=False, encoding="utf-8",
    )

    joblib.dump(
        {
            "model": model,
            "params": params,
            "feature_cols": feature_cols,
            "target_cols": TARGET_COLS,
            "split": split_name,
        },
        MODELS_DIR / f"tuned_lightgbm_{split_name}.joblib",
    )

    return test_metrics


def run_tuning(input_path: Path, n_trials: int = 80, seed: int = 42):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    X, y, feature_cols = get_feature_target_arrays(df)
    print(f"  rows={len(df)}, features={len(feature_cols)}")

    # --- Random split tuning ---
    train_idx, valid_idx, test_idx, msg = load_or_create_split_indices(len(df), random_state=seed)
    print(f"  random split: {msg}")
    print(f"  train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    meta_test = df.loc[test_idx, METADATA_COLS]

    print(f"\n=== Optuna search ({n_trials} trials) ===")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_valid, y_valid),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_trial.params
    print(f"\nBest valid avg MAE: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    ensure_dirs(TUNING_DIR, MODELS_DIR)
    save_json(best_params, TUNING_DIR / "best_params.json")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(TUNING_DIR / "optuna_study_summary.csv", index=False)

    # --- Evaluate on random split test ---
    print("\n=== Tuned model — random split test ===")
    random_metrics = evaluate_and_save(
        best_params, X_train, y_train, X_valid, y_valid,
        X_test, y_test, meta_test, feature_cols, "random", seed,
    )
    for t in TARGET_COLS:
        m = random_metrics[t]
        print(f"  {t:5s}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}")
    avg = random_metrics["average"]
    print(f"  avg  : MAE={avg['mae']:.4f}  RMSE={avg['rmse']:.4f}  R2={avg['r2']:.4f}")

    # --- Evaluate on scaffold split test ---
    print("\n=== Tuned model — scaffold split test ===")
    sc_train, sc_valid, sc_test = scaffold_split_indices(df, seed=seed)
    print(f"  scaffold split: train={len(sc_train)} valid={len(sc_valid)} test={len(sc_test)}")

    scaffold_metrics = evaluate_and_save(
        best_params,
        X[sc_train], y[sc_train],
        X[sc_valid], y[sc_valid],
        X[sc_test], y[sc_test],
        df.loc[sc_test, METADATA_COLS],
        feature_cols, "scaffold", seed,
    )
    for t in TARGET_COLS:
        m = scaffold_metrics[t]
        print(f"  {t:5s}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}")
    avg_sc = scaffold_metrics["average"]
    print(f"  avg  : MAE={avg_sc['mae']:.4f}  RMSE={avg_sc['rmse']:.4f}  R2={avg_sc['r2']:.4f}")

    # --- Comparison table ---
    baseline_random = {"model": "lightgbm_baseline", "split": "random",
                       "average_mae": 0.1755, "average_r2": 0.8952}
    baseline_scaffold = {"model": "lightgbm_baseline", "split": "scaffold",
                         "average_mae": 0.2047, "average_r2": 0.8642}
    tuned_random = {"model": "lightgbm_tuned", "split": "random",
                    "average_mae": avg["mae"], "average_r2": avg["r2"]}
    tuned_scaffold = {"model": "lightgbm_tuned", "split": "scaffold",
                      "average_mae": avg_sc["mae"], "average_r2": avg_sc["r2"]}

    comp = pd.DataFrame([baseline_random, tuned_random, baseline_scaffold, tuned_scaffold])
    comp["mae_improvement"] = ""
    for split in ["random", "scaffold"]:
        bl = comp[(comp["model"] == "lightgbm_baseline") & (comp["split"] == split)]["average_mae"].values[0]
        tu = comp[(comp["model"] == "lightgbm_tuned") & (comp["split"] == split)]["average_mae"].values[0]
        mask = (comp["model"] == "lightgbm_tuned") & (comp["split"] == split)
        comp.loc[mask, "mae_improvement"] = f"{(bl - tu) / bl * 100:.1f}%"

    comp.to_csv(TUNING_DIR / "tuned_vs_baseline_comparison.csv", index=False)

    print("\n=== Tuned vs Baseline ===")
    print(comp.to_string(index=False))

    save_json(
        {
            "best_params": best_params,
            "best_valid_mae": study.best_value,
            "random_test": random_metrics,
            "scaffold_test": scaffold_metrics,
        },
        TUNING_DIR / "tuning_result_summary.json",
    )

    print("\nDone. Results saved to results/tuning/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune LightGBM hyperparameters with Optuna")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_tuning(args.input, n_trials=args.n_trials, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
