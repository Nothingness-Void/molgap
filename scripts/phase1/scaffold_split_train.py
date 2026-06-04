"""
08_scaffold_split_train.py — scaffold-split validation for MolGap.

Random split can overestimate performance when similar molecules appear in both
train and test sets. This script groups molecules by Bemis-Murcko scaffold and
holds out entire scaffolds for validation/test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
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
    get_feature_target_arrays,
    murcko_scaffold_smiles,
    regression_metrics,
    save_json,
)


DEFAULT_INPUT = PROCESSED_DIR / "features_morgan2048_desc.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "scaffold"


def build_scaffold_models(random_state: int = 42, include_extratrees: bool = True) -> dict:
    models = {
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=random_state)),
            ]
        )
    }
    if include_extratrees:
        models["extratrees"] = ExtraTreesRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
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
        print("LightGBM unavailable; skipping lightgbm scaffold baseline.")
    return models


def make_prediction_table(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    out = meta.reset_index(drop=True).copy()
    for i, target in enumerate(TARGET_COLS):
        out[f"{target}_true"] = y_true[:, i]
        out[f"{target}_pred"] = y_pred[:, i]
        out[f"{target}_residual"] = y_true[:, i] - y_pred[:, i]
        out[f"{target}_abs_error"] = np.abs(y_true[:, i] - y_pred[:, i])
    return out


def scaffold_split_indices(
    df: pd.DataFrame,
    valid_size: float,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    print("Generating Bemis-Murcko scaffolds...")
    scaffolds = [
        murcko_scaffold_smiles(smi) or "INVALID_SCAFFOLD"
        for smi in tqdm(df["canonical_smiles"], desc="Scaffolds", unit="mol")
    ]
    scaffold_df = pd.DataFrame({"row_idx": np.arange(len(df)), "scaffold": scaffolds})
    groups = scaffold_df.groupby("scaffold")["row_idx"].apply(list).reset_index()
    groups["n_molecules"] = groups["row_idx"].map(len)

    rng = np.random.default_rng(random_state)
    groups = groups.sample(frac=1.0, random_state=random_state).sort_values(
        "n_molecules", ascending=False
    )

    target_test = int(round(len(df) * test_size))
    target_valid = int(round(len(df) * valid_size))
    test_groups = []
    valid_groups = []
    train_groups = []
    n_test = 0
    n_valid = 0

    # Greedy assignment: large scaffold groups first, with a bit of randomness from the pre-shuffle.
    for _, row in groups.iterrows():
        group = row["row_idx"]
        if n_test < target_test:
            test_groups.extend(group)
            n_test += len(group)
        elif n_valid < target_valid:
            valid_groups.extend(group)
            n_valid += len(group)
        else:
            train_groups.extend(group)

    train_idx = np.array(sorted(train_groups), dtype=int)
    valid_idx = np.array(sorted(valid_groups), dtype=int)
    test_idx = np.array(sorted(test_groups), dtype=int)

    summary = groups.drop(columns=["row_idx"]).copy()
    summary["split"] = "train"
    summary.loc[summary["scaffold"].isin(scaffold_df.loc[test_idx, "scaffold"].unique()), "split"] = "test"
    summary.loc[summary["scaffold"].isin(scaffold_df.loc[valid_idx, "scaffold"].unique()), "split"] = "valid"
    return train_idx, valid_idx, test_idx, summary


def save_random_vs_scaffold(scaffold_comparison: pd.DataFrame, output_dir: Path) -> None:
    random_path = RESULTS_DIR / "model_comparison_baseline.csv"
    if not random_path.exists():
        return
    random_comp = pd.read_csv(random_path)
    random_lgbm = random_comp[(random_comp["model"] == "lightgbm") & (random_comp["split"] == "test")].copy()
    scaffold_lgbm = scaffold_comparison[
        (scaffold_comparison["model"] == "lightgbm") & (scaffold_comparison["split"] == "test")
    ].copy()
    if random_lgbm.empty or scaffold_lgbm.empty:
        return
    random_lgbm["evaluation"] = "random_split"
    scaffold_lgbm["evaluation"] = "scaffold_split"
    combined = pd.concat([random_lgbm, scaffold_lgbm], ignore_index=True, sort=False)
    combined.to_csv(output_dir / "random_vs_scaffold_comparison.csv", index=False, encoding="utf-8")


def train_scaffold_models(
    input_path: Path,
    output_dir: Path,
    valid_size: float,
    test_size: float,
    random_state: int,
    include_extratrees: bool,
) -> pd.DataFrame:
    ensure_dirs(output_dir, MODELS_DIR)
    print(f"Loading features: {input_path}")
    df = pd.read_csv(input_path)
    X, y, feature_cols = get_feature_target_arrays(df)

    train_idx, valid_idx, test_idx, scaffold_summary = scaffold_split_indices(
        df, valid_size=valid_size, test_size=test_size, random_state=random_state
    )
    scaffold_summary.to_csv(output_dir / "scaffold_split_summary.csv", index=False, encoding="utf-8")
    np.savez(
        output_dir / "scaffold_split_indices.npz",
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        n_samples=np.array([len(df)], dtype=int),
    )

    print(f"scaffold split: train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}")
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    X_train_valid = X[np.concatenate([train_idx, valid_idx])]
    y_train_valid = y[np.concatenate([train_idx, valid_idx])]
    X_test, y_test = X[test_idx], y[test_idx]
    meta_test = df.loc[test_idx, METADATA_COLS]

    rows = []
    best_name = None
    best_mae = float("inf")
    models = build_scaffold_models(random_state=random_state, include_extratrees=include_extratrees)
    for name, model in tqdm(models.items(), desc="Train scaffold models", unit="model"):
        tqdm.write(f"Training scaffold model: {name}")
        model.fit(X_train, y_train)
        valid_pred = model.predict(X_valid)
        valid_metrics = regression_metrics(y_valid, valid_pred)
        rows.append(flatten_metrics(name, "valid", valid_metrics))
        if valid_metrics["average"]["mae"] < best_mae:
            best_mae = valid_metrics["average"]["mae"]
            best_name = name

        model.fit(X_train_valid, y_train_valid)
        test_pred = model.predict(X_test)
        test_metrics = regression_metrics(y_test, test_pred)
        rows.append(flatten_metrics(name, "test", test_metrics))
        save_json(test_metrics, output_dir / f"metrics_{name}_scaffold_test.json")
        pred_table = make_prediction_table(meta_test, y_test, test_pred)
        pred_table.to_csv(output_dir / f"test_predictions_{name}_scaffold.csv", index=False, encoding="utf-8")
        joblib.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
                "target_cols": TARGET_COLS,
                "metadata_cols": METADATA_COLS,
                "random_state": random_state,
                "split": "scaffold",
            },
            MODELS_DIR / f"scaffold_{name}.joblib",
        )
        tqdm.write(
            f"  test avg MAE={test_metrics['average']['mae']:.4f} "
            f"avg R2={test_metrics['average']['r2']:.4f}"
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "model_comparison_scaffold.csv", index=False, encoding="utf-8")
    save_random_vs_scaffold(comparison, output_dir)

    print("\n=== SCAFFOLD SUMMARY ===")
    print(f"best valid model: {best_name} (avg MAE={best_mae:.4f})")
    print(f"outputs: {output_dir}")
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MolGap scaffold-split validation")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-extratrees", action="store_true")
    args = parser.parse_args()

    train_scaffold_models(
        args.input,
        args.output_dir,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state,
        include_extratrees=not args.skip_extratrees,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
