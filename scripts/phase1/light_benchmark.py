"""
10_light_benchmark.py — lightweight feature/model benchmark for MolGap.

Designed for the local 16 GB RAM environment: compare a small number of feature
sets and models on the existing 10k dataset without expanding data or saving many
model artifacts.
"""

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import (
    DEFAULT_SPLIT_PATH,
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET_COLS,
    ensure_dirs,
    flatten_metrics,
    regression_metrics,
)


DEFAULT_INPUT = PROCESSED_DIR / "features_morgan2048_desc.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "benchmark"
SCAFFOLD_SPLIT_PATH = RESULTS_DIR / "scaffold" / "scaffold_split_indices.npz"
METADATA_AND_TARGETS = {"cid", "mw", "formula", "smiles", "canonical_smiles", *TARGET_COLS}


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with np.load(path, allow_pickle=False) as d:
        return d["train_idx"].astype(int), d["valid_idx"].astype(int), d["test_idx"].astype(int)


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    morgan = [c for c in df.columns if c.startswith("morgan_")]
    desc = [c for c in df.columns if c.startswith("desc_")]
    return {
        "morgan_only": morgan,
        "rdkit_desc_only": desc,
        "morgan_plus_rdkit": morgan + desc,
    }


def build_model(name: str, random_state: int):
    if name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=random_state)),
        ])
    if name == "lightgbm":
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
    if name == "extratrees":
        return ExtraTreesRegressor(n_estimators=150, random_state=random_state, n_jobs=-1)
    raise ValueError(f"Unknown model: {name}")


def evaluate_one(
    df: pd.DataFrame,
    features: list[str],
    split_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    split_name: str,
    feature_set_name: str,
    model_name: str,
    random_state: int,
) -> list[dict]:
    train_idx, valid_idx, test_idx = split_indices
    X = df[features].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    train_valid_idx = np.concatenate([train_idx, valid_idx])
    X_train_valid, y_train_valid = X[train_valid_idx], y[train_valid_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = build_model(model_name, random_state=random_state)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time_valid = time.time() - t0

    t0 = time.time()
    valid_pred = model.predict(X_valid)
    predict_time_valid = time.time() - t0
    valid_metrics = regression_metrics(y_valid, valid_pred)
    valid_row = flatten_metrics(model_name, "valid", valid_metrics)
    valid_row.update({
        "split_type": split_name,
        "feature_set": feature_set_name,
        "n_features": len(features),
        "train_time_sec": train_time_valid,
        "predict_time_sec": predict_time_valid,
    })

    model = build_model(model_name, random_state=random_state)
    t0 = time.time()
    model.fit(X_train_valid, y_train_valid)
    train_time_test = time.time() - t0

    t0 = time.time()
    test_pred = model.predict(X_test)
    predict_time_test = time.time() - t0
    test_metrics = regression_metrics(y_test, test_pred)
    test_row = flatten_metrics(model_name, "test", test_metrics)
    test_row.update({
        "split_type": split_name,
        "feature_set": feature_set_name,
        "n_features": len(features),
        "train_time_sec": train_time_test,
        "predict_time_sec": predict_time_test,
    })
    return [valid_row, test_row]


def plot_benchmark(test_rows: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    for metric, ylabel, fname in [
        ("average_r2", "Average R²", "benchmark_average_r2.png"),
        ("average_mae", "Average MAE (eV)", "benchmark_average_mae.png"),
    ]:
        plt.figure(figsize=(9, 5))
        sns.barplot(
            data=test_rows,
            x="feature_set",
            y=metric,
            hue="model",
        )
        plt.title(f"Test {ylabel} by feature set/model")
        plt.xlabel("Feature set")
        plt.ylabel(ylabel)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=300)
        plt.close()


def run_benchmark(
    input_path: Path,
    output_dir: Path,
    include_extratrees: bool,
    random_state: int,
) -> pd.DataFrame:
    ensure_dirs(output_dir)
    print(f"Loading features: {input_path}")
    df = pd.read_csv(input_path)
    fsets = feature_sets(df)
    splits = {
        "random": load_split(DEFAULT_SPLIT_PATH),
        "scaffold": load_split(SCAFFOLD_SPLIT_PATH),
    }
    models = ["ridge", "lightgbm"] + (["extratrees"] if include_extratrees else [])
    combos = list(product(splits.items(), fsets.items(), models))

    rows = []
    for (split_name, split_idx), (feature_set_name, features), model_name in tqdm(
        combos, desc="Benchmark experiments", unit="exp"
    ):
        if not features:
            tqdm.write(f"Skipping empty feature set: {feature_set_name}")
            continue
        tqdm.write(f"{split_name} | {feature_set_name} ({len(features)} features) | {model_name}")
        rows.extend(
            evaluate_one(
                df,
                features,
                split_idx,
                split_name,
                feature_set_name,
                model_name,
                random_state,
            )
        )

    result = pd.DataFrame(rows)
    result = result[
        ["split_type", "feature_set", "model", "split", "n_features"]
        + [c for c in result.columns if c not in {"split_type", "feature_set", "model", "split", "n_features"}]
    ]
    result.to_csv(output_dir / "light_feature_model_benchmark.csv", index=False, encoding="utf-8")

    test_rows = result[result["split"] == "test"].copy()
    best = test_rows.sort_values(["split_type", "average_mae"]).groupby("split_type").head(1)
    best.to_csv(output_dir / "light_feature_model_benchmark_best_by_split.csv", index=False, encoding="utf-8")

    summary = test_rows.sort_values(["split_type", "average_r2"], ascending=[True, False])
    summary.to_csv(output_dir / "light_feature_model_benchmark_summary.csv", index=False, encoding="utf-8")
    plot_benchmark(test_rows, output_dir)

    print("\n=== LIGHT BENCHMARK SUMMARY ===")
    print(best[["split_type", "feature_set", "model", "average_mae", "average_r2"]])
    print(f"outputs: {output_dir}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight MolGap feature/model benchmark")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-extratrees", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    run_benchmark(args.input, args.output_dir, args.include_extratrees, args.random_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
