"""
03b_feature_selection.py — lightweight gain-based feature selection.

Trains a quick LightGBM on full features, drops zero-gain features,
saves a filtered CSV for downstream training.

Input:
  data/processed/features_morgan2048_desc.csv

Output:
  data/processed/features_selected.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import (
    METADATA_COLS,
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET_COLS,
    ensure_dirs,
    load_or_create_split_indices,
)

DEFAULT_INPUT = PROCESSED_DIR / "features_morgan2048_desc.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "features_selected.csv"


def select_features(
    input_path: Path,
    output_path: Path,
    min_total_gain: float = 0.0,
    random_state: int = 42,
) -> pd.DataFrame:
    print(f"Loading features: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  rows: {len(df)}, columns: {len(df.columns)}")

    required = set(METADATA_COLS + TARGET_COLS)
    feature_cols = [c for c in df.columns if c not in required]
    print(f"  total feature columns: {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)

    train_idx, valid_idx, test_idx, split_msg = load_or_create_split_indices(
        len(df), random_state=random_state
    )
    print(f"  {split_msg}")

    X_train = X[train_idx]
    y_train = y[train_idx]

    print("  training quick LightGBM for feature importance...")
    model = MultiOutputRegressor(
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
    model.fit(X_train, y_train)

    total_gain = np.zeros(len(feature_cols))
    for est in model.estimators_:
        total_gain += est.feature_importances_

    gain_df = pd.DataFrame({
        "feature": feature_cols,
        "total_gain": total_gain,
    }).sort_values("total_gain", ascending=False)

    keep_mask = gain_df["total_gain"] > min_total_gain
    kept_features = gain_df.loc[keep_mask, "feature"].tolist()
    dropped = len(feature_cols) - len(kept_features)

    by_type = {}
    for f in kept_features:
        prefix = f.split("_")[0]
        by_type[prefix] = by_type.get(prefix, 0) + 1

    print(f"\n=== FEATURE SELECTION SUMMARY ===")
    print(f"  original features : {len(feature_cols)}")
    print(f"  kept features     : {len(kept_features)}")
    print(f"  dropped (gain=0)  : {dropped}")
    print(f"  by type           : {by_type}")

    keep_cols = list(required) + kept_features
    out = df[[c for c in keep_cols if c in df.columns]]
    ensure_dirs(output_path.parent)
    out.to_csv(output_path, index=False, encoding="utf-8")
    print(f"  output            : {output_path}")

    ensure_dirs(RESULTS_DIR)
    gain_df.to_csv(RESULTS_DIR / "feature_selection_gain.csv", index=False)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight feature selection")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    select_features(args.input, args.output, args.min_gain, args.random_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
