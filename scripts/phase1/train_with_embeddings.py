"""
14_train_with_embeddings.py — train models using embedding features.

Compares:
  1. Embeddings only (ChemBERTa / MolFormer / both)
  2. Traditional features only (Morgan + RDKit)
  3. Fusion (traditional + embeddings)

Input:
  data/processed/features_selected.csv       (traditional features)
  data/processed/embeddings_chemberta.csv    (from Colab)
  data/processed/embeddings_molformer.csv    (from Colab)
  data/processed/embeddings_all.csv          (merged, from Colab)

Outputs:
  results/embeddings/embedding_model_comparison.csv
  results/embeddings/fusion_vs_traditional.csv
  results/embeddings/embedding_experiment_summary.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from molgap.utils import (
    METADATA_COLS,
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET_COLS,
    ensure_dirs,
    flatten_metrics,
    load_or_create_split_indices,
    regression_metrics,
    save_json,
)

EMB_DIR = PROCESSED_DIR
RESULTS_EMB = RESULTS_DIR / "phase1" / "embeddings"


def load_and_merge(trad_path, emb_path):
    trad = pd.read_csv(trad_path)
    emb = pd.read_csv(emb_path)
    merged = trad.merge(emb, on="cid", how="inner")
    if len(merged) < len(trad) * 0.95:
        print(f"  WARNING: merge dropped {len(trad) - len(merged)} rows")
    return merged


def get_feature_cols(df, prefix_filter=None):
    required = set(METADATA_COLS + TARGET_COLS)
    cols = [c for c in df.columns if c not in required]
    if prefix_filter:
        cols = [c for c in cols if any(c.startswith(p) for p in prefix_filter)]
    return cols


def train_evaluate(X_train, y_train, X_test, y_test, model_name="lightgbm", seed=42):
    if model_name == "lightgbm":
        model = MultiOutputRegressor(LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9,
            random_state=seed, n_jobs=-1, verbose=-1,
        ))
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=seed)),
        ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return regression_metrics(y_test, pred)


def run_experiment(trad_path: Path, seed: int = 42):
    ensure_dirs(RESULTS_EMB)
    print(f"Traditional features: {trad_path}")
    trad_df = pd.read_csv(trad_path)
    n = len(trad_df)
    print(f"  rows: {n}")

    train_idx, valid_idx, test_idx, msg = load_or_create_split_indices(n, random_state=seed)
    tv_idx = np.concatenate([train_idx, valid_idx])
    print(f"  split: {msg}")
    print(f"  train+valid={len(tv_idx)}, test={len(test_idx)}")

    trad_feat_cols = get_feature_cols(trad_df)
    y = trad_df[TARGET_COLS].values.astype(np.float32)

    experiments = {}

    # Traditional only
    print("\n--- Traditional features only ---")
    X_trad = trad_df[trad_feat_cols].values.astype(np.float32)
    for model_name in ["ridge", "lightgbm"]:
        m = train_evaluate(X_trad[tv_idx], y[tv_idx], X_trad[test_idx], y[test_idx], model_name, seed)
        key = f"traditional_{model_name}"
        experiments[key] = m
        print(f"  {key}: avg MAE={m['average']['mae']:.4f}, R2={m['average']['r2']:.4f}")

    # Embedding experiments
    emb_configs = [
        ("chemberta", "embeddings_chemberta.csv", ["chemberta_z"]),
        ("molformer", "embeddings_molformer.csv", ["molformer_x"]),
        ("both_emb", "embeddings_all.csv", ["chemberta_z", "molformer_x"]),
    ]

    for emb_name, emb_file, prefixes in emb_configs:
        emb_path = EMB_DIR / emb_file
        if not emb_path.exists():
            print(f"\n--- {emb_name}: SKIPPED ({emb_file} not found) ---")
            continue

        print(f"\n--- {emb_name} ---")
        merged = load_and_merge(trad_path, emb_path)
        y_m = merged[TARGET_COLS].values.astype(np.float32)

        # Embedding only
        emb_cols = get_feature_cols(merged, prefixes)
        if emb_cols:
            X_emb = merged[emb_cols].values.astype(np.float32)
            for model_name in ["ridge", "lightgbm"]:
                m = train_evaluate(X_emb[tv_idx], y_m[tv_idx], X_emb[test_idx], y_m[test_idx], model_name, seed)
                key = f"{emb_name}_only_{model_name}"
                experiments[key] = m
                print(f"  {key}: avg MAE={m['average']['mae']:.4f}, R2={m['average']['r2']:.4f} ({len(emb_cols)} dims)")

        # Fusion: traditional + embedding
        all_cols = trad_feat_cols + [c for c in merged.columns if c not in set(METADATA_COLS + TARGET_COLS + trad_feat_cols)]
        X_fuse = merged[all_cols].values.astype(np.float32)
        for model_name in ["ridge", "lightgbm"]:
            m = train_evaluate(X_fuse[tv_idx], y_m[tv_idx], X_fuse[test_idx], y_m[test_idx], model_name, seed)
            key = f"fusion_{emb_name}_{model_name}"
            experiments[key] = m
            print(f"  {key}: avg MAE={m['average']['mae']:.4f}, R2={m['average']['r2']:.4f} ({len(all_cols)} dims)")

    # Summary table
    rows = []
    for key, m in experiments.items():
        row = {"experiment": key}
        for target in TARGET_COLS + ["average"]:
            for metric in ["mae", "rmse", "r2"]:
                row[f"{target}_{metric}"] = m[target][metric]
        rows.append(row)

    comp = pd.DataFrame(rows).sort_values("average_mae")
    comp.to_csv(RESULTS_EMB / "embedding_model_comparison.csv", index=False)
    print(f"\n=== RANKING (by avg MAE) ===")
    print(comp[["experiment", "average_mae", "average_r2"]].to_string(index=False))

    save_json(
        {k: v for k, v in experiments.items()},
        RESULTS_EMB / "embedding_experiment_summary.json",
    )
    print(f"\nResults saved to {RESULTS_EMB}/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train with embedding features")
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "features_selected.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment(args.input, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
