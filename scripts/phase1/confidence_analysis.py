"""
07_confidence_analysis.py — confidence proxies for MolGap predictions.

This script estimates prediction reliability using two practical signals:
  1. disagreement among existing baseline models
  2. applicability-domain distance to nearest training molecules

These are uncertainty proxies, not calibrated probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import (
    DEFAULT_SPLIT_PATH,
    METADATA_COLS,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET_COLS,
    ensure_dirs,
    get_feature_target_arrays,
    load_model_bundle,
    load_split_indices_or_raise,
)


DEFAULT_INPUT = PROCESSED_DIR / "features_morgan2048_desc.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "phase1" / "confidence"
DEFAULT_MODELS = {
    "extratrees": MODELS_DIR / "baseline_extratrees.joblib",
    "randomforest": MODELS_DIR / "baseline_randomforest.joblib",
    "lightgbm": MODELS_DIR / "baseline_lightgbm.joblib",
}


def load_predictions(model_paths: dict[str, Path], X_test: np.ndarray) -> dict[str, np.ndarray]:
    preds = {}
    for name, path in tqdm(model_paths.items(), desc="Predict baseline models", unit="model"):
        bundle = load_model_bundle(path)
        preds[name] = bundle["model"].predict(X_test)
    return preds


def compute_applicability_distance(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Nearest-neighbor distance in scaled PCA space."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_components = min(n_components, X_train_scaled.shape[1], X_train_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_train_pca)
    distances, _ = nn.kneighbors(X_test_pca)
    return distances[:, 0]


def assign_confidence_bins(score: pd.Series) -> pd.Series:
    """Assign high/medium/low confidence from an uncertainty score."""
    try:
        return pd.qcut(score, q=3, labels=["high", "medium", "low"], duplicates="drop")
    except ValueError:
        return pd.Series(["medium"] * len(score), index=score.index)


def build_confidence_table(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    y_test: np.ndarray,
    preds: dict[str, np.ndarray],
    distances: np.ndarray,
    reference_model: str,
) -> pd.DataFrame:
    meta = df.loc[test_idx, METADATA_COLS].reset_index(drop=True).copy()
    pred_stack = np.stack([preds[name] for name in preds], axis=0)
    disagreement = pred_stack.std(axis=0)
    reference_pred = preds[reference_model]

    out = meta
    out["applicability_distance"] = distances

    for i, target in enumerate(TARGET_COLS):
        out[f"{target}_true"] = y_test[:, i]
        out[f"{target}_pred"] = reference_pred[:, i]
        out[f"{target}_abs_error"] = np.abs(y_test[:, i] - reference_pred[:, i])
        out[f"{target}_model_disagreement"] = disagreement[:, i]

    out["average_abs_error"] = out[[f"{t}_abs_error" for t in TARGET_COLS]].mean(axis=1)
    out["average_model_disagreement"] = out[[f"{t}_model_disagreement" for t in TARGET_COLS]].mean(axis=1)

    # Combine rank-normalized uncertainty proxies into one confidence score.
    disagreement_rank = out["average_model_disagreement"].rank(pct=True)
    distance_rank = out["applicability_distance"].rank(pct=True)
    out["uncertainty_score"] = (disagreement_rank + distance_rank) / 2.0
    out["confidence_bin"] = assign_confidence_bins(out["uncertainty_score"])
    return out


def summarize_bins(conf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bin_name, group in conf.groupby("confidence_bin", observed=True):
        row = {
            "confidence_bin": bin_name,
            "n": len(group),
            "average_abs_error": group["average_abs_error"].mean(),
            "average_model_disagreement": group["average_model_disagreement"].mean(),
            "applicability_distance": group["applicability_distance"].mean(),
        }
        for target in TARGET_COLS:
            row[f"{target}_mae"] = group[f"{target}_abs_error"].mean()
            row[f"{target}_disagreement"] = group[f"{target}_model_disagreement"].mean()
        rows.append(row)
    order = {"high": 0, "medium": 1, "low": 2}
    return pd.DataFrame(rows).sort_values("confidence_bin", key=lambda s: s.map(order))


def plot_confidence(conf: pd.DataFrame, bins: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(6, 4.5))
    sns.scatterplot(
        data=conf,
        x="average_model_disagreement",
        y="average_abs_error",
        hue="confidence_bin",
        alpha=0.7,
        s=24,
    )
    plt.xlabel("Average model disagreement (eV)")
    plt.ylabel("Average absolute error (eV)")
    plt.title("Error vs model-disagreement uncertainty")
    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_uncertainty.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    sns.barplot(data=bins, x="confidence_bin", y="average_abs_error", order=["high", "medium", "low"])
    plt.xlabel("Confidence bin")
    plt.ylabel("Observed average absolute error (eV)")
    plt.title("Observed error by confidence bin")
    plt.tight_layout()
    plt.savefig(output_dir / "error_by_confidence_bin.png", dpi=300)
    plt.close()


def run_confidence_analysis(
    input_path: Path,
    output_dir: Path,
    reference_model: str,
    pca_components: int,
) -> pd.DataFrame:
    ensure_dirs(output_dir)
    sns.set_theme(style="whitegrid")

    print(f"Loading features: {input_path}")
    df = pd.read_csv(input_path)
    X, y, _ = get_feature_target_arrays(df)
    train_idx, valid_idx, test_idx = load_split_indices_or_raise(len(df), DEFAULT_SPLIT_PATH)
    train_valid_idx = np.concatenate([train_idx, valid_idx])

    model_paths = {name: path for name, path in DEFAULT_MODELS.items() if path.exists()}
    if reference_model not in model_paths:
        raise FileNotFoundError(f"Missing reference model: {reference_model}")
    if len(model_paths) < 2:
        raise RuntimeError("Need at least two trained models for disagreement analysis.")

    X_test = X[test_idx]
    y_test = y[test_idx]
    preds = load_predictions(model_paths, X_test)

    print("Computing applicability-domain distance...")
    distances = compute_applicability_distance(X[train_valid_idx], X_test, n_components=pca_components)

    conf = build_confidence_table(df, test_idx, y_test, preds, distances, reference_model)
    bins = summarize_bins(conf)

    conf.to_csv(output_dir / "confidence_predictions.csv", index=False, encoding="utf-8")
    bins.to_csv(output_dir / "error_by_confidence_bin.csv", index=False, encoding="utf-8")
    conf[["applicability_distance"]].describe().to_csv(
        output_dir / "applicability_distance_summary.csv", encoding="utf-8"
    )
    bins.to_csv(output_dir / "confidence_summary.csv", index=False, encoding="utf-8")
    plot_confidence(conf, bins, output_dir)

    print("\n=== CONFIDENCE SUMMARY ===")
    print(bins)
    print(f"outputs: {output_dir}")
    return conf


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze MolGap prediction confidence proxies")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-model", choices=list(DEFAULT_MODELS), default="lightgbm")
    parser.add_argument("--pca-components", type=int, default=50)
    args = parser.parse_args()

    run_confidence_analysis(args.input, args.output_dir, args.reference_model, args.pca_components)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
