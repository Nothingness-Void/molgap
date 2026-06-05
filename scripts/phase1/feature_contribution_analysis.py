"""
12_feature_contribution_analysis.py — complete feature contribution analysis.

This script complements the first LightGBM feature-importance output with the
minimum report-ready analyses usually expected in a molecular ML project:
  - target-specific LightGBM gain/split importance
  - feature-group contribution (Morgan vs RDKit descriptors)
  - overall and descriptor-only rankings
  - top-feature overlap among HOMO/LUMO/gap
  - Ridge coefficient importance when available
  - lightweight permutation importance on a small test-set sample

SHAP is intentionally not used here to keep runtime/dependencies suitable for the
local 16 GB RAM environment.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import (
    DEFAULT_SPLIT_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET_COLS,
    ensure_dirs,
    load_model_bundle,
    load_split_indices_or_raise,
)


DEFAULT_FEATURES = PROCESSED_DIR / "features_morgan2048_desc.csv"
DEFAULT_LIGHTGBM = MODELS_DIR / "baseline_lightgbm.joblib"
DEFAULT_RIDGE = MODELS_DIR / "baseline_ridge.joblib"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "phase1" / "feature_contribution"


def feature_type(feature: str) -> str:
    if feature.startswith("morgan_"):
        return "morgan"
    if feature.startswith("desc_"):
        return "rdkit_desc"
    return "other"


def plot_barh(df: pd.DataFrame, value_col: str, label_col: str, title: str, output_path: Path) -> None:
    data = df.sort_values(value_col, ascending=True)
    plt.figure(figsize=(9, max(4.8, len(data) * 0.28)))
    plt.barh(data[label_col], data[value_col])
    plt.xlabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def extract_lightgbm_importance(model_path: Path, output_dir: Path, top_n: int) -> pd.DataFrame:
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    feature_cols = list(bundle["feature_cols"])
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        raise TypeError("Expected MultiOutputRegressor with estimators_ in LightGBM bundle.")

    all_rows = []
    for target, estimator in zip(TARGET_COLS, estimators):
        gain = estimator.booster_.feature_importance(importance_type="gain")
        split = estimator.booster_.feature_importance(importance_type="split")
        gain_sum = gain.sum() if gain.sum() > 0 else 1.0
        split_sum = split.sum() if split.sum() > 0 else 1.0
        rows = pd.DataFrame({
            "target": target,
            "feature": feature_cols,
            "feature_type": [feature_type(f) for f in feature_cols],
            "importance_gain": gain,
            "importance_split": split,
            "importance_gain_norm": gain / gain_sum,
            "importance_split_norm": split / split_sum,
        }).sort_values("importance_gain", ascending=False)
        all_rows.append(rows)
        plot_barh(
            rows.head(top_n),
            "importance_gain",
            "feature",
            f"Top {top_n} LightGBM gain features: {target.upper()}",
            output_dir / f"top_features_{target}.png",
        )

    importance = pd.concat(all_rows, ignore_index=True)
    importance.to_csv(output_dir / "feature_importance_lightgbm_gain_split.csv", index=False, encoding="utf-8")
    return importance


def summarize_importance(importance: pd.DataFrame, output_dir: Path, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_target = importance.groupby(["target", "feature_type"], as_index=False).agg(
        gain_norm_sum=("importance_gain_norm", "sum"),
        split_norm_sum=("importance_split_norm", "sum"),
        n_nonzero_gain=("importance_gain", lambda s: int((s > 0).sum())),
        n_features=("feature", "count"),
    )
    by_target.to_csv(output_dir / "feature_importance_summary_by_target.csv", index=False, encoding="utf-8")

    group = importance.groupby("feature_type", as_index=False).agg(
        gain_norm_sum=("importance_gain_norm", "sum"),
        split_norm_sum=("importance_split_norm", "sum"),
        mean_gain_norm=("importance_gain_norm", "mean"),
        n_nonzero_gain=("importance_gain", lambda s: int((s > 0).sum())),
        n_features=("feature", "count"),
    )
    # Normalize sums over three targets to average target contribution.
    group["gain_norm_mean_over_targets"] = group["gain_norm_sum"] / len(TARGET_COLS)
    group["split_norm_mean_over_targets"] = group["split_norm_sum"] / len(TARGET_COLS)
    group.to_csv(output_dir / "feature_group_importance.csv", index=False, encoding="utf-8")

    plt.figure(figsize=(6, 4.5))
    sns.barplot(data=group, x="feature_type", y="gain_norm_mean_over_targets")
    plt.xlabel("Feature group")
    plt.ylabel("Mean normalized gain contribution")
    plt.title("Feature group contribution")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_group_importance.png", dpi=300)
    plt.close()

    overall = importance.groupby(["feature", "feature_type"], as_index=False).agg(
        mean_gain_norm=("importance_gain_norm", "mean"),
        sum_gain=("importance_gain", "sum"),
        mean_split_norm=("importance_split_norm", "mean"),
        sum_split=("importance_split", "sum"),
    ).sort_values("mean_gain_norm", ascending=False)
    overall.to_csv(output_dir / "top_features_overall.csv", index=False, encoding="utf-8")
    plot_barh(
        overall.head(top_n),
        "mean_gain_norm",
        "feature",
        f"Top {top_n} overall features (mean normalized gain)",
        output_dir / "top_features_overall.png",
    )

    rdkit = overall[overall["feature_type"] == "rdkit_desc"].copy()
    rdkit.to_csv(output_dir / "top_rdkit_descriptors.csv", index=False, encoding="utf-8")
    plot_barh(
        rdkit.head(top_n),
        "mean_gain_norm",
        "feature",
        f"Top {top_n} RDKit descriptors",
        output_dir / "rdkit_descriptor_importance.png",
    )

    morgan = overall[overall["feature_type"] == "morgan"].copy()
    morgan.to_csv(output_dir / "top_morgan_bits.csv", index=False, encoding="utf-8")
    return overall, group


def target_overlap(importance: pd.DataFrame, output_dir: Path, top_n: int) -> pd.DataFrame:
    top_sets = {
        target: set(
            importance[importance["target"] == target]
            .sort_values("importance_gain", ascending=False)
            .head(top_n)["feature"]
        )
        for target in TARGET_COLS
    }
    rows = []
    for a, b in itertools.combinations(TARGET_COLS, 2):
        inter = top_sets[a] & top_sets[b]
        union = top_sets[a] | top_sets[b]
        rows.append({
            "target_a": a,
            "target_b": b,
            "top_n": top_n,
            "intersection_size": len(inter),
            "union_size": len(union),
            "jaccard": len(inter) / len(union) if union else 0.0,
            "overlap_features": ";".join(sorted(inter)),
        })
    all_inter = set.intersection(*top_sets.values())
    rows.append({
        "target_a": "homo_lumo_gap",
        "target_b": "all",
        "top_n": top_n,
        "intersection_size": len(all_inter),
        "union_size": len(set.union(*top_sets.values())),
        "jaccard": np.nan,
        "overlap_features": ";".join(sorted(all_inter)),
    })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "target_feature_overlap.csv", index=False, encoding="utf-8")
    return df


def ridge_coefficients(ridge_path: Path, output_dir: Path) -> pd.DataFrame | None:
    if not ridge_path.exists():
        return None
    bundle = load_model_bundle(ridge_path)
    model = bundle["model"]
    feature_cols = list(bundle["feature_cols"])
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        return None
    coef = np.asarray(coef)
    rows = []
    for target, target_coef in zip(TARGET_COLS, coef):
        abs_coef = np.abs(target_coef)
        norm = abs_coef / (abs_coef.sum() if abs_coef.sum() > 0 else 1.0)
        rows.extend({
            "target": target,
            "feature": feature,
            "feature_type": feature_type(feature),
            "coef": float(c),
            "abs_coef": float(abs_c),
            "abs_coef_norm": float(n),
        } for feature, c, abs_c, n in zip(feature_cols, target_coef, abs_coef, norm))
    df = pd.DataFrame(rows).sort_values(["target", "abs_coef"], ascending=[True, False])
    df.to_csv(output_dir / "ridge_coefficient_importance.csv", index=False, encoding="utf-8")
    return df


def average_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]))


def permutation_importance_subset(
    features_path: Path,
    model_path: Path,
    output_dir: Path,
    candidate_features: list[str],
    sample_size: int,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    print("Running lightweight permutation importance...")
    df = pd.read_csv(features_path)
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    feature_cols = list(bundle["feature_cols"])
    feature_index = {f: i for i, f in enumerate(feature_cols)}
    candidate_features = [f for f in candidate_features if f in feature_index]

    _, _, test_idx = load_split_indices_or_raise(len(df), DEFAULT_SPLIT_PATH)
    rng = np.random.default_rng(random_state)
    if len(test_idx) > sample_size:
        test_idx = rng.choice(test_idx, size=sample_size, replace=False)

    X = df[feature_cols].values.astype(np.float32)[test_idx]
    y = df[TARGET_COLS].values.astype(np.float32)[test_idx]
    baseline_pred = model.predict(X)
    baseline_score = average_r2_score(y, baseline_pred)

    rows = []
    for feature in tqdm(candidate_features, desc="Permutation features", unit="feature"):
        idx = feature_index[feature]
        scores = []
        for repeat in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, idx] = rng.permutation(X_perm[:, idx])
            scores.append(average_r2_score(y, model.predict(X_perm)))
        scores = np.array(scores)
        rows.append({
            "feature": feature,
            "feature_type": feature_type(feature),
            "baseline_average_r2": baseline_score,
            "permuted_average_r2_mean": float(scores.mean()),
            "permuted_average_r2_std": float(scores.std()),
            "importance_r2_drop_mean": float(baseline_score - scores.mean()),
            "importance_r2_drop_std": float(scores.std()),
            "n_repeats": n_repeats,
            "sample_size": len(test_idx),
        })

    result = pd.DataFrame(rows).sort_values("importance_r2_drop_mean", ascending=False)
    result.to_csv(output_dir / "permutation_importance_lightgbm_sample.csv", index=False, encoding="utf-8")
    plot_barh(
        result.head(25),
        "importance_r2_drop_mean",
        "feature",
        "Top permutation importance features (average R² drop)",
        output_dir / "permutation_importance_top.png",
    )
    return result


def run_analysis(
    features_path: Path,
    lightgbm_path: Path,
    ridge_path: Path,
    output_dir: Path,
    top_n: int,
    permutation_top_n: int,
    permutation_sample_size: int,
    permutation_repeats: int,
    random_state: int,
) -> None:
    ensure_dirs(output_dir)
    sns.set_theme(style="whitegrid")

    importance = extract_lightgbm_importance(lightgbm_path, output_dir, top_n=top_n)
    overall, group = summarize_importance(importance, output_dir, top_n=top_n)
    overlap = target_overlap(importance, output_dir, top_n=top_n)
    ridge = ridge_coefficients(ridge_path, output_dir)

    candidate_features = overall.head(permutation_top_n)["feature"].tolist()
    perm = permutation_importance_subset(
        features_path,
        lightgbm_path,
        output_dir,
        candidate_features=candidate_features,
        sample_size=permutation_sample_size,
        n_repeats=permutation_repeats,
        random_state=random_state,
    )

    print("\n=== FEATURE CONTRIBUTION SUMMARY ===")
    print("Feature group contribution:")
    print(group[["feature_type", "gain_norm_mean_over_targets", "n_nonzero_gain", "n_features"]])
    print("\nTop overall features:")
    print(overall.head(10)[["feature", "feature_type", "mean_gain_norm"]])
    print("\nTop permutation features:")
    print(perm.head(10)[["feature", "feature_type", "importance_r2_drop_mean"]])
    if ridge is not None:
        print("\nRidge coefficient importance saved.")
    print(f"outputs: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MolGap feature contribution analysis")
    parser.add_argument("--features", type=Path, default=PROCESSED_DIR / "features_morgan2048_desc.csv")
    parser.add_argument("--lightgbm", type=Path, default=MODELS_DIR / "baseline_lightgbm.joblib")
    parser.add_argument("--ridge", type=Path, default=MODELS_DIR / "baseline_ridge.joblib")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "phase1" / "feature_contribution")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--permutation-top-n", type=int, default=50)
    parser.add_argument("--permutation-sample-size", type=int, default=1000)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    run_analysis(
        args.features,
        args.lightgbm,
        args.ridge,
        args.output_dir,
        top_n=args.top_n,
        permutation_top_n=args.permutation_top_n,
        permutation_sample_size=args.permutation_sample_size,
        permutation_repeats=args.permutation_repeats,
        random_state=args.random_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
