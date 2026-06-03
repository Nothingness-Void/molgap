"""
09_predict_commercial.py — predict HOMO/LUMO/gap for commercial molecules.

Input is a user-curated commercial molecule CSV. The script canonicalizes SMILES,
generates the same Morgan+RDKit features used by the PubChemQC baseline, aligns
columns to the trained model bundle, predicts HOMO/LUMO/gap, and attaches simple
confidence proxies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
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
    build_feature_row_from_smiles,
    canonicalize_smiles,
    ensure_dirs,
    get_feature_target_arrays,
    load_model_bundle,
    load_split_indices_or_raise,
    save_json,
)


DEFAULT_INPUT = Path("data/commercial/commercial_molecules.csv")
DEFAULT_FALLBACK_INPUT = Path("data/commercial/commercial_molecules_template.csv")
DEFAULT_OUTPUT = RESULTS_DIR / "database" / "commercial_molgap_predictions_v1.csv"
DEFAULT_SUMMARY = RESULTS_DIR / "database" / "commercial_prediction_summary.json"
DEFAULT_TRAIN_FEATURES = PROCESSED_DIR / "features_morgan2048_desc.csv"

REQUIRED_COLS = ["name", "supplier", "smiles"]
OPTIONAL_COLS = [
    "catalog_id",
    "cid",
    "formula",
    "mw",
    "category",
    "application",
    "reference_url",
    "notes",
]
MODEL_PATHS = {
    "lightgbm": MODELS_DIR / "baseline_lightgbm.joblib",
    "extratrees": MODELS_DIR / "baseline_extratrees.joblib",
    "randomforest": MODELS_DIR / "baseline_randomforest.joblib",
}
MODEL_VERSION = "lightgbm_morgan_rdkit_v1"
PREDICTION_SOURCE = "ML_from_PubChemQC"


def prepare_commercial_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonicalize commercial SMILES and build model-aligned feature matrix."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required commercial columns: {missing}")
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = ""

    rows = []
    feature_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Commercial features", unit="mol"):
        out = row.to_dict()
        can = canonicalize_smiles(row.get("smiles"))
        out["canonical_smiles"] = can
        if can is None:
            out["prediction_status"] = "invalid_smiles"
            rows.append(out)
            continue
        generated = build_feature_row_from_smiles(can)
        if generated is None:
            out["prediction_status"] = "feature_failed"
            rows.append(out)
            continue

        aligned = {col: generated.get(col, 0.0) for col in feature_cols}
        feature_rows.append(aligned)
        out["prediction_status"] = "ok"
        out["feature_row_idx"] = len(feature_rows) - 1
        rows.append(out)

    meta = pd.DataFrame(rows)
    X = pd.DataFrame(feature_rows, columns=feature_cols)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return meta, X


def applicability_distance(train_features_path: Path, X_commercial: np.ndarray, pca_components: int) -> np.ndarray:
    """Distance from commercial molecules to nearest PubChemQC training molecule."""
    train_df = pd.read_csv(train_features_path)
    X_all, _, feature_cols = get_feature_target_arrays(train_df)
    train_idx, valid_idx, _ = load_split_indices_or_raise(len(train_df), DEFAULT_SPLIT_PATH)
    train_valid_idx = np.concatenate([train_idx, valid_idx])
    X_train = X_all[train_valid_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_com_scaled = scaler.transform(X_commercial)
    n_components = min(pca_components, X_train_scaled.shape[1], X_train_scaled.shape[0] - 1, len(X_commercial))
    if n_components < 1:
        return np.zeros(len(X_commercial), dtype=float)
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_com_pca = pca.transform(X_com_scaled)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_train_pca)
    distances, _ = nn.kneighbors(X_com_pca)
    return distances[:, 0]


def assign_bins(values: np.ndarray) -> list[str]:
    """Assign high/medium/low confidence from an uncertainty score."""
    if len(values) == 0:
        return []
    ranks = pd.Series(values).rank(pct=True).values
    bins = []
    for rank in ranks:
        if rank <= 1 / 3:
            bins.append("high")
        elif rank <= 2 / 3:
            bins.append("medium")
        else:
            bins.append("low")
    return bins


def predict_commercial(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    train_features_path: Path,
    reference_model: str,
    pca_components: int,
) -> pd.DataFrame:
    if not input_path.exists() and DEFAULT_FALLBACK_INPUT.exists():
        print(f"Input {input_path} not found; using template for smoke test: {DEFAULT_FALLBACK_INPUT}")
        input_path = DEFAULT_FALLBACK_INPUT

    ensure_dirs(output_path.parent, summary_path.parent)
    print(f"Loading commercial molecules: {input_path}")
    commercial = pd.read_csv(input_path)

    ref_bundle = load_model_bundle(MODEL_PATHS[reference_model])
    feature_cols = ref_bundle["feature_cols"]
    meta, X_df = prepare_commercial_features(commercial, feature_cols)
    ok_mask = meta["prediction_status"] == "ok"

    for target in TARGET_COLS:
        meta[f"{target}_pred"] = np.nan
    meta["model_disagreement"] = np.nan
    meta["applicability_distance"] = np.nan
    meta["uncertainty_score"] = np.nan
    meta["confidence_bin"] = "unavailable"
    meta["prediction_source"] = PREDICTION_SOURCE
    meta["model_version"] = MODEL_VERSION

    if X_df.empty:
        meta.to_csv(output_path, index=False, encoding="utf-8")
        save_json({"n_input": len(meta), "n_predicted": 0}, summary_path)
        return meta

    X_df = X_df.astype(np.float32)
    X_array = X_df.to_numpy(dtype=np.float32)
    predictions = {}
    for name, path in tqdm(MODEL_PATHS.items(), desc="Predict models", unit="model"):
        if path.exists():
            bundle = load_model_bundle(path)
            pred_input = X_df if name == "lightgbm" else X_array
            predictions[name] = bundle["model"].predict(pred_input)

    ref_pred = predictions[reference_model]
    ok_indices = meta.index[ok_mask].to_numpy()
    for i, target in enumerate(TARGET_COLS):
        meta.loc[ok_indices, f"{target}_pred"] = ref_pred[:, i]

    if len(predictions) >= 2:
        stack = np.stack(list(predictions.values()), axis=0)
        disagreement = stack.std(axis=0).mean(axis=1)
    else:
        disagreement = np.zeros(len(X), dtype=float)

    distances = applicability_distance(train_features_path, X_array, pca_components=pca_components)
    dis_rank = pd.Series(disagreement).rank(pct=True).values
    dist_rank = pd.Series(distances).rank(pct=True).values
    uncertainty = (dis_rank + dist_rank) / 2.0

    meta.loc[ok_indices, "model_disagreement"] = disagreement
    meta.loc[ok_indices, "applicability_distance"] = distances
    meta.loc[ok_indices, "uncertainty_score"] = uncertainty
    meta.loc[ok_indices, "confidence_bin"] = assign_bins(uncertainty)

    output_cols = [
        "name",
        "supplier",
        "catalog_id",
        "cid",
        "smiles",
        "canonical_smiles",
        "formula",
        "mw",
        "category",
        "application",
        "homo_pred",
        "lumo_pred",
        "gap_pred",
        "confidence_bin",
        "uncertainty_score",
        "model_disagreement",
        "applicability_distance",
        "prediction_status",
        "prediction_source",
        "model_version",
        "reference_url",
        "notes",
    ]
    for col in output_cols:
        if col not in meta.columns:
            meta[col] = ""
    out = meta[output_cols]
    out.to_csv(output_path, index=False, encoding="utf-8")

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "n_input": int(len(meta)),
        "n_predicted": int(ok_mask.sum()),
        "n_invalid_or_failed": int((~ok_mask).sum()),
        "reference_model": reference_model,
        "prediction_source": PREDICTION_SOURCE,
        "model_version": MODEL_VERSION,
    }
    save_json(summary, summary_path)

    print("\n=== COMMERCIAL PREDICTION SUMMARY ===")
    print(summary)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict commercial MolGap database entries")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--train-features", type=Path, default=DEFAULT_TRAIN_FEATURES)
    parser.add_argument("--reference-model", choices=list(MODEL_PATHS), default="lightgbm")
    parser.add_argument("--pca-components", type=int, default=50)
    args = parser.parse_args()

    predict_commercial(
        args.input,
        args.output,
        args.summary,
        args.train_features,
        args.reference_model,
        args.pca_components,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
