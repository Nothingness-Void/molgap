"""
Phase 4 Step 6: SchNet + LightGBM fusion.
Uses SchNet 3D predictions as extra features for LightGBM stacking.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import (
    METADATA_COLS,
    MODELS_DIR,
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
GRAPHS_PATH = OUT_DIR / "pyg_3d_graphs.pt"
SCHNET_MODEL = MODELS_DIR / "gnn_schnet_3d.pt"
SEED = 42


def get_schnet_predictions(data_list, indices, y_mean, y_std, device):
    """Get SchNet predictions for given indices."""
    from scripts.phase4.gnn_schnet_3d import SchNetWrapper
    from torch_geometric.loader import DataLoader

    model = SchNetWrapper(
        hidden_channels=256, num_filters=256, num_interactions=5,
        num_gaussians=50, cutoff=10.0,
    ).to(device)
    model.load_state_dict(torch.load(SCHNET_MODEL, weights_only=True))
    model.eval()

    subset = [data_list[i] for i in indices]
    # Standardize targets same way as training
    for d in subset:
        pass  # y already in original scale since we loaded fresh

    loader = DataLoader(subset, batch_size=64)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast("cuda"):
                out = model(batch.z, batch.pos, batch.batch)
            # Inverse standardize
            out_real = out.cpu().numpy() * y_std + y_mean
            preds.append(out_real)
    return np.vstack(preds)


def main():
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor

    ensure_dirs(OUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Phase 4 Step 6: SchNet + LightGBM Fusion ===\n")

    # Load tabular features
    df = pd.read_csv(PHASE3_FEAT)
    required = set(METADATA_COLS + TARGET_COLS)
    feature_cols = [c for c in df.columns if c not in required]

    gain_path = PHASE3_OPT / "feature_gain.csv"
    if gain_path.exists():
        gain_df = pd.read_csv(gain_path)
        kept = gain_df[gain_df["total_gain"] > 0]["feature"].tolist()
        feature_cols = [c for c in kept if c in df.columns]

    X_tab = df[feature_cols].values.astype(np.float32)
    y_tab = df[TARGET_COLS].values.astype(np.float32)

    # Load 3D graphs
    print("  Loading 3D graphs...")
    data_list = torch.load(GRAPHS_PATH, weights_only=False)

    # Align sizes (tabular may have slightly more rows than graph)
    n = min(len(X_tab), len(data_list))
    X_tab = X_tab[:n]
    y_tab = y_tab[:n]
    data_list = data_list[:n]

    train_idx, valid_idx, test_idx = create_split_indices(n, random_state=SEED)
    tv_idx = np.concatenate([train_idx, valid_idx])

    # Get SchNet predictions for all data
    # First compute y_mean/y_std from training targets
    train_y = y_tab[train_idx]
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    # Need to standardize graph targets for model
    for d in data_list:
        d.y = (d.y - torch.tensor(y_mean)) / torch.tensor(y_std)

    print("  Getting SchNet predictions...")
    from torch_geometric.loader import DataLoader as PyGLoader

    # Load SchNet model directly
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from phase4.gnn_schnet_3d import SchNetWrapper

    model = SchNetWrapper(
        hidden_channels=256, num_filters=256, num_interactions=5,
        num_gaussians=50, cutoff=10.0,
    ).to(device)
    model.load_state_dict(torch.load(SCHNET_MODEL, weights_only=True))
    model.eval()

    all_schnet_preds = []
    loader = PyGLoader(data_list, batch_size=64)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast("cuda"):
                out = model(batch.z, batch.pos, batch.batch)
            out_real = out.cpu().numpy() * y_std + y_mean
            all_schnet_preds.append(out_real)
    schnet_preds = np.vstack(all_schnet_preds)
    print(f"  SchNet predictions shape: {schnet_preds.shape}")

    # Load best LGBM params
    with open(PHASE3_OPT / "best_params_lgbm.json") as f:
        lgbm_params = json.load(f)
    with open(PHASE3_OPT / "best_params_xgb.json") as f:
        xgb_params = json.load(f)

    # === Strategy 1: Tabular + SchNet predictions as extra features ===
    print("\n  Strategy 1: Tabular features + SchNet preds as extra columns")
    X_fused = np.hstack([X_tab, schnet_preds])
    print(f"  Fused features: {X_fused.shape[1]} ({len(feature_cols)} tab + 3 schnet)")

    model_lgbm = MultiOutputRegressor(LGBMRegressor(
        random_state=SEED, n_jobs=-1, verbose=-1, **lgbm_params))
    model_lgbm.fit(X_fused[tv_idx], y_tab[tv_idx])
    pred_fused = model_lgbm.predict(X_fused[test_idx])
    m_fused = regression_metrics(y_tab[test_idx], pred_fused)
    print(f"  LGBM+SchNet fused: MAE={m_fused['average']['mae']:.4f} R2={m_fused['average']['r2']:.4f}")
    for t in TARGET_COLS:
        print(f"    {t:5s}: MAE={m_fused[t]['mae']:.4f} R2={m_fused[t]['r2']:.4f}")

    # === Strategy 2: Ridge stacking of LGBM + XGB + SchNet ===
    print("\n  Strategy 2: Ridge stacking (LGBM + XGB + SchNet)")

    # Train LGBM and XGB on train, predict valid
    lgbm_model = MultiOutputRegressor(LGBMRegressor(
        random_state=SEED, n_jobs=-1, verbose=-1, **lgbm_params))
    lgbm_model.fit(X_tab[train_idx], y_tab[train_idx])
    lgbm_val = lgbm_model.predict(X_tab[valid_idx])

    xgb_model = MultiOutputRegressor(XGBRegressor(
        random_state=SEED, n_jobs=-1, verbosity=0, tree_method="hist", **xgb_params))
    xgb_model.fit(X_tab[train_idx], y_tab[train_idx])
    xgb_val = xgb_model.predict(X_tab[valid_idx])

    schnet_val = schnet_preds[valid_idx]

    # Stack features = predictions from all models
    stack_val = np.hstack([lgbm_val, xgb_val, schnet_val])

    # Retrain base models on train+valid for test predictions
    lgbm_model.fit(X_tab[tv_idx], y_tab[tv_idx])
    lgbm_test = lgbm_model.predict(X_tab[test_idx])

    xgb_model.fit(X_tab[tv_idx], y_tab[tv_idx])
    xgb_test = xgb_model.predict(X_tab[test_idx])

    schnet_test = schnet_preds[test_idx]
    stack_test = np.hstack([lgbm_test, xgb_test, schnet_test])

    ridge = MultiOutputRegressor(Ridge(alpha=1.0))
    ridge.fit(stack_val, y_tab[valid_idx])
    pred_stack = ridge.predict(stack_test)
    m_stack = regression_metrics(y_tab[test_idx], pred_stack)
    print(f"  Ridge stack: MAE={m_stack['average']['mae']:.4f} R2={m_stack['average']['r2']:.4f}")
    for t in TARGET_COLS:
        print(f"    {t:5s}: MAE={m_stack[t]['mae']:.4f} R2={m_stack[t]['r2']:.4f}")

    # === Strategy 3: Weighted blend of LGBM + SchNet ===
    print("\n  Strategy 3: Weighted blend optimization")
    best_w = None
    best_mae = float("inf")
    for alpha in np.arange(0, 1.01, 0.05):
        blended = alpha * lgbm_val + (1 - alpha) * schnet_val
        mae = np.mean(np.abs(blended - y_tab[valid_idx]))
        if mae < best_mae:
            best_mae = mae
            best_w = alpha

    blended_test = best_w * lgbm_test + (1 - best_w) * schnet_test
    m_blend = regression_metrics(y_tab[test_idx], blended_test)
    print(f"  Best alpha (LGBM weight): {best_w:.2f}")
    print(f"  Blend: MAE={m_blend['average']['mae']:.4f} R2={m_blend['average']['r2']:.4f}")
    for t in TARGET_COLS:
        print(f"    {t:5s}: MAE={m_blend[t]['mae']:.4f} R2={m_blend[t]['r2']:.4f}")

    # === Summary ===
    print(f"\n{'='*55}")
    print(f"  FUSION RESULTS SUMMARY")
    print(f"{'='*55}")

    results = [
        {"model": "SchNet_3D_alone", "average_mae": 0.1492, "average_r2": 0.8942},
        {"model": "Tuned_LGBM_alone", "average_mae": 0.1596, "average_r2": 0.8853},
        {"model": "LGBM+SchNet_features", "average_mae": m_fused["average"]["mae"],
         "average_r2": m_fused["average"]["r2"],
         **{f"{t}_{k}": m_fused[t][k] for t in TARGET_COLS for k in ["mae", "r2"]}},
        {"model": "Ridge_stack(LGBM+XGB+SchNet)", "average_mae": m_stack["average"]["mae"],
         "average_r2": m_stack["average"]["r2"],
         **{f"{t}_{k}": m_stack[t][k] for t in TARGET_COLS for k in ["mae", "r2"]}},
        {"model": f"Blend(LGBM*{best_w:.2f}+SchNet*{1-best_w:.2f})",
         "average_mae": m_blend["average"]["mae"], "average_r2": m_blend["average"]["r2"],
         **{f"{t}_{k}": m_blend[t][k] for t in TARGET_COLS for k in ["mae", "r2"]}},
    ]

    for r in sorted(results, key=lambda x: -x["average_r2"]):
        print(f"  {r['model']:40s} MAE={r['average_mae']:.4f} R2={r['average_r2']:.4f}")

    best = max(results, key=lambda x: x["average_r2"])
    print(f"\n  BEST: {best['model']}  R2={best['average_r2']:.4f}")
    if best["average_r2"] >= 0.9:
        print("  >>> R2 >= 0.9 TARGET REACHED! <<<")

    pd.DataFrame(results).to_csv(OUT_DIR / "fusion_comparison.csv", index=False)
    save_json({
        "best_model": best["model"],
        "best_mae": float(best["average_mae"]),
        "best_r2": float(best["average_r2"]),
        "lgbm_schnet_blend_alpha": float(best_w),
        "all_results": results,
    }, OUT_DIR / "fusion_summary.json")

    print(f"\n  Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
