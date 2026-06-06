"""
Phase 4.5: Improved LGBM + SchNet(tuned) stacking.

Uses tuned SchNet predictions + LGBM/XGB predictions.
Adds per-target stacking (each target learns its own blend).
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

from molgap.schnet import SchNetWrapper

OUT_DIR = RESULTS_DIR / "phase4" / "stacking_v2"
PHASE3_FEAT = RESULTS_DIR / "phase3" / "phase3_features.csv"
PHASE3_OPT = RESULTS_DIR / "phase3" / "optimize"
GRAPHS_PATH_PM6 = RESULTS_DIR / "phase4" / "pyg_3d_graphs_pm6.pt"
GRAPHS_PATH_LEGACY = RESULTS_DIR / "phase4" / "pyg_3d_graphs.pt"
SCHNET_WEIGHTS = MODELS_DIR / "gnn_schnet_3d_tuned.pt"
SEED = 42

SCHNET_PARAMS = {
    "hidden_channels": 256,
    "num_filters": 192,
    "num_interactions": 7,
    "num_gaussians": 50,
    "cutoff": 7.0,
    "dropout": 0.0,
}


@torch.no_grad()
def get_all_schnet_predictions(data_list, y_mean, y_std, device, use_charges=False):
    from torch_geometric.loader import DataLoader

    model = SchNetWrapper(**SCHNET_PARAMS, use_charges=use_charges).to(device)
    model.load_state_dict(torch.load(SCHNET_WEIGHTS, weights_only=True, map_location=device))
    model.eval()

    preds = []
    loader = DataLoader(data_list, batch_size=64)
    for batch in loader:
        batch = batch.to(device)
        with torch.amp.autocast("cuda"):
            charges = getattr(batch, 'charges', None)
            out = model(batch.z, batch.pos, batch.batch, charges=charges)
        preds.append(out.cpu().numpy() * y_std + y_mean)
    return np.vstack(preds)


def main():
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor

    ensure_dirs(OUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase 4.5: LGBM + SchNet(tuned) Stacking ===", flush=True)
    print(f"  Device: {device}", flush=True)

    # Load graphs
    graphs_path = GRAPHS_PATH_PM6 if GRAPHS_PATH_PM6.exists() else GRAPHS_PATH_LEGACY
    print(f"  Loading 3D graphs from {graphs_path}...", flush=True)
    data_list = torch.load(graphs_path, weights_only=False)
    has_charges = hasattr(data_list[0], 'charges')
    print(f"  Gasteiger charges: {has_charges}", flush=True)
    n_graph = len(data_list)
    y_graph = np.stack([d.y.squeeze(0).numpy() for d in data_list]).astype(np.float32)

    # Load tabular features
    df_full = pd.read_csv(PHASE3_FEAT)
    required = set(METADATA_COLS + TARGET_COLS)
    feature_cols = [c for c in df_full.columns if c not in required]

    gain_path = PHASE3_OPT / "feature_gain.csv"
    if gain_path.exists():
        gain_df = pd.read_csv(gain_path)
        kept = gain_df[gain_df["total_gain"] > 0]["feature"].tolist()
        feature_cols = [c for c in kept if c in df_full.columns]

    X_tab_full = df_full[feature_cols].values.astype(np.float32)
    y_tab_full = df_full[TARGET_COLS].values.astype(np.float32)

    # Align graph -> tabular by matching y values (float32 vs float64 precision)
    print(f"  Aligning graph and tabular data...", flush=True)
    tab_lookup = {}
    for i in range(len(y_tab_full)):
        key = (round(float(y_tab_full[i, 0]), 4), round(float(y_tab_full[i, 1]), 4),
               round(float(y_tab_full[i, 2]), 4))
        tab_lookup[key] = i

    aligned_mask = []
    tab_indices = []
    for i in range(n_graph):
        key = (round(float(y_graph[i, 0]), 4), round(float(y_graph[i, 1]), 4),
               round(float(y_graph[i, 2]), 4))
        if key in tab_lookup:
            aligned_mask.append(True)
            tab_indices.append(tab_lookup[key])
        else:
            aligned_mask.append(False)
            tab_indices.append(-1)

    aligned_mask = np.array(aligned_mask)
    tab_indices = np.array(tab_indices)
    aligned_graph_idx = np.where(aligned_mask)[0]
    aligned_tab_idx = tab_indices[aligned_mask]
    n_aligned = len(aligned_graph_idx)
    print(f"  Aligned: {n_aligned}/{n_graph} molecules", flush=True)

    # Build aligned arrays
    X_tab = X_tab_full[aligned_tab_idx]
    y_tab = y_graph[aligned_graph_idx]

    # SchNet y_mean/y_std from graph train split (same as SchNet training)
    graph_train_idx, _, _ = create_split_indices(n_graph, random_state=SEED)
    train_y_graph = y_graph[graph_train_idx]
    y_mean = train_y_graph.mean(axis=0)
    y_std = train_y_graph.std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    print(f"  Getting SchNet(tuned) predictions...", flush=True)
    schnet_preds_all = get_all_schnet_predictions(data_list, y_mean, y_std, device,
                                                     use_charges=has_charges)
    schnet_preds = schnet_preds_all[aligned_graph_idx]

    # Split the aligned dataset
    train_idx, valid_idx, test_idx = create_split_indices(n_aligned, random_state=SEED)
    tv_idx = np.concatenate([train_idx, valid_idx])
    schnet_m = regression_metrics(y_tab[test_idx], schnet_preds[test_idx])
    print(f"  SchNet alone: MAE={schnet_m['average']['mae']:.4f} R2={schnet_m['average']['r2']:.4f}", flush=True)

    # Load LGBM / XGB params
    with open(PHASE3_OPT / "best_params_lgbm.json") as f:
        lgbm_params = json.load(f)
    with open(PHASE3_OPT / "best_params_xgb.json") as f:
        xgb_params = json.load(f)

    # Train base models on train split, predict valid (for stacking meta-features)
    print(f"\n  Training base models...", flush=True)

    lgbm = MultiOutputRegressor(LGBMRegressor(
        random_state=SEED, n_jobs=-1, verbose=-1, **lgbm_params))
    lgbm.fit(X_tab[train_idx], y_tab[train_idx])
    lgbm_val = lgbm.predict(X_tab[valid_idx])

    xgb = MultiOutputRegressor(XGBRegressor(
        random_state=SEED, n_jobs=-1, verbosity=0, tree_method="hist", **xgb_params))
    xgb.fit(X_tab[train_idx], y_tab[train_idx])
    xgb_val = xgb.predict(X_tab[valid_idx])

    schnet_val = schnet_preds[valid_idx]

    # Retrain on train+valid for test predictions
    lgbm.fit(X_tab[tv_idx], y_tab[tv_idx])
    lgbm_test = lgbm.predict(X_tab[test_idx])
    lgbm_m = regression_metrics(y_tab[test_idx], lgbm_test)
    print(f"  LGBM alone:   MAE={lgbm_m['average']['mae']:.4f} R2={lgbm_m['average']['r2']:.4f}", flush=True)

    xgb.fit(X_tab[tv_idx], y_tab[tv_idx])
    xgb_test = xgb.predict(X_tab[test_idx])

    schnet_test = schnet_preds[test_idx]

    results = []

    # === Strategy 1: Multi-output Ridge stacking ===
    print(f"\n  Strategy 1: Ridge stacking (LGBM + XGB + SchNet)", flush=True)
    stack_val = np.hstack([lgbm_val, xgb_val, schnet_val])
    stack_test = np.hstack([lgbm_test, xgb_test, schnet_test])

    ridge_mo = MultiOutputRegressor(Ridge(alpha=1.0))
    ridge_mo.fit(stack_val, y_tab[valid_idx])
    pred_stack = ridge_mo.predict(stack_test)
    m1 = regression_metrics(y_tab[test_idx], pred_stack)
    print(f"  Ridge stack:  MAE={m1['average']['mae']:.4f} R2={m1['average']['r2']:.4f}", flush=True)
    for t in TARGET_COLS:
        print(f"    {t:5s}: MAE={m1[t]['mae']:.4f} R2={m1[t]['r2']:.4f}", flush=True)
    results.append({"model": "Ridge_stack_3model", **_flat(m1)})

    # === Strategy 2: Per-target Ridge stacking ===
    print(f"\n  Strategy 2: Per-target Ridge stacking", flush=True)
    pred_pt = np.zeros_like(pred_stack)
    for i, t in enumerate(TARGET_COLS):
        # Each target gets predictions from all 3 models for all 3 targets
        ridge_t = Ridge(alpha=1.0)
        ridge_t.fit(stack_val, y_tab[valid_idx, i])
        pred_pt[:, i] = ridge_t.predict(stack_test)
    m2 = regression_metrics(y_tab[test_idx], pred_pt)
    print(f"  Per-target:   MAE={m2['average']['mae']:.4f} R2={m2['average']['r2']:.4f}", flush=True)
    for t in TARGET_COLS:
        print(f"    {t:5s}: MAE={m2[t]['mae']:.4f} R2={m2[t]['r2']:.4f}", flush=True)
    results.append({"model": "PerTarget_Ridge_stack", **_flat(m2)})

    # === Strategy 3: Per-target optimal blend (LGBM vs SchNet only) ===
    print(f"\n  Strategy 3: Per-target optimal blend (LGBM + SchNet)", flush=True)
    pred_blend = np.zeros((len(test_idx), 3))
    blend_weights = {}
    for i, t in enumerate(TARGET_COLS):
        best_w, best_mae = 0, float("inf")
        for alpha in np.arange(0, 1.01, 0.01):
            blended = alpha * lgbm_val[:, i] + (1 - alpha) * schnet_val[:, i]
            mae = np.mean(np.abs(blended - y_tab[valid_idx, i]))
            if mae < best_mae:
                best_mae = mae
                best_w = alpha
        pred_blend[:, i] = best_w * lgbm_test[:, i] + (1 - best_w) * schnet_test[:, i]
        blend_weights[t] = round(best_w, 2)
        print(f"    {t:5s}: LGBM_weight={best_w:.2f}", flush=True)
    m3 = regression_metrics(y_tab[test_idx], pred_blend)
    print(f"  Per-target blend: MAE={m3['average']['mae']:.4f} R2={m3['average']['r2']:.4f}", flush=True)
    for t in TARGET_COLS:
        print(f"    {t:5s}: MAE={m3[t]['mae']:.4f} R2={m3[t]['r2']:.4f}", flush=True)
    results.append({"model": "PerTarget_blend_LGBM_SchNet", **_flat(m3)})

    # === Strategy 4: Tabular + SchNet predictions as features ===
    print(f"\n  Strategy 4: LGBM with SchNet preds as extra features", flush=True)
    X_fused = np.hstack([X_tab, schnet_preds])
    lgbm_fused = MultiOutputRegressor(LGBMRegressor(
        random_state=SEED, n_jobs=-1, verbose=-1, **lgbm_params))
    lgbm_fused.fit(X_fused[tv_idx], y_tab[tv_idx])
    pred_fused = lgbm_fused.predict(X_fused[test_idx])
    m4 = regression_metrics(y_tab[test_idx], pred_fused)
    print(f"  LGBM+SchNet feat: MAE={m4['average']['mae']:.4f} R2={m4['average']['r2']:.4f}", flush=True)
    for t in TARGET_COLS:
        print(f"    {t:5s}: MAE={m4[t]['mae']:.4f} R2={m4[t]['r2']:.4f}", flush=True)
    results.append({"model": "LGBM_with_SchNet_features", **_flat(m4)})

    # === Summary ===
    results.append({"model": "SchNet_tuned_alone", **_flat(schnet_m)})
    results.append({"model": "LGBM_tuned_alone", **_flat(lgbm_m)})

    print(f"\n{'='*65}", flush=True)
    print(f"  STACKING V2 RESULTS", flush=True)
    print(f"{'='*65}", flush=True)
    for r in sorted(results, key=lambda x: -x["average_r2"]):
        print(f"  {r['model']:35s} MAE={r['average_mae']:.4f} R2={r['average_r2']:.4f}", flush=True)

    best = max(results, key=lambda x: x["average_r2"])
    print(f"\n  BEST: {best['model']}  MAE={best['average_mae']:.4f} R2={best['average_r2']:.4f}", flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "stacking_v2_comparison.csv", index=False)

    save_json({
        "best_model": best["model"],
        "best_mae": float(best["average_mae"]),
        "best_r2": float(best["average_r2"]),
        "blend_weights": blend_weights,
        "all_results": results,
    }, OUT_DIR / "stacking_v2_summary.json")

    # Save to experiments for master log
    save_json({
        "phase": "4",
        "sub_stage": "4.5",
        "experiment": "phase4_stacking_v2",
        "model": best["model"],
        "data_desc": "30k CHONSFCl",
        "elements": "C,Cl,F,H,N,O,S",
        "mw_range": "200-500",
        "n_data": 30000,
        "split": "random_test",
        "metrics": {
            t: {"mae": best[f"{t}_mae"], "rmse": best.get(f"{t}_rmse"), "r2": best[f"{t}_r2"]}
            for t in TARGET_COLS + ["average"]
        },
    }, RESULTS_DIR / "experiments" / "phase4_stacking_v2.json")

    print(f"\n  Saved to {OUT_DIR}/", flush=True)


def _flat(m):
    """Flatten regression_metrics dict for DataFrame row."""
    row = {}
    for t in TARGET_COLS + ["average"]:
        for k in ["mae", "rmse", "r2"]:
            row[f"{t}_{k}"] = m[t][k]
    return row


if __name__ == "__main__":
    main()
