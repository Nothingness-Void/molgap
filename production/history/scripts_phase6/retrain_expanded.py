"""Phase 6: Merge MW 200-503 + MW 500-1000 data, build ETKDG graphs, retrain SchNet."""
import time, json
import numpy as np
import pandas as pd
import torch

from molgap.constants import (
    DATA_PHASE3, DATA_PHASE6_LARGE, RESULTS_DIR, MODELS_DIR, TARGET_COLS,
    SCRIPTS_DIR,
)
from molgap.utils import (
    create_split_indices, ensure_dirs, regression_metrics, save_json,
    canonicalize_smiles,
)
from molgap.schnet import SchNetWrapper

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PHASE6_DIR = RESULTS_DIR / "phase6"
GRAPH_PATH = PHASE6_DIR / "pyg_3d_graphs_etkdg_expanded.pt"
ensure_dirs(PHASE6_DIR, MODELS_DIR)

# ── Step 1: Merge data ──
print("=" * 60)
print("Step 1: Merging datasets")
print("=" * 60)

df_small = pd.read_csv(DATA_PHASE3)
df_large = pd.read_csv(DATA_PHASE6_LARGE)
print(f"  Small MW (200-503): {len(df_small)}")
print(f"  Large MW (500-1000): {len(df_large)}")

df_merged = pd.concat([df_small, df_large], ignore_index=True)

for col in ["homo", "lumo", "gap", "mw"]:
    df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce")
df_merged = df_merged.dropna(subset=["homo", "lumo", "gap", "smiles"])
df_merged = df_merged[df_merged["gap"] > 0]

df_merged["canonical_smiles"] = df_merged["smiles"].apply(canonicalize_smiles)
df_merged = df_merged.dropna(subset=["canonical_smiles"])
df_merged = df_merged.drop_duplicates(subset=["canonical_smiles"])
df_merged = df_merged.reset_index(drop=True)

print(f"  After dedup: {len(df_merged)}")
print(f"  MW range: {df_merged['mw'].min():.1f} - {df_merged['mw'].max():.1f}")
print(f"  MW median: {df_merged['mw'].median():.1f}")

# ── Step 2: Build ETKDG graphs ──
print(f"\n{'=' * 60}")
print("Step 2: Building ETKDG 3D graphs")
print("=" * 60)

if GRAPH_PATH.exists():
    data_list = torch.load(GRAPH_PATH, weights_only=False)
    if len(data_list) >= len(df_merged) * 0.9:
        print(f"  Reusing cached graphs: {len(data_list)}")
    else:
        data_list = None
else:
    data_list = None

if data_list is None:
    from molgap.graphs import build_labeled_graphs

    smiles_list = df_merged["canonical_smiles"].tolist()
    targets = df_merged[TARGET_COLS].values.astype(np.float32)

    t0 = time.time()
    data_list = build_labeled_graphs(smiles_list, targets, use_charges=True)
    elapsed = time.time() - t0
    print(f"  Built {len(data_list)} graphs in {elapsed:.0f}s")

    torch.save(data_list, GRAPH_PATH)
    print(f"  Saved to {GRAPH_PATH}")

# ── Step 3: Train/test split ──
print(f"\n{'=' * 60}")
print("Step 3: Retrain with Optuna best params")
print("=" * 60)

with open(RESULTS_DIR / "phase4" / "schnet_optuna" / "optuna_best_params.json") as f:
    params = json.load(f)
print(f"  Params: {params}")

train_idx, valid_idx, test_idx = create_split_indices(len(data_list), random_state=SEED)
train_data = [data_list[i] for i in train_idx]
valid_data = [data_list[i] for i in valid_idx]
test_data = [data_list[i] for i in test_idx]
print(f"  Split: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")

train_y = np.stack([d.y.squeeze(0).numpy() for d in train_data])
y_mean = train_y.mean(axis=0)
y_std = train_y.std(axis=0)
y_std[y_std < 1e-6] = 1.0
for d in train_data + valid_data + test_data:
    d.y = (d.y - torch.tensor(y_mean)) / torch.tensor(y_std)

has_charges = hasattr(data_list[0], 'charges')
del data_list

from torch_geometric.loader import DataLoader
bs = params["batch_size"]
train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=bs)
test_loader = DataLoader(test_data, batch_size=bs)

import sys
sys.path.insert(0, str(SCRIPTS_DIR / "phase4"))
from schnet_optuna import run_training, evaluate

print(f"\n  Full retrain: 500 epochs, patience=40, charges={has_charges}")
best_mae, best_epoch, best_state, log_rows = run_training(
    params, train_loader, valid_loader, y_mean, y_std,
    device, max_epochs=500, patience=40, verbose=True, use_charges=has_charges)

pd.DataFrame(log_rows).to_csv(PHASE6_DIR / "retrain_log.csv", index=False)

# ── Step 4: Evaluate ──
model = SchNetWrapper(
    hidden_channels=params["hidden_channels"], num_filters=params["num_filters"],
    num_interactions=params["num_interactions"], num_gaussians=params["num_gaussians"],
    cutoff=params["cutoff"], dropout=params["dropout"], use_charges=has_charges).to(device)
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

test_pred, test_true = evaluate(model, test_loader, device)
test_pred_real = test_pred * y_std + y_mean
test_true_real = test_true * y_std + y_mean
m = regression_metrics(test_true_real, test_pred_real)

print(f"\n{'=' * 60}")
print(f"  Phase 6: SchNet Expanded (MW 200-1000) Results")
print(f"{'=' * 60}")
for t in TARGET_COLS:
    print(f"  {t:5s}: MAE={m[t]['mae']:.4f}  RMSE={m[t]['rmse']:.4f}  R2={m[t]['r2']:.4f}")
print(f"  avg  : MAE={m['average']['mae']:.4f}  RMSE={m['average']['rmse']:.4f}  R2={m['average']['r2']:.4f}")

# Compare with Phase 4
prev_path = RESULTS_DIR / "phase4" / "schnet_optuna" / "schnet_tuned_metrics.json"
if prev_path.exists():
    with open(prev_path) as f:
        prev = json.load(f)
    prev_mae = prev["metrics"]["average"]["mae"]
    prev_r2 = prev["metrics"]["average"]["r2"]
    print(f"\n  vs Phase 4 (MW 200-503 only):")
    print(f"    Phase 4: MAE={prev_mae:.4f}  R2={prev_r2:.4f}")
    print(f"    Phase 6: MAE={m['average']['mae']:.4f}  R2={m['average']['r2']:.4f}")
    print(f"    Delta MAE: {m['average']['mae'] - prev_mae:+.4f}")
    print(f"    Delta R2:  {m['average']['r2'] - prev_r2:+.4f}")

n_total = len(train_data) + len(valid_data) + len(test_data)
torch.save(best_state, MODELS_DIR / "gnn_schnet_3d_expanded.pt")
save_json({
    "model": "SchNet_3D_ETKDG_expanded",
    "params": params,
    "n_data": n_total,
    "mw_range": "200-1000",
    "best_epoch": best_epoch,
    "epochs_trained": len(log_rows),
    "metrics": m,
}, PHASE6_DIR / "schnet_expanded_metrics.json")

save_json({
    "phase": "6", "experiment": "phase6_schnet_expanded",
    "model": "SchNet_3D_ETKDG_expanded",
    "data_desc": f"{n_total} CHONSFCl MW200-1000",
    "mw_range": "200-1000", "n_data": n_total,
    "split": "random_test", "metrics": m,
}, RESULTS_DIR / "experiments" / "phase6_schnet_expanded.json")

print(f"\nDone! Model saved to models/gnn_schnet_3d_expanded.pt")
