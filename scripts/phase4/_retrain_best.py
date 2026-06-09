"""Retrain SchNet with saved Optuna best params (ETKDG, skip search phase)."""
import sys, time, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"D:\文档\molgap\src")
from pathlib import Path
from molgap.utils import RESULTS_DIR, MODELS_DIR, TARGET_COLS, create_split_indices, ensure_dirs, regression_metrics, save_json
from molgap.schnet import SchNetWrapper

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda")
OUT_DIR = RESULTS_DIR / "phase4" / "schnet_optuna"
ensure_dirs(OUT_DIR, MODELS_DIR)

with open(OUT_DIR / "optuna_best_params.json") as f:
    params = json.load(f)
print(f"Params: {params}", flush=True)

graph_path = RESULTS_DIR / "phase4" / "pyg_3d_graphs_etkdg.pt"
data_list = torch.load(graph_path, weights_only=False)
print(f"Loaded {len(data_list)} graphs", flush=True)

train_idx, valid_idx, test_idx = create_split_indices(len(data_list), random_state=SEED)
train_data = [data_list[i] for i in train_idx]
valid_data = [data_list[i] for i in valid_idx]
test_data = [data_list[i] for i in test_idx]

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

sys.path.insert(0, str(Path(r"D:\文档\molgap\scripts\phase4")))
from schnet_optuna import run_training, evaluate

print(f"\nFull retrain: 500 epochs, patience=40, charges={has_charges}", flush=True)
best_mae, best_epoch, best_state, log_rows = run_training(
    params, train_loader, valid_loader, y_mean, y_std,
    device, max_epochs=500, patience=40, verbose=True, use_charges=has_charges)

pd.DataFrame(log_rows).to_csv(OUT_DIR / "retrain_log.csv", index=False)

model = SchNetWrapper(
    hidden_channels=params["hidden_channels"], num_filters=params["num_filters"],
    num_interactions=params["num_interactions"], num_gaussians=params["num_gaussians"],
    cutoff=params["cutoff"], dropout=params["dropout"], use_charges=has_charges).to(device)
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

test_pred, test_true = evaluate(model, test_loader, device)
test_pred_real = test_pred * y_std + y_mean
test_true_real = test_true * y_std + y_mean
m = regression_metrics(test_true_real, test_pred_real)

print(f"\n{'='*60}", flush=True)
print(f"  SchNet ETKDG Optuna-tuned Test Results", flush=True)
print(f"{'='*60}", flush=True)
for t in TARGET_COLS:
    print(f"  {t:5s}: MAE={m[t]['mae']:.4f}  RMSE={m[t]['rmse']:.4f}  R2={m[t]['r2']:.4f}", flush=True)
print(f"  avg  : MAE={m['average']['mae']:.4f}  RMSE={m['average']['rmse']:.4f}  R2={m['average']['r2']:.4f}", flush=True)

prev_path = RESULTS_DIR / "phase4" / "schnet_metrics.json"
if prev_path.exists():
    with open(prev_path) as f:
        prev = json.load(f)
    prev_mae = prev["metrics"]["average"]["mae"]
    prev_r2 = prev["metrics"]["average"]["r2"]
    print(f"\n  vs ETKDG baseline: MAE={prev_mae:.4f} R2={prev_r2:.4f}", flush=True)
    print(f"  MAE improvement: {prev_mae - m['average']['mae']:.4f}", flush=True)
    print(f"  R2  improvement: {m['average']['r2'] - prev_r2:.4f}", flush=True)

torch.save(best_state, MODELS_DIR / "gnn_schnet_3d_tuned.pt")
save_json({"model": "SchNet_3D_ETKDG_optuna", "params": params,
    "n_params": sum(p.numel() for p in model.parameters()),
    "best_epoch": best_epoch, "epochs_trained": len(log_rows), "metrics": m},
    OUT_DIR / "schnet_tuned_metrics.json")
save_json({"phase": "4", "sub_stage": "4.4", "experiment": "phase4_schnet_optuna_etkdg",
    "model": "SchNet_3D_ETKDG_optuna", "data_desc": "30k CHONSFCl MW200-1000",
    "mw_range": "200-1000", "n_data": 30000, "split": "random_test", "metrics": m},
    RESULTS_DIR / "experiments" / "phase4_schnet_optuna.json")
print(f"\nDone! Model saved.", flush=True)
