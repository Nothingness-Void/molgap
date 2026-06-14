"""
Evaluate SchNet 300k best model on test set.

Usage:
  .venv\Scripts\python.exe scripts/phase7/eval_schnet_300k.py
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

from molgap.constants import RESULTS_DIR, MODELS_DIR, SEED
from molgap.schnet import SchNetWrapper

PHASE7_DIR = RESULTS_DIR / "phase7"
GRAPH_PATH = PHASE7_DIR / "pyg_3d_graphs_etkdg_300k.pt"
BEST_PATH = PHASE7_DIR / "schnet_300k_best.pt"

BEST_PARAMS = {
    "hidden_channels": 192,
    "num_filters": 192,
    "num_interactions": 6,
    "num_gaussians": 50,
    "cutoff": 6.0,
    "dropout": 0.0,
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading graphs from {GRAPH_PATH} ...")
    graphs = torch.load(str(GRAPH_PATH), weights_only=False)
    print(f"Loaded {len(graphs)} graphs")

    N = len(graphs)
    idx = np.random.RandomState(SEED).permutation(N)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    test_set = [graphs[i] for i in idx[n_train + n_val:]]
    print(f"Test set: {len(test_set)}")

    p = BEST_PARAMS
    model = SchNetWrapper(
        hidden_channels=p["hidden_channels"],
        num_filters=p["num_filters"],
        num_interactions=p["num_interactions"],
        num_gaussians=p["num_gaussians"],
        cutoff=p["cutoff"],
        dropout=p["dropout"],
        use_charges=True,
    ).to(device)

    state = torch.load(str(BEST_PATH), weights_only=False, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Loaded best model from epoch 80")

    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            with torch.amp.autocast("cuda"):
                pred = model(batch_data.z, batch_data.pos, batch_data.batch,
                             charges=batch_data.charges)
            all_pred.append(pred.cpu().numpy())
            all_true.append(batch_data.y.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    target_names = ["HOMO", "LUMO", "Gap"]
    metrics = {"best_val_mae": 0.1050, "best_epoch": 80, "best_params": BEST_PARAMS}

    print("\n=== Test Results ===")
    for i, name in enumerate(target_names):
        mae = mean_absolute_error(all_true[:, i], all_pred[:, i])
        r2 = r2_score(all_true[:, i], all_pred[:, i])
        print(f"  {name}: MAE={mae:.4f} eV, R2={r2:.4f}")
        metrics[name] = {"mae": float(mae), "r2": float(r2)}

    avg_mae = np.mean([metrics[n]["mae"] for n in target_names])
    avg_r2 = np.mean([metrics[n]["r2"] for n in target_names])
    print(f"  Avg:  MAE={avg_mae:.4f} eV, R2={avg_r2:.4f}")
    metrics["avg"] = {"mae": float(avg_mae), "r2": float(avg_r2)}

    metrics_path = PHASE7_DIR / "schnet_300k_metrics.json"
    with open(str(metrics_path), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    final_dst = MODELS_DIR / "gnn_schnet_3d_300k.pt"
    shutil.copy2(str(BEST_PATH), str(final_dst))
    print(f"Best model copied to {final_dst}")


if __name__ == "__main__":
    main()
