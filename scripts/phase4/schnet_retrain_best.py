"""
Phase 4: Retrain SchNet with Optuna-selected best params (Trial 3).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import (
    MODELS_DIR,
    RESULTS_DIR,
    TARGET_COLS,
    create_split_indices,
    ensure_dirs,
    regression_metrics,
    save_json,
)

OUT_DIR = RESULTS_DIR / "phase4" / "schnet_optuna"
PROCESSED_GRAPHS = RESULTS_DIR / "phase4" / "pyg_3d_graphs.pt"
SEED = 42

BEST_PARAMS = {
    "hidden_channels": 256,
    "num_filters": 192,
    "num_interactions": 7,
    "num_gaussians": 50,
    "cutoff": 7.0,
    "dropout": 0.0,
    "lr": 0.0011506408247250173,
    "weight_decay": 0.00013199942261535007,
    "batch_size": 64,
    "scheduler": "plateau",
}

EPOCHS = 300
PATIENCE = 35


class SchNetWrapper(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_interactions,
                 num_gaussians, cutoff, dropout=0.1, n_targets=3):
        super().__init__()
        from torch_geometric.nn.models import SchNet

        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels // 2, n_targets),
        )

    def forward(self, z, pos, batch):
        from torch_geometric.nn import global_mean_pool

        h = self.schnet.embedding(z)
        edge_index, edge_weight = self._radius_graph(pos, batch)
        edge_attr = self.schnet.distance_expansion(edge_weight)

        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = global_mean_pool(h, batch)
        return self.head(h)

    def _radius_graph(self, pos, batch):
        from torch_geometric.nn.models.schnet import radius_graph
        edge_index = radius_graph(pos, r=self.schnet.cutoff, batch=batch,
                                  max_num_neighbors=32)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(batch.z, batch.pos, batch.batch)
            loss = F.l1_loss(out, batch.y)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.amp.autocast("cuda"):
            out = model(batch.z, batch.pos, batch.batch)
        preds.append(out.cpu().numpy())
        trues.append(batch.y.cpu().numpy())
    return np.vstack(preds), np.vstack(trues)


def main():
    from torch_geometric.loader import DataLoader

    ensure_dirs(OUT_DIR, MODELS_DIR)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SchNet Retrain (Optuna Trial 3 best params) ===", flush=True)
    print(f"  Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Load graphs
    print(f"  Loading graphs...", flush=True)
    data_list = torch.load(PROCESSED_GRAPHS, weights_only=False)
    print(f"  Loaded {len(data_list)} graphs", flush=True)

    # Split
    train_idx, valid_idx, test_idx = create_split_indices(
        len(data_list), random_state=SEED)
    train_data = [data_list[i] for i in train_idx]
    valid_data = [data_list[i] for i in valid_idx]
    test_data = [data_list[i] for i in test_idx]

    # Standardize targets
    train_y = np.stack([d.y.squeeze(0).numpy() for d in train_data])
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    for d in train_data + valid_data + test_data:
        d.y = (d.y - torch.tensor(y_mean)) / torch.tensor(y_std)

    p = BEST_PARAMS
    train_loader = DataLoader(train_data, batch_size=p["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=p["batch_size"])
    test_loader = DataLoader(test_data, batch_size=p["batch_size"])

    model = SchNetWrapper(
        hidden_channels=p["hidden_channels"],
        num_filters=p["num_filters"],
        num_interactions=p["num_interactions"],
        num_gaussians=p["num_gaussians"],
        cutoff=p["cutoff"],
        dropout=p["dropout"],
    ).to(device)

    n_params = sum(par.numel() for par in model.parameters())
    print(f"  Params: {p}", flush=True)
    print(f"  Model params: {n_params:,}", flush=True)
    print(f"  Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}", flush=True)
    print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}\n", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=max(5, PATIENCE // 3), min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    best_val_mae = float("inf")
    best_epoch = 0
    log_rows = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        val_pred, val_true = evaluate(model, valid_loader, device)

        val_pred_real = val_pred * y_std + y_mean
        val_true_real = val_true * y_std + y_mean
        val_mae = float(np.mean(np.abs(val_pred_real - val_true_real)))

        scheduler.step(val_mae)
        elapsed = time.time() - t0

        log_rows.append({"epoch": epoch, "train_loss": train_loss,
                         "val_mae": val_mae, "lr": optimizer.param_groups[0]["lr"],
                         "time_s": elapsed})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), MODELS_DIR / "gnn_schnet_3d_tuned.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | val_MAE={val_mae:.4f} | "
                  f"best={best_val_mae:.4f}@{best_epoch} | lr={optimizer.param_groups[0]['lr']:.1e} | {elapsed:.1f}s",
                  flush=True)

        if epoch - best_epoch >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (best={best_epoch})", flush=True)
            break

    pd.DataFrame(log_rows).to_csv(OUT_DIR / "retrain_log.csv", index=False)

    # Evaluate on test
    model.load_state_dict(torch.load(MODELS_DIR / "gnn_schnet_3d_tuned.pt", weights_only=True))
    test_pred, test_true = evaluate(model, test_loader, device)
    test_pred_real = test_pred * y_std + y_mean
    test_true_real = test_true * y_std + y_mean

    m = regression_metrics(test_true_real, test_pred_real)

    print(f"\n{'='*60}", flush=True)
    print(f"  SchNet Tuned Test Results", flush=True)
    print(f"{'='*60}", flush=True)
    for t in TARGET_COLS:
        print(f"  {t:5s}: MAE={m[t]['mae']:.4f}  RMSE={m[t]['rmse']:.4f}  R2={m[t]['r2']:.4f}", flush=True)
    print(f"  avg  : MAE={m['average']['mae']:.4f}  RMSE={m['average']['rmse']:.4f}  R2={m['average']['r2']:.4f}", flush=True)

    # Compare with previous
    prev_path = RESULTS_DIR / "phase4" / "schnet_metrics.json"
    if prev_path.exists():
        import json
        with open(prev_path) as f:
            prev = json.load(f)
        prev_mae = prev["metrics"]["average"]["mae"]
        prev_r2 = prev["metrics"]["average"]["r2"]
        print(f"\n  vs previous SchNet: MAE={prev_mae:.4f} R2={prev_r2:.4f}", flush=True)
        print(f"  MAE change: {m['average']['mae'] - prev_mae:+.4f}", flush=True)
        print(f"  R2  change: {m['average']['r2'] - prev_r2:+.4f}", flush=True)

    save_json({
        "model": "SchNet_3D_optuna",
        "params": BEST_PARAMS,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "epochs_trained": len(log_rows),
        "metrics": m,
    }, OUT_DIR / "schnet_tuned_metrics.json")

    save_json(BEST_PARAMS, OUT_DIR / "optuna_best_params.json")

    save_json({
        "phase": "4",
        "sub_stage": "4.4",
        "experiment": "phase4_schnet_optuna",
        "model": "SchNet_3D_optuna",
        "data_desc": "30k CHONSFCl",
        "elements": "C,Cl,F,H,N,O,S",
        "mw_range": "200-500",
        "n_data": 30000,
        "split": "random_test",
        "metrics": m,
    }, RESULTS_DIR / "experiments" / "phase4_schnet_optuna.json")

    print(f"\n  Saved to {OUT_DIR}/", flush=True)
    print(f"  Model: {MODELS_DIR / 'gnn_schnet_3d_tuned.pt'}", flush=True)


if __name__ == "__main__":
    main()
