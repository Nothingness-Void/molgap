"""
Resume SchNet 300k full retrain from Colab checkpoint on local RTX 5060.

Usage:
  .venv\Scripts\python.exe scripts/phase7/resume_schnet_300k.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

from molgap.constants import RESULTS_DIR, MODELS_DIR, SEED
from molgap.schnet import SchNetWrapper

PHASE7_DIR = RESULTS_DIR / "phase7"
GRAPH_PATH = PHASE7_DIR / "pyg_3d_graphs_etkdg_300k.pt"
CKPT_PATH = PHASE7_DIR / "checkpoint.pt"
BEST_PATH = PHASE7_DIR / "best_model.pt"

BEST_PARAMS = {
    "hidden_channels": 192,
    "num_filters": 192,
    "num_interactions": 6,
    "num_gaussians": 50,
    "cutoff": 6.0,
    "dropout": 0.0,
    "lr": 0.00021150972021685588,
    "weight_decay": 1.4656553886225336e-05,
    "batch_size": 128,
    "scheduler": "cosine",
}

FULL_EPOCHS = 500
FULL_PATIENCE = 40


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"{props.name} | {props.total_memory / 1e9:.1f} GB")

    print(f"Loading graphs from {GRAPH_PATH} ...")
    graphs = torch.load(str(GRAPH_PATH), weights_only=False)
    print(f"Loaded {len(graphs)} graphs")

    N = len(graphs)
    idx = np.random.RandomState(SEED).permutation(N)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    train_set = [graphs[i] for i in idx[:n_train]]
    val_set = [graphs[i] for i in idx[n_train:n_train + n_val]]
    test_set = [graphs[i] for i in idx[n_train + n_val:]]
    print(f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

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

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=p["lr"],
                                  weight_decay=p["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FULL_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler()
    criterion = nn.L1Loss()

    # Resume from checkpoint
    ckpt = torch.load(str(CKPT_PATH), weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    sched.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_val = ckpt.get("best_val_mae", ckpt.get("best_val", float("inf")))
    print(f"Resumed from epoch {start_epoch}, best_val={best_val:.4f}")

    bs = p["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False,
                            num_workers=0)

    wait = 0
    best_epoch = start_epoch

    for epoch in range(start_epoch, FULL_EPOCHS):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                pred = model(batch_data.z, batch_data.pos, batch_data.batch,
                             charges=batch_data.charges)
                loss = criterion(pred, batch_data.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * batch_data.num_graphs
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                with torch.amp.autocast("cuda"):
                    pred = model(batch_data.z, batch_data.pos, batch_data.batch,
                             charges=batch_data.charges)
                    loss = criterion(pred, batch_data.y)
                val_loss += loss.item() * batch_data.num_graphs
        val_loss /= len(val_set)

        sched.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), str(PHASE7_DIR / "schnet_300k_best.pt"))
            improved = " *"
        else:
            wait += 1

        print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | val_MAE={val_loss:.4f} | "
              f"best={best_val:.4f}@{best_epoch} | lr={lr_now:.2e} | {elapsed:.1f}s{improved}")

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val,
            }, str(PHASE7_DIR / "checkpoint.pt"))

        if wait >= FULL_PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    # Test evaluation
    print("\n=== Test Evaluation ===")
    best_state = torch.load(str(PHASE7_DIR / "schnet_300k_best.pt"),
                            weights_only=False, map_location=device)
    model.load_state_dict(best_state)
    model.eval()

    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            pred = model(batch_data.z, batch_data.pos, batch_data.batch)
            all_pred.append(pred.cpu().numpy())
            all_true.append(batch_data.y.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    target_names = ["HOMO", "LUMO", "Gap"]
    metrics = {"best_params": BEST_PARAMS, "best_val_mae": float(best_val),
               "best_epoch": best_epoch}
    for i, name in enumerate(target_names):
        mae = mean_absolute_error(all_true[:, i], all_pred[:, i])
        r2 = r2_score(all_true[:, i], all_pred[:, i])
        print(f"  {name}: MAE={mae:.4f} eV, R2={r2:.4f}")
        metrics[name] = {"mae": float(mae), "r2": float(r2)}

    metrics_path = PHASE7_DIR / "schnet_300k_metrics.json"
    with open(str(metrics_path), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save final model
    import shutil
    final_dst = MODELS_DIR / "gnn_schnet_3d_300k.pt"
    shutil.copy2(str(PHASE7_DIR / "schnet_300k_best.pt"), str(final_dst))
    print(f"Best model saved to {final_dst}")


if __name__ == "__main__":
    main()
