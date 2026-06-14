"""
Phase 7: GPS 2D training on local RTX 5060.
Optuna search (SQLite for resume) + full retrain + test eval + embedding extraction.

Usage:
  .venv\Scripts\python.exe scripts/phase7/train_gps_2d_local.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

from molgap.constants import RESULTS_DIR, SEED
from molgap.gps import GPSWrapper

PHASE7_DIR = RESULTS_DIR / "phase7"
GRAPH_PATH = PHASE7_DIR / "pyg_2d_graphs_bond_300k.pt"
FULL_EPOCHS = 150
FULL_PATIENCE = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(params, train_set, val_set, max_epochs=80,
                 patience=15, save_prefix="gps", verbose=True):
    model = GPSWrapper(
        hidden_channels=params["hidden_channels"],
        num_layers=params["num_layers"],
        num_heads=params["num_heads"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params["lr"],
                                  weight_decay=params["weight_decay"])
    if params["scheduler"] == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler()
    criterion = nn.L1Loss()

    ckpt_path = str(PHASE7_DIR / f"{save_prefix}_ckpt.pt")
    start_epoch = 0
    best_val = float("inf")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"  Resumed from epoch {start_epoch}, best_val={best_val:.4f}")

    bs = params["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False,
                            num_workers=0)

    wait = 0

    n_batches = (len(train_set) + bs - 1) // bs

    for epoch in range(start_epoch, max_epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for bi, batch_data in enumerate(train_loader):
            if bi % 50 == 0:
                print(f"    batch {bi}/{n_batches}", end="\r", flush=True)
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                pred = model(batch_data.x, batch_data.edge_index,
                             batch_data.edge_attr, batch_data.batch)
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
                    pred = model(batch_data.x, batch_data.edge_index,
                                 batch_data.edge_attr, batch_data.batch)
                    loss = criterion(pred, batch_data.y)
                val_loss += loss.item() * batch_data.num_graphs
        val_loss /= len(val_set)

        if params["scheduler"] == "plateau":
            sched.step(val_loss)
        else:
            sched.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        if verbose:
            print(f"  ep{epoch:03d} train={train_loss:.4f} val={val_loss:.4f} "
                  f"lr={lr_now:.2e} {elapsed:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(),
                       str(PHASE7_DIR / f"{save_prefix}_best.pt"))
        else:
            wait += 1

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val,
            }, ckpt_path)

        if wait >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    # Final checkpoint
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val": best_val,
    }, ckpt_path)

    return best_val, model


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"{props.name} | {props.total_memory / 1e9:.1f} GB")

    # ── Load data ──
    print(f"Loading graphs from {GRAPH_PATH} ...")
    graphs = torch.load(str(GRAPH_PATH), weights_only=False)
    print(f"Loaded {len(graphs)} 2D graphs")

    N = len(graphs)
    idx = np.random.RandomState(SEED).permutation(N)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    train_set = [graphs[i] for i in idx[:n_train]]
    val_set = [graphs[i] for i in idx[n_train:n_train + n_val]]
    test_set = [graphs[i] for i in idx[n_train + n_val:]]
    print(f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    # Best params from Kaggle Optuna (20 trials on 50k subset, MAE=0.1445)
    bp = {
        "hidden_channels": 192,
        "num_layers": 7,
        "num_heads": 4,
        "dropout": 0.05,
        "lr": 0.0004754654349367296,
        "weight_decay": 1.3094136884618282e-05,
        "batch_size": 256,
        "scheduler": "cosine",
    }
    print("\nUsing Kaggle Optuna best params:")
    print(bp)

    # ── Full retrain ──
    print(f"\n=== Full Retrain: {FULL_EPOCHS} epochs, patience={FULL_PATIENCE} ===")
    best_val, model = run_training(
        bp, train_set, val_set,
        max_epochs=FULL_EPOCHS, patience=FULL_PATIENCE,
        save_prefix="gps_2d_best", verbose=True)
    print(f"Best val MAE: {best_val:.4f}")

    # ── Test evaluation ──
    print("\n=== Test Evaluation ===")
    best_state = torch.load(str(PHASE7_DIR / "gps_2d_best_best.pt"), weights_only=False)
    model.load_state_dict(best_state)
    model.eval()

    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            pred = model(batch_data.x, batch_data.edge_index,
                         batch_data.edge_attr, batch_data.batch)
            all_pred.append(pred.cpu().numpy())
            all_true.append(batch_data.y.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    target_names = ["HOMO", "LUMO", "Gap"]
    metrics = {"best_params": bp, "best_val_mae": float(best_val)}
    for i, name in enumerate(target_names):
        mae = mean_absolute_error(all_true[:, i], all_pred[:, i])
        r2 = r2_score(all_true[:, i], all_pred[:, i])
        print(f"  {name}: MAE={mae:.4f} eV, R2={r2:.4f}")
        metrics[name] = {"mae": float(mae), "r2": float(r2)}

    with open(str(PHASE7_DIR / "gps_2d_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved")

    # ── Save model weights ──
    from molgap.constants import MODELS_DIR
    save_path = MODELS_DIR / "gps_2d_300k.pt"
    torch.save(model.state_dict(), str(save_path))
    print(f"\nModel saved to {save_path}")

    # ── Extract embeddings ──
    print("\n=== Extracting 2D Embeddings ===")
    all_loader = DataLoader(graphs, batch_size=256, shuffle=False, num_workers=2)
    embeddings = []
    with torch.no_grad():
        for batch_data in all_loader:
            batch_data = batch_data.to(device)
            emb = model.encode(batch_data.x, batch_data.edge_index,
                               batch_data.edge_attr, batch_data.batch)
            embeddings.append(emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    print(f"2D embeddings: {embeddings.shape}")
    torch.save(embeddings, str(PHASE7_DIR / "gps_2d_embeddings.pt"))
    print("Done!")


if __name__ == "__main__":
    main()
