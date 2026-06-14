"""
Phase 7 Fusion: combine pre-trained GPS 2D + ViSNet 3D embeddings locally.
Trains only the fusion layer + MLP head on RTX 5060.

Usage:
  .venv\Scripts\python.exe scripts/phase7/fusion_local.py

Requires:
  - results/phase7/schnet_3d_embeddings.pt          (extract_schnet_3d_embeddings.py)
  - results/phase7/gps_2d_embeddings_aligned.pt     (align_2d_to_3d.py)
  - results/phase7/pyg_3d_graphs_etkdg_300k.pt      (for labels)

Note: 2D embeddings are pre-aligned to the 3D molecule set (3D dropped 371
ETKDG failures), so row i of both files is the same molecule.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from molgap.constants import RESULTS_DIR, MODELS_DIR

PHASE7_DIR = RESULTS_DIR / "phase7"
SEED = 42


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pre-computed embeddings
    emb_3d = torch.load(PHASE7_DIR / "schnet_3d_embeddings.pt", weights_only=False)
    emb_2d = torch.load(PHASE7_DIR / "gps_2d_embeddings_aligned.pt", weights_only=False)
    print(f"3D embeddings: {emb_3d.shape}, 2D embeddings: {emb_2d.shape}")

    assert emb_3d.shape[0] == emb_2d.shape[0], "Embedding count mismatch!"

    # Load labels from 3D graphs
    graphs = torch.load(PHASE7_DIR / "pyg_3d_graphs_etkdg_300k.pt", weights_only=False)
    labels = torch.stack([g.y.squeeze(0) for g in graphs])
    print(f"Labels: {labels.shape}")
    del graphs

    N = emb_3d.shape[0]
    idx = np.random.RandomState(SEED).permutation(N)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)

    splits = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }

    def make_loader(split_name, batch_size=512, shuffle=False):
        ii = splits[split_name]
        ds = TensorDataset(emb_2d[ii], emb_3d[ii], labels[ii])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=True, num_workers=0)

    # Fusion model (lightweight)
    dim_2d = emb_2d.shape[1]
    dim_3d = emb_3d.shape[1]
    hidden = 128

    class FusionHead(nn.Module):
        def __init__(self, fusion_type="gate"):
            super().__init__()
            self.proj_2d = nn.Linear(dim_2d, hidden)
            self.proj_3d = nn.Linear(dim_3d, hidden)
            self.fusion_type = fusion_type

            if fusion_type == "gate":
                self.gate = nn.Sequential(
                    nn.Linear(hidden * 2, hidden),
                    nn.Sigmoid(),
                )
                head_in = hidden
            else:
                head_in = hidden * 2

            self.head = nn.Sequential(
                nn.Linear(head_in, hidden),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden // 2),
                nn.SiLU(),
                nn.Linear(hidden // 2, 3),
            )

        def forward(self, h_2d, h_3d):
            h_2d = self.proj_2d(h_2d)
            h_3d = self.proj_3d(h_3d)
            if self.fusion_type == "gate":
                g = self.gate(torch.cat([h_2d, h_3d], dim=-1))
                h = g * h_2d + (1 - g) * h_3d
            else:
                h = torch.cat([h_2d, h_3d], dim=-1)
            return self.head(h)

    # Train both fusion types and pick best
    results = {}
    for fusion_type in ["gate", "concat"]:
        print(f"\n=== Fusion: {fusion_type} ===")
        model = FusionHead(fusion_type).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-6)
        criterion = nn.L1Loss()

        train_loader = make_loader("train", batch_size=1024, shuffle=True)
        val_loader = make_loader("val", batch_size=2048)

        best_val = float("inf")
        wait = 0
        patience = 20

        for epoch in range(200):
            model.train()
            t0 = time.time()
            total_loss = 0.0
            count = 0
            for h2d, h3d, y in train_loader:
                h2d, h3d, y = h2d.to(device), h3d.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(h2d, h3d)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * y.size(0)
                count += y.size(0)
            train_mae = total_loss / count

            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for h2d, h3d, y in val_loader:
                    h2d, h3d, y = h2d.to(device), h3d.to(device), y.to(device)
                    pred = model(h2d, h3d)
                    val_loss += criterion(pred, y).item() * y.size(0)
                    val_count += y.size(0)
            val_mae = val_loss / val_count
            scheduler.step(val_mae)

            elapsed = time.time() - t0
            if epoch % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  ep{epoch:03d} train={train_mae:.4f} val={val_mae:.4f} "
                      f"lr={lr:.2e} {elapsed:.1f}s")

            if val_mae < best_val:
                best_val = val_mae
                wait = 0
                torch.save(model.state_dict(),
                           PHASE7_DIR / f"fusion_{fusion_type}_best.pt")
            else:
                wait += 1

            if wait >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

        # Test evaluation
        model.load_state_dict(
            torch.load(PHASE7_DIR / f"fusion_{fusion_type}_best.pt",
                       weights_only=False))
        model.eval()
        test_loader = make_loader("test", batch_size=2048)
        all_pred, all_true = [], []
        with torch.no_grad():
            for h2d, h3d, y in test_loader:
                h2d, h3d, y = h2d.to(device), h3d.to(device), y.to(device)
                pred = model(h2d, h3d)
                all_pred.append(pred.cpu().numpy())
                all_true.append(y.cpu().numpy())

        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)

        target_names = ["HOMO", "LUMO", "Gap"]
        metrics = {"fusion": fusion_type, "val_mae": float(best_val)}
        for i, name in enumerate(target_names):
            mae = mean_absolute_error(all_true[:, i], all_pred[:, i])
            r2 = r2_score(all_true[:, i], all_pred[:, i])
            print(f"  {name}: MAE={mae:.4f} eV, R2={r2:.4f}")
            metrics[name] = {"mae": float(mae), "r2": float(r2)}

        results[fusion_type] = metrics

    # Save best
    best_type = min(results, key=lambda k: results[k]["val_mae"])
    print(f"\n=== Best fusion: {best_type} ===")
    print(json.dumps(results[best_type], indent=2))

    with open(PHASE7_DIR / "hybrid_fusion_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Copy best model to models/
    import shutil
    best_src = PHASE7_DIR / f"fusion_{best_type}_best.pt"
    best_dst = MODELS_DIR / "hybrid_gps_visnet_fusion.pt"
    shutil.copy2(best_src, best_dst)
    print(f"Best model saved to {best_dst}")


if __name__ == "__main__":
    main()
