"""
Phase 7: formal OOD comparison — GPS 2D vs SchNet 3D vs Hybrid (gate) fusion
on the SAME 1000 unseen molecules. One unified table + per-model metrics.

Reads results/phase7/ood_1000/ood_molecules_1000.csv (from fetch_ood_1000.py).
For each molecule: build 2D + 3D graphs (keep only those with a valid 3D
ETKDG conformer, so all three models see the same set), then:
  - GPS 2D     -> predict + 192-d embedding
  - SchNet 3D  -> predict + 192-d embedding
  - Hybrid     -> gate-fusion of the two embeddings

Outputs:
  results/phase7/ood_1000/ood_comparison_3models.csv
  results/phase7/ood_1000/ood_comparison_3models_metrics.json

Usage:
  .venv\\Scripts\\python.exe scripts/phase7/compare_ood_1000.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from molgap.constants import MODELS_DIR, RESULTS_DIR, TARGET_COLS
from molgap.graphs import smiles_to_pyg, smiles_to_2d_pyg
from molgap.schnet import SchNetWrapper

from train_gps_2d_local import GPSWrapper  # same dir

OOD_DIR = RESULTS_DIR / "phase7" / "ood_1000"
OOD_CSV = OOD_DIR / "ood_molecules_1000.csv"

GPS_2D_MODEL = MODELS_DIR / "gps_2d_300k.pt"
SCHNET_MODEL = MODELS_DIR / "gnn_schnet_3d_300k.pt"
FUSION_MODEL = MODELS_DIR / "hybrid_gps_visnet_fusion.pt"
FUSION_METRICS = RESULTS_DIR / "phase7" / "hybrid_fusion_metrics.json"

GPS_PARAMS = {"hidden_channels": 192, "num_layers": 7, "num_heads": 4, "dropout": 0.05}
SCHNET_PARAMS = {"hidden_channels": 192, "num_filters": 192, "num_interactions": 6,
                 "num_gaussians": 50, "cutoff": 6.0, "dropout": 0.0}

DIM_2D = DIM_3D = 192
HIDDEN = 128


class FusionHead(nn.Module):
    """Must match fusion_local.py exactly (gate variant)."""

    def __init__(self, fusion_type="gate"):
        super().__init__()
        self.proj_2d = nn.Linear(DIM_2D, HIDDEN)
        self.proj_3d = nn.Linear(DIM_3D, HIDDEN)
        self.fusion_type = fusion_type
        if fusion_type == "gate":
            self.gate = nn.Sequential(nn.Linear(HIDDEN * 2, HIDDEN), nn.Sigmoid())
            head_in = HIDDEN
        else:
            head_in = HIDDEN * 2
        self.head = nn.Sequential(
            nn.Linear(head_in, HIDDEN), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.SiLU(),
            nn.Linear(HIDDEN // 2, 3),
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


def metrics_block(y_true, y_pred):
    out = {}
    for i, t in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        out[t] = {"mae": float(mae), "rmse": rmse, "r2": float(r2)}
    out["average"] = {
        "mae": float(np.mean([out[t]["mae"] for t in TARGET_COLS])),
        "rmse": float(np.mean([out[t]["rmse"] for t in TARGET_COLS])),
        "r2": float(np.mean([out[t]["r2"] for t in TARGET_COLS])),
    }
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(OOD_CSV)
    print(f"Loaded {len(df)} OOD molecules from {OOD_CSV}")

    # Build 2D + 3D graphs per molecule; keep only those with a valid 3D conformer
    print("\nBuilding 2D + 3D graphs (keep 3D-valid only)...")
    g2d_list, g3d_list, keep_rows = [], [], []
    for i, row in df.iterrows():
        g3d = smiles_to_pyg(row["smiles"])
        if g3d is None:
            continue
        g2d = smiles_to_2d_pyg(row["smiles"])
        if g2d is None:
            continue
        g3d_list.append(g3d)
        g2d_list.append(g2d)
        keep_rows.append(i)
    df = df.loc[keep_rows].reset_index(drop=True)
    print(f"  Valid (2D+3D) molecules: {len(df)}/{len(keep_rows)} kept")

    y_true = df[TARGET_COLS].values.astype(np.float32)

    # ── GPS 2D ──
    print("\nLoading GPS 2D...")
    gps = GPSWrapper(**GPS_PARAMS).to(device)
    gps.load_state_dict(torch.load(str(GPS_2D_MODEL), weights_only=False, map_location=device))
    gps.eval()

    from torch_geometric.loader import DataLoader
    pred_2d, emb_2d = [], []
    with torch.no_grad():
        for batch in DataLoader(g2d_list, batch_size=256):
            batch = batch.to(device)
            e = gps.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            p = gps.head(e)
            emb_2d.append(e.cpu()); pred_2d.append(p.cpu().numpy())
    pred_2d = np.concatenate(pred_2d)
    emb_2d = torch.cat(emb_2d)

    # ── SchNet 3D ──
    print("Loading SchNet 3D...")
    schnet = SchNetWrapper(**SCHNET_PARAMS, use_charges=True).to(device)
    schnet.load_state_dict(torch.load(str(SCHNET_MODEL), weights_only=False, map_location=device))
    schnet.eval()

    pred_3d, emb_3d = [], []
    with torch.no_grad():
        for batch in DataLoader(g3d_list, batch_size=128):
            batch = batch.to(device)
            charges = batch.charges if hasattr(batch, "charges") else None
            e = schnet.encode(batch.z, batch.pos, batch.batch, charges=charges)
            p = schnet.head(e)
            emb_3d.append(e.cpu()); pred_3d.append(p.cpu().numpy())
    pred_3d = np.concatenate(pred_3d)
    emb_3d = torch.cat(emb_3d)

    # ── Hybrid fusion ──
    best_type = "gate"
    if FUSION_METRICS.exists():
        with open(FUSION_METRICS) as f:
            fm = json.load(f)
        best_type = min(fm, key=lambda k: fm[k]["val_mae"])
    print(f"Loading Hybrid fusion ({best_type})...")
    fusion = FusionHead(best_type).to(device)
    fusion.load_state_dict(torch.load(str(FUSION_MODEL), weights_only=False, map_location=device))
    fusion.eval()
    with torch.no_grad():
        pred_hy = fusion(emb_2d.to(device), emb_3d.to(device)).cpu().numpy()

    # ── Metrics ──
    m2d = metrics_block(y_true, pred_2d)
    m3d = metrics_block(y_true, pred_3d)
    mhy = metrics_block(y_true, pred_hy)

    print(f"\n{'='*78}")
    print(f"  OOD Comparison ({len(df)} unseen molecules)")
    print(f"{'='*78}")
    print(f"  {'':5s}  {'GPS 2D':^16s}  {'SchNet 3D':^16s}  {'Hybrid':^16s}")
    print(f"  {'':5s}  {'MAE':>7s} {'R2':>7s}  {'MAE':>7s} {'R2':>7s}  {'MAE':>7s} {'R2':>7s}")
    for t in TARGET_COLS + ["average"]:
        print(f"  {t:5s}  {m2d[t]['mae']:7.4f} {m2d[t]['r2']:7.4f}  "
              f"{m3d[t]['mae']:7.4f} {m3d[t]['r2']:7.4f}  "
              f"{mhy[t]['mae']:7.4f} {mhy[t]['r2']:7.4f}")

    # ── Unified table ──
    for i, t in enumerate(TARGET_COLS):
        df[f"{t}_true"] = y_true[:, i]
        df[f"{t}_2d"] = pred_2d[:, i]
        df[f"{t}_3d"] = pred_3d[:, i]
        df[f"{t}_hybrid"] = pred_hy[:, i]
        df[f"{t}_err_2d"] = pred_2d[:, i] - y_true[:, i]
        df[f"{t}_err_3d"] = pred_3d[:, i] - y_true[:, i]
        df[f"{t}_err_hybrid"] = pred_hy[:, i] - y_true[:, i]

    df.to_csv(OOD_DIR / "ood_comparison_3models.csv", index=False, encoding="utf-8")
    with open(OOD_DIR / "ood_comparison_3models_metrics.json", "w") as f:
        json.dump({"n_molecules": len(df), "best_fusion": best_type,
                   "gps_2d": m2d, "schnet_3d": m3d, "hybrid": mhy}, f, indent=2)
    print(f"\n  Saved table + metrics to {OOD_DIR}/")


if __name__ == "__main__":
    main()
