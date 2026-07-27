"""A/B step 2: train the shared 2D GPS encoder on the 10k subset.

The 2D modality is identical across all three arms, so we train it once (same
10k, same scaffold split) and reuse its embeddings. Trained on the train split
only; embeddings are then extracted for ALL 10k molecules (row-matched to the
3D graphs) for the fusion step.

Outputs (results/ab3d/):
  emb_2d.pt            [n, 192] float32, row-matched to graphs_2d/graphs_3d
  encoder_2d_gps.json  standalone GPS test MAE/R² (reference)

Usage:
  .venv\\Scripts\\python.exe scripts/ab3d/train_2d.py
"""
from __future__ import annotations

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import json

import numpy as np
import torch
import torch.nn as nn

from molgap.constants import PARAMS_GPS_2D, RESULTS_DIR, SEED
from molgap.gps import GPSWrapper
from molgap.utils import regression_metrics

OUT = RESULTS_DIR / "ab3d"
MAX_EPOCHS = 150
PATIENCE = 25
BATCH = 256
LR = 4.75e-4
WD = 1.3e-5


def main():
    from torch_geometric.loader import DataLoader

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    graphs = torch.load(str(OUT / "graphs_2d.pt"), weights_only=False)
    split = json.loads((OUT / "split.json").read_text(encoding="utf-8"))
    tr = [graphs[i] for i in split["train"]]
    va = [graphs[i] for i in split["val"]]
    te = [graphs[i] for i in split["test"]]
    print(f"2D graphs: {len(graphs)} | train/val/test {len(tr)}/{len(va)}/{len(te)}")

    model = GPSWrapper(**PARAMS_GPS_2D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-6)
    crit = nn.L1Loss()

    tl = DataLoader(tr, batch_size=BATCH, shuffle=True)
    vl = DataLoader(va, batch_size=BATCH)

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for b in tl:
            b = b.to(device)
            opt.zero_grad()
            loss = crit(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()

        model.eval()
        vsum, vn = 0.0, 0
        with torch.no_grad():
            for b in vl:
                b = b.to(device)
                vsum += crit(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y).item() * b.num_graphs
                vn += b.num_graphs
        vmae = vsum / vn
        if vmae < best_val - 1e-5:
            best_val, best_state, wait = vmae, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if epoch % 10 == 0 or wait == 0:
            print(f"  epoch {epoch:3d} val_MAE {vmae:.4f} (best {best_val:.4f})")
        if wait >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    # Test metrics (standalone GPS, reference).
    P, T = [], []
    with torch.no_grad():
        for b in DataLoader(te, batch_size=BATCH):
            b = b.to(device)
            P.append(model(b.x, b.edge_index, b.edge_attr, b.batch).cpu().numpy())
            T.append(b.y.cpu().numpy())
    metrics = regression_metrics(np.concatenate(T), np.concatenate(P),
                                 targets=["HOMO", "LUMO", "Gap"])
    (OUT / "encoder_2d_gps.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"GPS standalone test: Gap MAE {metrics['Gap']['mae']:.4f} R² {metrics['Gap']['r2']:.4f}")

    # Extract embeddings for ALL 10k (row order = graphs order).
    emb = []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=512):
            b = b.to(device)
            emb.append(model.encode(b.x, b.edge_index, b.edge_attr, b.batch).cpu())
    emb = torch.cat(emb)
    torch.save(emb, str(OUT / "emb_2d.pt"))
    print(f"[OK] Saved emb_2d {tuple(emb.shape)} to {OUT/'emb_2d.pt'}")


if __name__ == "__main__":
    main()
