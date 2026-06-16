"""A/B step 4: train the FusionHead for one arm (shared 2D + this arm's 3D).

Fixed FusionHead config across all arms (gate, hidden 128) — no Optuna, for
speed and fairness. Same scaffold split as everything else. This is the
deployment-style "2D + <3D encoder>" confirmation; the standalone 3D metric from
train_encoder.py remains the primary discriminator.

Outputs (results/ab3d/):
  fusion_<name>.json    fusion test MAE/R²

Usage:
  .venv\\Scripts\\python.exe scripts/ab3d/train_fusion.py schnet
"""
from __future__ import annotations

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from molgap.constants import AB_ENCODERS, RESULTS_DIR, SEED
from molgap.fusion import FusionHead
from molgap.utils import regression_metrics

OUT = RESULTS_DIR / "ab3d"
MAX_EPOCHS = 200
PATIENCE = 30
BATCH = 1024
LR = 5e-4
WD = 1e-4
HIDDEN = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(AB_ENCODERS.keys()))
    args = ap.parse_args()
    name = args.name

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_2d = torch.load(str(OUT / "emb_2d.pt"), weights_only=False).float()
    emb_3d = torch.load(str(OUT / f"emb_3d_{name}.pt"), weights_only=False).float()
    labels = torch.load(str(OUT / "labels.pt"), weights_only=False).float()
    split = json.loads((OUT / "split.json").read_text(encoding="utf-8"))
    assert emb_2d.shape[0] == emb_3d.shape[0] == labels.shape[0], "row mismatch"
    print(f"[{name}] emb_2d {tuple(emb_2d.shape)} emb_3d {tuple(emb_3d.shape)}")

    def take(split_name):
        idx = torch.tensor(split[split_name], dtype=torch.long)
        return emb_2d[idx].to(device), emb_3d[idx].to(device), labels[idx].to(device)

    x2_tr, x3_tr, y_tr = take("train")
    x2_va, x3_va, y_va = take("val")
    x2_te, x3_te, y_te = take("test")

    model = FusionHead("gate", HIDDEN, dim_2d=emb_2d.shape[1], dim_3d=emb_3d.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1e-6)
    crit = nn.L1Loss()

    n_tr = y_tr.shape[0]
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for s in range(0, n_tr, BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            loss = crit(model(x2_tr[b], x3_tr[b]), y_tr[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vmae = crit(model(x2_va, x3_va), y_va).item()
        sched.step(vmae)
        if vmae < best_val - 1e-5:
            best_val, best_state, wait = vmae, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x2_te, x3_te).cpu().numpy()
    metrics = regression_metrics(y_te.cpu().numpy(), pred, targets=["HOMO", "LUMO", "Gap"])
    result = {"encoder": name, "fusion": "gate", "hidden": HIDDEN,
              "best_val_mae": round(best_val, 4), "test": metrics}
    (OUT / f"fusion_{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{name}] fusion test: Gap MAE {metrics['Gap']['mae']:.4f} R² {metrics['Gap']['r2']:.4f}")


if __name__ == "__main__":
    main()
