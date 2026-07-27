"""
Train a MoE fusion head on Phase 7 frozen embeddings, then compare it on OOD-1000.

This is a head-only experiment:
  - GPS 2D and SchNet 3D encoders stay frozen.
  - Training uses the same Phase 7 ETKDG-aligned embeddings and random split.
  - OOD evaluation rebuilds 2D/3D graphs from SMILES, encodes with the same
    Phase 7 encoders, and compares the saved Phase 7 Hybrid vs the MoE head.

Usage:
  .venv\\Scripts\\python.exe scripts/phase7/train_moe_ood_compare.py
  .venv\\Scripts\\python.exe scripts/phase7/train_moe_ood_compare.py --skip-train
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

from molgap.constants import MODELS_DIR, RESULTS_DIR, TARGET_COLS
from molgap.fusion import MoEFusionHead
from molgap.graphs import smiles_to_2d_pyg, smiles_to_pyg
from molgap.inference import load_hybrid

PHASE7_DIR = RESULTS_DIR / "phase7"
OUT_DIR = PHASE7_DIR / "moe_experiment"
OOD_CSV = PHASE7_DIR / "ood_1000" / "ood_molecules_1000.csv"
DEFAULT_CKPT = MODELS_DIR / "hybrid_fusion_moe_e4.pt"
DEFAULT_RESULT = OUT_DIR / "ood_moe_e4_metrics.json"
DEFAULT_PRED = OUT_DIR / "ood_moe_e4_predictions.csv"


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_phase7_training_data(max_samples: int | None = None):
    emb_3d = torch.load(PHASE7_DIR / "schnet_3d_embeddings.pt", weights_only=False)
    emb_2d = torch.load(PHASE7_DIR / "gps_2d_embeddings_aligned.pt", weights_only=False)
    graphs = torch.load(PHASE7_DIR / "pyg_3d_graphs_etkdg_300k.pt", weights_only=False)
    labels = torch.stack([g.y.squeeze(0) for g in graphs])
    del graphs

    n = emb_3d.shape[0]
    assert emb_2d.shape[0] == n == labels.shape[0]
    idx = np.random.RandomState(42).permutation(n)
    if max_samples is not None:
        idx = idx[:max_samples]
    n_tr, n_va = int(0.8 * len(idx)), int(0.1 * len(idx))
    split = {
        "train": idx[:n_tr],
        "val": idx[n_tr:n_tr + n_va],
        "test": idx[n_tr + n_va:],
    }
    return emb_2d, emb_3d, labels, split


def make_loader(emb_2d, emb_3d, labels, indices, batch_size, shuffle):
    ds = TensorDataset(emb_2d[indices], emb_3d[indices], labels[indices])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for i, target in enumerate(TARGET_COLS):
        out[target] = {
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
        }
    out["average"] = {
        "mae": float(np.mean([out[t]["mae"] for t in TARGET_COLS])),
        "r2": float(np.mean([out[t]["r2"] for t in TARGET_COLS])),
    }
    return out


def train_moe(args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_phase7_training_data(args.max_samples)
    emb_2d, emb_3d, labels, split = data
    print(
        f"Training data: N={emb_2d.shape[0]} "
        f"train/val/test={len(split['train'])}/{len(split['val'])}/{len(split['test'])}",
        flush=True,
    )

    model = MoEFusionHead(
        hidden=args.hidden,
        dropout=args.dropout,
        n_experts=args.experts,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=8, factor=0.5, min_lr=1e-6
    )
    crit = nn.L1Loss()
    train_loader = make_loader(emb_2d, emb_3d, labels, split["train"], args.batch_size, True)
    val_loader = make_loader(emb_2d, emb_3d, labels, split["val"], 2048, False)

    best_val = float("inf")
    best_state = None
    wait = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for h2, h3, y in train_loader:
            h2, h3, y = h2.to(device), h3.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(h2, h3), y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_count += y.size(0)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for h2, h3, y in val_loader:
                h2, h3, y = h2.to(device), h3.to(device), y.to(device)
                val_loss += crit(model(h2, h3), y).item() * y.size(0)
                val_count += y.size(0)
        train_mae = train_loss / train_count
        val_mae = val_loss / val_count
        sched.step(val_mae)

        improved = val_mae < best_val
        if improved:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % args.log_every == 0 or improved:
            mark = "*" if improved else " "
            print(
                f"{mark} epoch {epoch:03d} train_mae={train_mae:.5f} "
                f"val_mae={val_mae:.5f} best={best_val:.5f} wait={wait}",
                flush=True,
            )
        if wait >= args.patience:
            print(f"Early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No best MoE state captured")
    model.load_state_dict(best_state)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "config": {
                "hidden": args.hidden,
                "dropout": args.dropout,
                "n_experts": args.experts,
                "seed": args.seed,
                "max_samples": args.max_samples,
            },
            "best_val_mae": best_val,
            "train_time_s": time.time() - t0,
        },
        args.checkpoint,
    )
    print(f"Saved MoE checkpoint: {args.checkpoint}", flush=True)
    return {"best_val_mae": float(best_val), "train_time_s": float(time.time() - t0)}


def load_moe(checkpoint, device: torch.device) -> MoEFusionHead:
    ckpt = torch.load(checkpoint, weights_only=False, map_location=device)
    cfg = ckpt.get("config", {})
    model = MoEFusionHead(
        hidden=cfg.get("hidden", 192),
        dropout=cfg.get("dropout", 0.0),
        n_experts=cfg.get("n_experts", 4),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict_ood(device: torch.device, moe: MoEFusionHead):
    ood = pd.read_csv(OOD_CSV)
    gps, schnet, fusion, _ = load_hybrid(device, key="phase7_hybrid")

    g2d_list, g3d_list, valid_idx = [], [], []
    for i, smi in enumerate(ood["smiles"].tolist()):
        g3d = smiles_to_pyg(smi)
        if g3d is None:
            continue
        g2d = smiles_to_2d_pyg(smi)
        if g2d is None:
            continue
        g2d_list.append(g2d)
        g3d_list.append(g3d)
        valid_idx.append(i)

    emb_2d = []
    pred_2d = []
    with torch.no_grad():
        for batch in GeometricDataLoader(g2d_list, batch_size=256):
            batch = batch.to(device)
            e = gps.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            emb_2d.append(e.cpu())
            pred_2d.append(gps.head(e).cpu().numpy())
    emb_2d = torch.cat(emb_2d)
    pred_2d = np.concatenate(pred_2d)

    emb_3d = []
    pred_3d = []
    with torch.no_grad():
        for batch in GeometricDataLoader(g3d_list, batch_size=128):
            batch = batch.to(device)
            charges = batch.charges if hasattr(batch, "charges") else None
            e = schnet.encode(batch.z, batch.pos, batch.batch, charges=charges)
            emb_3d.append(e.cpu())
            pred_3d.append(schnet.head(e).cpu().numpy())
    emb_3d = torch.cat(emb_3d)
    pred_3d = np.concatenate(pred_3d)

    with torch.no_grad():
        pred_hybrid = fusion(emb_2d.to(device), emb_3d.to(device)).cpu().numpy()
        pred_moe = moe(emb_2d.to(device), emb_3d.to(device)).cpu().numpy()

    valid_idx = np.array(valid_idx)
    y_true = ood.iloc[valid_idx][TARGET_COLS].values.astype(np.float32)
    return ood.iloc[valid_idx].reset_index(drop=True), y_true, {
        "2d": pred_2d,
        "3d": pred_3d,
        "hybrid": pred_hybrid,
        "moe": pred_moe,
    }


def print_ood_table(blocks: dict):
    names = ["2d", "3d", "hybrid", "moe"]
    print("\nOOD-1000 B3LYP labels")
    print("target     " + "  ".join(f"{name:^18s}" for name in names))
    print("           " + "  ".join(f"{'MAE':>8s} {'R2':>8s}" for _ in names))
    for target in TARGET_COLS + ["average"]:
        row = f"{target:8s} "
        for name in names:
            row += f"  {blocks[name][target]['mae']:8.4f} {blocks[name][target]['r2']:8.4f}"
        print(row)


def save_predictions(path, df, preds):
    out = df.copy()
    for model_name, values in preds.items():
        for i, target in enumerate(TARGET_COLS):
            out[f"{target}_{model_name}"] = values[:, i]
    out.to_csv(path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5.4e-4)
    parser.add_argument("--weight-decay", type=float, default=4.4132081179616e-06)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--result-json", type=str, default=str(DEFAULT_RESULT))
    parser.add_argument("--pred-csv", type=str, default=str(DEFAULT_PRED))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    args.checkpoint = Path(args.checkpoint)
    if not args.checkpoint.is_absolute():
        args.checkpoint = MODELS_DIR / args.checkpoint
    args.result_json = Path(args.result_json)
    args.pred_csv = Path(args.pred_csv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}", flush=True)

    train_metrics = None
    if not args.skip_train:
        train_metrics = train_moe(args, device)
    elif not args.checkpoint.exists():
        raise FileNotFoundError(f"--skip-train requested, but checkpoint is missing: {args.checkpoint}")

    moe = load_moe(args.checkpoint, device)
    ood_df, y_true, preds = predict_ood(device, moe)
    blocks = {name: metrics_block(y_true, pred) for name, pred in preds.items()}
    print_ood_table(blocks)

    result = {
        "checkpoint": str(args.checkpoint),
        "train": train_metrics,
        "ood": {"n": int(len(ood_df)), **blocks},
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    save_predictions(args.pred_csv, ood_df, preds)
    print(f"\nSaved metrics: {args.result_json}", flush=True)
    print(f"Saved predictions: {args.pred_csv}", flush=True)


if __name__ == "__main__":
    main()
