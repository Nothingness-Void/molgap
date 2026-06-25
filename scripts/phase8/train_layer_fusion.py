"""
Train a frozen-encoder intermediate-layer fusion head.

This probes whether GPS/SchNet final pooled embeddings discard useful information
from earlier layers. Encoders stay frozen; only a standard FusionHead is trained
on concatenated selected-layer embeddings.

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/train_layer_fusion.py --tag replacement30k
  .venv\\Scripts\\python.exe scripts/phase8/train_layer_fusion.py --tag replacement30k --epochs 3 --max-samples 2000
  .venv\\Scripts\\python.exe scripts/phase8/train_layer_fusion.py --tag phase7_300k --graph-2d results/phase7/pyg_2d_graphs_bond_300k.pt --graph-3d results/phase7/pyg_3d_graphs_etkdg_300k.pt --gps-model models/gps_2d_300k.pt --schnet-model models/gnn_schnet_3d_300k.pt --align-2d-idx results/phase7/align_2d_idx.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

from molgap.constants import PARAMS_GPS_2D, PARAMS_SCHNET_300K, RESULTS_DIR, SEED
from molgap.fusion import FusionHead
from molgap.gps import GPSWrapper
from molgap.schnet import SchNetWrapper
from molgap.utils import ensure_dirs

PHASE8_DIR = RESULTS_DIR / "phase8"
TARGETS = ["HOMO", "LUMO", "Gap"]


def _parse_layers(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _graph_path(kind: str, tag: str) -> Path:
    prefix = "pyg_2d_graphs_bond" if kind == "gps" else "pyg_3d_graphs_etkdg"
    return PHASE8_DIR / f"{prefix}_{tag}.pt"


def _model_path(kind: str, tag: str) -> Path:
    return PHASE8_DIR / f"{kind}_{tag}.pt"


def _make_split(n: int, max_samples: int | None):
    idx = np.random.RandomState(SEED).permutation(n)
    if max_samples is not None:
        idx = idx[:max_samples]
    n_train, n_val = int(0.8 * len(idx)), int(0.1 * len(idx))
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


@torch.no_grad()
def _extract_gps_layers(graphs, model_path: Path, layers: tuple[int, ...], batch_size: int, device):
    model = GPSWrapper(**PARAMS_GPS_2D).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    embs, source_idx = [], []
    loader = GeometricDataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    t0 = time.time()
    offset = 0
    for bi, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            emb = model.encode_layers(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch, layers=layers
            )
        embs.append(emb.float().cpu())
        if "source_idx" in batch:
            source_idx.append(batch.source_idx.view(-1).cpu())
        else:
            source_idx.append(torch.arange(offset, offset + batch.num_graphs, dtype=torch.long))
        offset += batch.num_graphs
        if bi % 20 == 0 or bi == len(loader) - 1:
            print(f"  GPS layer batch {bi + 1}/{len(loader)} ({time.time() - t0:.0f}s)", flush=True)
    return torch.cat(embs), torch.cat(source_idx)


@torch.no_grad()
def _extract_schnet_layers(graphs, model_path: Path, layers: tuple[int, ...], batch_size: int, device):
    model = SchNetWrapper(**PARAMS_SCHNET_300K, use_charges=True).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    embs, source_idx, labels = [], [], []
    loader = GeometricDataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    t0 = time.time()
    offset = 0
    for bi, batch in enumerate(loader):
        batch = batch.to(device)
        charges = batch.charges if hasattr(batch, "charges") else None
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            emb = model.encode_layers(batch.z, batch.pos, batch.batch, charges=charges, layers=layers)
        embs.append(emb.float().cpu())
        if "source_idx" in batch:
            source_idx.append(batch.source_idx.view(-1).cpu())
        else:
            source_idx.append(torch.arange(offset, offset + batch.num_graphs, dtype=torch.long))
        offset += batch.num_graphs
        labels.append(batch.y.float().cpu())
        if bi % 20 == 0 or bi == len(loader) - 1:
            print(f"  SchNet layer batch {bi + 1}/{len(loader)} ({time.time() - t0:.0f}s)", flush=True)
    return torch.cat(embs), torch.cat(source_idx), torch.cat(labels)


def _align(h2, idx2, h3, idx3, y3):
    pos2 = {int(v): i for i, v in enumerate(idx2.tolist())}
    pos3 = {int(v): i for i, v in enumerate(idx3.tolist())}
    common = sorted(set(pos2).intersection(pos3))
    ii2 = torch.tensor([pos2[i] for i in common], dtype=torch.long)
    ii3 = torch.tensor([pos3[i] for i in common], dtype=torch.long)
    return h2[ii2], h3[ii3], y3[ii3], torch.tensor(common, dtype=torch.long)


def _make_loader(h2, h3, y, idx, batch_size, shuffle):
    return TorchDataLoader(
        TensorDataset(h2[idx], h3[idx], y[idx]),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,
    )


def _metrics(pred, true):
    out = {}
    for i, name in enumerate(TARGETS):
        out[name] = {
            "mae": float(mean_absolute_error(true[:, i], pred[:, i])),
            "r2": float(r2_score(true[:, i], pred[:, i])),
        }
    out["average"] = {
        "mae": float(np.mean([out[k]["mae"] for k in TARGETS])),
        "r2": float(np.mean([out[k]["r2"] for k in TARGETS])),
    }
    return out


@torch.no_grad()
def _eval(model, h2, h3, y, idx, batch_size, device):
    model.eval()
    pred, true = [], []
    for b2, b3, by in _make_loader(h2, h3, y, idx, batch_size, False):
        pred.append(model(b2.to(device), b3.to(device)).float().cpu().numpy())
        true.append(by.numpy())
    return _metrics(np.concatenate(pred), np.concatenate(true))


def _train_head(h2, h3, y, split, args, device):
    model = FusionHead(
        "gate",
        hidden=args.hidden,
        dropout=args.dropout,
        dim_2d=h2.shape[1],
        dim_3d=h3.shape[1],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-6)
    crit = nn.L1Loss()
    train_loader = _make_loader(h2, h3, y, split["train"], args.batch_size, True)

    best_val, best_state, best_epoch, wait = float("inf"), None, -1, 0
    log_rows = []
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        for b2, b3, by in train_loader:
            opt.zero_grad()
            loss = crit(model(b2.to(device), b3.to(device)), by.to(device))
            loss.backward()
            opt.step()

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for b2, b3, by in _make_loader(h2, h3, y, split["val"], 2048, False):
                loss = crit(model(b2.to(device), b3.to(device)), by.to(device))
                total += loss.item() * by.size(0)
                n += by.size(0)
        val = total / max(n, 1)
        sched.step(val)
        if val < best_val:
            best_val = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        log_rows.append({"epoch": epoch, "val_mae": float(val), "time_s": time.time() - t0})
        print(f"ep{epoch:03d} val={val:.4f} best={best_val:.4f}@{best_epoch}", flush=True)
        if wait >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No valid layer-fusion checkpoint produced")
    model.load_state_dict(best_state)
    metrics = _eval(model, h2, h3, y, split["test"], 2048, device)
    metrics["best_val_mae"] = float(best_val)
    metrics["best_epoch"] = int(best_epoch)
    metrics["n_params"] = int(sum(p.numel() for p in model.parameters()))
    metrics["log"] = log_rows
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 8 intermediate-layer fusion")
    parser.add_argument("--tag", default="replacement30k")
    parser.add_argument("--graph-2d", type=Path, default=None)
    parser.add_argument("--graph-3d", type=Path, default=None)
    parser.add_argument("--gps-model", type=Path, default=None)
    parser.add_argument("--schnet-model", type=Path, default=None)
    parser.add_argument("--align-2d-idx", type=Path, default=None)
    parser.add_argument("--gps-layers", default="2,4,-1")
    parser.add_argument("--schnet-layers", default="2,4,-1")
    parser.add_argument("--gps-batch-size", type=int, default=256)
    parser.add_argument("--schnet-batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5.4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model-out", type=Path, default=None)
    args = parser.parse_args()

    ensure_dirs(PHASE8_DIR)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | tag={args.tag}", flush=True)

    gps_layers = _parse_layers(args.gps_layers)
    schnet_layers = _parse_layers(args.schnet_layers)
    graph_2d = args.graph_2d or _graph_path("gps", args.tag)
    graph_3d = args.graph_3d or _graph_path("schnet", args.tag)
    gps_model = args.gps_model or _model_path("gps", args.tag)
    schnet_model = args.schnet_model or _model_path("schnet", args.tag)
    print(f"2D graphs: {graph_2d}", flush=True)
    print(f"3D graphs: {graph_3d}", flush=True)
    print(f"GPS model: {gps_model}", flush=True)
    print(f"SchNet model: {schnet_model}", flush=True)
    g2d = torch.load(graph_2d, weights_only=False)
    g3d = torch.load(graph_3d, weights_only=False)

    h2, idx2 = _extract_gps_layers(
        g2d, gps_model, gps_layers, args.gps_batch_size, device
    )
    h3, idx3, y3 = _extract_schnet_layers(
        g3d, schnet_model, schnet_layers, args.schnet_batch_size, device
    )
    if args.align_2d_idx is not None:
        keep = torch.load(args.align_2d_idx, weights_only=False).long()
        if keep.numel() != h3.shape[0]:
            raise ValueError(
                f"align-2d-idx length {keep.numel()} does not match 3D N={h3.shape[0]}"
            )
        h2 = h2[keep]
        idx2 = torch.arange(keep.numel(), dtype=torch.long)
        idx3 = torch.arange(h3.shape[0], dtype=torch.long)
    h2, h3, y, source_idx = _align(h2, idx2, h3, idx3, y3)
    split = _make_split(h2.shape[0], args.max_samples)
    print(
        f"Aligned N={h2.shape[0]} dims={h2.shape[1]}+{h3.shape[1]} "
        f"split={len(split['train'])}/{len(split['val'])}/{len(split['test'])}",
        flush=True,
    )

    model, metrics = _train_head(h2, h3, y, split, args, device)
    result = {
        "tag": args.tag,
        "n_aligned": int(h2.shape[0]),
        "max_samples": args.max_samples,
        "source_idx_min": int(source_idx.min().item()),
        "source_idx_max": int(source_idx.max().item()),
        "gps_layers": list(gps_layers),
        "schnet_layers": list(schnet_layers),
        "graph_2d": str(graph_2d),
        "graph_3d": str(graph_3d),
        "gps_model": str(gps_model),
        "schnet_model": str(schnet_model),
        "align_2d_idx": None if args.align_2d_idx is None else str(args.align_2d_idx),
        "dim_2d": int(h2.shape[1]),
        "dim_3d": int(h3.shape[1]),
        "metrics": metrics,
    }

    out = args.out or PHASE8_DIR / f"layer_fusion_{args.tag}_metrics.json"
    model_out = args.model_out or PHASE8_DIR / f"layer_fusion_{args.tag}.pt"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), model_out)
    print(f"Metrics -> {out}", flush=True)
    print(f"Model -> {model_out}", flush=True)
    print(
        f"Layer fusion avg={metrics['average']['mae']:.5f} gap={metrics['Gap']['mae']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
