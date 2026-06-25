"""
Train one Phase 8 encoder and extract full-cache embeddings.

This is a thin Phase 8 wrapper around the reusable model classes in src/molgap.
It never writes Phase 7 checkpoints.

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/train_encoder.py --kind gps
  .venv\\Scripts\\python.exe scripts/phase8/train_encoder.py --kind schnet
  .venv\\Scripts\\python.exe scripts/phase8/train_encoder.py --kind gps --max-samples 2000 --epochs 2
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
from torch_geometric.loader import DataLoader

from molgap.constants import MODELS_DIR, PARAMS_GPS_2D, PARAMS_SCHNET_300K, RESULTS_DIR, SEED
from molgap.gps import GPSWrapper
from molgap.schnet import SchNetWrapper
from molgap.utils import ensure_dirs

PHASE8_DIR = RESULTS_DIR / "phase8"
GRAPH_2D = PHASE8_DIR / "pyg_2d_graphs_bond_replacement_300k.pt"
GRAPH_3D = PHASE8_DIR / "pyg_3d_graphs_etkdg_replacement_300k.pt"

TRAIN_PARAMS = {
    "gps": {
        **PARAMS_GPS_2D,
        "lr": 0.0004754654349367296,
        "weight_decay": 1.3094136884618282e-05,
        "batch_size": 256,
        "scheduler": "cosine",
    },
    "schnet": {
        **PARAMS_SCHNET_300K,
        "lr": 0.00021150972021685588,
        "weight_decay": 1.4656553886225336e-05,
        "batch_size": 128,
        "scheduler": "cosine",
    },
}


def _make_model(kind: str):
    if kind == "gps":
        p = PARAMS_GPS_2D
        return GPSWrapper(**p)
    if kind == "schnet":
        p = PARAMS_SCHNET_300K
        return SchNetWrapper(**p, use_charges=True)
    raise ValueError(kind)


def _forward(kind: str, model, batch):
    if kind == "gps":
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    charges = batch.charges if hasattr(batch, "charges") else None
    return model(batch.z, batch.pos, batch.batch, charges=charges)


def _encode(kind: str, model, batch):
    if kind == "gps":
        return model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    charges = batch.charges if hasattr(batch, "charges") else None
    return model.encode(batch.z, batch.pos, batch.batch, charges=charges)


@torch.no_grad()
def _evaluate(kind: str, model, loader, device):
    model.eval()
    pred, true = [], []
    for batch in loader:
        batch = batch.to(device)
        out = _forward(kind, model, batch)
        pred.append(out.float().cpu().numpy())
        true.append(batch.y.float().cpu().numpy())
    return np.concatenate(pred), np.concatenate(true)


def _metrics(pred, true):
    out = {}
    for i, name in enumerate(["HOMO", "LUMO", "Gap"]):
        out[name] = {
            "mae": float(mean_absolute_error(true[:, i], pred[:, i])),
            "r2": float(r2_score(true[:, i], pred[:, i])),
        }
    out["average"] = {
        "mae": float(np.mean([out[k]["mae"] for k in ["HOMO", "LUMO", "Gap"]])),
        "r2": float(np.mean([out[k]["r2"] for k in ["HOMO", "LUMO", "Gap"]])),
    }
    return out


def _extract_embeddings(kind: str, model, graphs, device, out_path: Path, batch_size: int):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    n_batches = (len(graphs) + batch_size - 1) // batch_size
    embs, source_idx = [], []
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                emb = _encode(kind, model, batch)
            embs.append(emb.float().cpu())
            source_idx.append(batch.source_idx.view(-1).cpu())
            if bi % 20 == 0 or bi == n_batches - 1:
                print(f"  embed batch {bi + 1}/{n_batches} ({time.time() - t0:.0f}s)",
                      flush=True)
    payload = {
        "embeddings": torch.cat(embs, dim=0),
        "source_idx": torch.cat(source_idx, dim=0),
    }
    torch.save(payload, out_path)
    print(f"Embeddings -> {out_path} {tuple(payload['embeddings'].shape)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train a Phase 8 encoder")
    parser.add_argument("--kind", choices=["gps", "schnet"], required=True)
    parser.add_argument("--graphs", type=Path, default=None)
    parser.add_argument("--model-out", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--embeddings-out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--extract-only", action="store_true",
                        help="load --model-out and only write --embeddings-out")
    args = parser.parse_args()

    ensure_dirs(PHASE8_DIR, MODELS_DIR)
    graph_path = args.graphs or (GRAPH_2D if args.kind == "gps" else GRAPH_3D)
    model_out = args.model_out or MODELS_DIR / f"phase8_{args.kind}_replacement_300k.pt"
    metrics_out = args.metrics_out or PHASE8_DIR / f"{args.kind}_replacement_300k_metrics.json"
    embeddings_out = args.embeddings_out or PHASE8_DIR / f"{args.kind}_replacement_300k_embeddings.pt"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | kind={args.kind}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"{props.name} | {props.total_memory / 1e9:.1f} GB", flush=True)

    graphs = torch.load(graph_path, weights_only=False)
    if args.max_samples is not None:
        graphs = graphs[:args.max_samples]
    print(f"Loaded {len(graphs)} graphs from {graph_path}", flush=True)

    if args.extract_only:
        model = _make_model(args.kind).to(device)
        model.load_state_dict(torch.load(model_out, weights_only=False, map_location=device))
        _extract_embeddings(args.kind, model, graphs, device, embeddings_out,
                            max(int(TRAIN_PARAMS[args.kind]["batch_size"]), 256))
        return

    idx = np.random.RandomState(SEED).permutation(len(graphs))
    n_train, n_val = int(0.8 * len(graphs)), int(0.1 * len(graphs))
    train_set = [graphs[i] for i in idx[:n_train]]
    val_set = [graphs[i] for i in idx[n_train:n_train + n_val]]
    test_set = [graphs[i] for i in idx[n_train + n_val:]]
    print(f"Split: train={len(train_set)} val={len(val_set)} test={len(test_set)}", flush=True)

    p = TRAIN_PARAMS[args.kind]
    model = _make_model(args.kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    criterion = nn.L1Loss()
    bs = int(p["batch_size"])
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=0)

    best_val, best_state, best_epoch, wait = float("inf"), None, -1, 0
    log_rows = []
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss = criterion(_forward(args.kind, model, batch), batch.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * batch.num_graphs
            n += batch.num_graphs
        train_loss = total / max(n, 1)

        model.eval()
        vtotal, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    loss = criterion(_forward(args.kind, model, batch), batch.y)
                vtotal += loss.item() * batch.num_graphs
                vn += batch.num_graphs
        val_loss = vtotal / max(vn, 1)
        scheduler.step()

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        elapsed = time.time() - t0
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_mae": val_loss,
                         "lr": optimizer.param_groups[0]["lr"], "time_s": elapsed})
        mark = " *" if improved else ""
        print(f"ep{epoch:03d} train={train_loss:.4f} val={val_loss:.4f} "
              f"best={best_val:.4f}@{best_epoch} lr={optimizer.param_groups[0]['lr']:.2e} "
              f"{elapsed:.1f}s{mark}", flush=True)
        if wait >= args.patience:
            print(f"Early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint state was produced")
    model.load_state_dict(best_state)
    torch.save(best_state, model_out)
    print(f"Model -> {model_out}", flush=True)

    test_loader = DataLoader(test_set, batch_size=max(bs, 256), shuffle=False, num_workers=0)
    pred, true = _evaluate(args.kind, model, test_loader, device)
    result = {
        "kind": args.kind,
        "graph_path": str(graph_path),
        "n_graphs": len(graphs),
        "best_val_mae": float(best_val),
        "best_epoch": int(best_epoch),
        "params": p,
        "test_metrics": _metrics(pred, true),
        "log": log_rows,
    }
    metrics_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Metrics -> {metrics_out}", flush=True)

    if not args.no_embeddings:
        _extract_embeddings(args.kind, model, graphs, device, embeddings_out, max(bs, 256))


if __name__ == "__main__":
    main()
