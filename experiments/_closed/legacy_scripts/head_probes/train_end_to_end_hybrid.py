"""
Train a Phase 8 end-to-end hybrid on aligned 2D/3D graph caches.

Unlike the frozen-embedding fusion probes, this jointly trains:
  GPS 2D encoder + SchNet 3D encoder + FusionHead/MoEFusionHead.

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/head_probes/train_end_to_end_hybrid.py --tag replacement30k --head moe
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/head_probes/train_end_to_end_hybrid.py --tag replacement30k --head moe --max-samples 512 --epochs 1
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
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from molgap.constants import PARAMS_GPS_2D, PARAMS_SCHNET_300K, RESULTS_DIR, SEED
from molgap.gps import GPSWrapper
from molgap.hybrid import EndToEndHybrid
from molgap.schnet import SchNetWrapper
from molgap.utils import ensure_dirs

PHASE8_DIR = RESULTS_DIR / "phase8"


class PairedGraphDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_pairs(items):
    graphs_2d = [x[0] for x in items]
    graphs_3d = [x[1] for x in items]
    return Batch.from_data_list(graphs_2d), Batch.from_data_list(graphs_3d)


def load_aligned_pairs(graphs_2d_path: Path, graphs_3d_path: Path, max_samples: int | None):
    graphs_2d = torch.load(graphs_2d_path, weights_only=False)
    graphs_3d = torch.load(graphs_3d_path, weights_only=False)
    by_2d = {int(g.source_idx.view(-1)[0].item()): g for g in graphs_2d}
    by_3d = {int(g.source_idx.view(-1)[0].item()): g for g in graphs_3d}
    common = sorted(set(by_2d).intersection(by_3d))
    if max_samples is not None:
        common = common[:max_samples]
    pairs = [(by_2d[i], by_3d[i]) for i in common]
    return pairs, common


def make_split(n: int):
    idx = np.random.RandomState(SEED).permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


def subset(pairs, idx):
    return [pairs[int(i)] for i in idx]


def make_model(head: str, experts: int):
    gps = GPSWrapper(**PARAMS_GPS_2D)
    schnet = SchNetWrapper(**PARAMS_SCHNET_300K, use_charges=True)
    return EndToEndHybrid(gps, schnet, head=head, hidden=192, dropout=0.0, n_experts=experts)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    pred, true = [], []
    for batch_2d, batch_3d in loader:
        batch_2d = batch_2d.to(device)
        batch_3d = batch_3d.to(device)
        out = model(batch_2d, batch_3d)
        pred.append(out.float().cpu().numpy())
        true.append(batch_3d.y.float().cpu().numpy())
    pred = np.concatenate(pred)
    true = np.concatenate(true)
    metrics = {}
    for i, name in enumerate(["HOMO", "LUMO", "Gap"]):
        metrics[name] = {
            "mae": float(mean_absolute_error(true[:, i], pred[:, i])),
            "r2": float(r2_score(true[:, i], pred[:, i])),
        }
    metrics["average"] = {
        "mae": float(np.mean([metrics[k]["mae"] for k in ["HOMO", "LUMO", "Gap"]])),
        "r2": float(np.mean([metrics[k]["r2"] for k in ["HOMO", "LUMO", "Gap"]])),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train end-to-end Phase 8 hybrid")
    parser.add_argument("--tag", default="replacement30k",
                        help="graph tag, e.g. old30k or replacement30k")
    parser.add_argument("--graphs-2d", type=Path, default=None)
    parser.add_argument("--graphs-3d", type=Path, default=None)
    parser.add_argument("--head", choices=["single", "moe"], default="moe")
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model-out", type=Path, default=None)
    args = parser.parse_args()

    ensure_dirs(PHASE8_DIR)
    graphs_2d = args.graphs_2d or PHASE8_DIR / f"pyg_2d_graphs_bond_{args.tag}.pt"
    graphs_3d = args.graphs_3d or PHASE8_DIR / f"pyg_3d_graphs_etkdg_{args.tag}.pt"
    suffix = f"{args.tag}_{args.head}"
    if args.max_samples:
        suffix += f"_n{args.max_samples}"
    out_path = args.out or PHASE8_DIR / f"end2end_{suffix}_metrics.json"
    model_path = args.model_out or PHASE8_DIR / f"end2end_{suffix}.pt"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | tag={args.tag} | head={args.head}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"{props.name} | {props.total_memory / 1e9:.1f} GB", flush=True)

    pairs, common = load_aligned_pairs(graphs_2d, graphs_3d, args.max_samples)
    split = make_split(len(pairs))
    print(
        f"Aligned N={len(pairs)} source_idx={common[0]}..{common[-1]} "
        f"split={len(split['train'])}/{len(split['val'])}/{len(split['test'])}",
        flush=True,
    )

    train_loader = DataLoader(
        PairedGraphDataset(subset(pairs, split["train"])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_pairs,
    )
    val_loader = DataLoader(
        PairedGraphDataset(subset(pairs, split["val"])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pairs,
    )
    test_loader = DataLoader(
        PairedGraphDataset(subset(pairs, split["test"])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pairs,
    )

    model = make_model(args.head, args.experts).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    crit = nn.L1Loss()

    best_val, best_state, best_epoch, wait = float("inf"), None, -1, 0
    log_rows = []
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for batch_2d, batch_3d in train_loader:
            batch_2d = batch_2d.to(device)
            batch_3d = batch_3d.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                pred = model(batch_2d, batch_3d)
                loss = crit(pred, batch_3d.y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            total += loss.item() * batch_3d.num_graphs
            n += batch_3d.num_graphs
        train_loss = total / max(n, 1)

        model.eval()
        vtotal, vn = 0.0, 0
        with torch.no_grad():
            for batch_2d, batch_3d in val_loader:
                batch_2d = batch_2d.to(device)
                batch_3d = batch_3d.to(device)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    loss = crit(model(batch_2d, batch_3d), batch_3d.y)
                vtotal += loss.item() * batch_3d.num_graphs
                vn += batch_3d.num_graphs
        val_loss = vtotal / max(vn, 1)
        sched.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            mark = " *"
        else:
            wait += 1
            mark = ""
        elapsed = time.time() - t0
        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_loss,
            "lr": opt.param_groups[0]["lr"],
            "time_s": elapsed,
        })
        print(
            f"ep{epoch:03d} train={train_loss:.4f} val={val_loss:.4f} "
            f"best={best_val:.4f}@{best_epoch} lr={opt.param_groups[0]['lr']:.2e} "
            f"{elapsed:.1f}s{mark}",
            flush=True,
        )
        if wait >= args.patience:
            print(f"Early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No checkpoint state was produced")
    model.load_state_dict(best_state)
    torch.save(best_state, model_path)
    metrics = evaluate(model, test_loader, device)
    result = {
        "tag": args.tag,
        "head": args.head,
        "experts": args.experts if args.head == "moe" else None,
        "n_aligned": len(pairs),
        "best_val_mae": float(best_val),
        "best_epoch": int(best_epoch),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "test_metrics": metrics,
        "log": log_rows,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Model -> {model_path}", flush=True)
    print(f"Metrics -> {out_path}", flush=True)
    print(
        f"Test avg MAE={metrics['average']['mae']:.4f} Gap MAE={metrics['Gap']['mae']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
