"""
Phase 9: LoRA-adapt GPS/SchNet encoders plus FusionHead from B3LYP to GW.

This is the next step after FusionHead-only LoRA. The B3LYP checkpoints remain
unchanged; adapters are saved as separate experimental checkpoints.

Targets:
  --targets gps fusion          # 2D encoder + fusion adapters
  --targets schnet fusion       # 3D encoder + fusion adapters
  --targets gps schnet fusion   # both encoders + fusion adapters

Usage:
  .venv\\Scripts\\python.exe scripts/phase9/train_encoder_lora_delta.py --targets gps fusion --rank 4
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch_geometric.data import Batch

from molgap.constants import MODELS_DIR, RESULTS_DIR, TARGET_COLS
from molgap.graphs import smiles_to_2d_pyg, smiles_to_pyg
from molgap.inference import load_hybrid
from molgap.utils import murcko_scaffold_smiles

PHASE9 = RESULTS_DIR / "phase9"
CSV = PHASE9 / "delta_oe62.csv"
GRAPH_CACHE = PHASE9 / "delta_oe62_graphs.pt"
SEED = 42
TEST_FRAC = 0.2


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_b(self.lora_a(self.dropout(x)))


def inject_lora(module: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    if isinstance(module, nn.MultiheadAttention):
        # MultiheadAttention.forward accesses out_proj.weight directly, so a
        # wrapper module breaks it. Keep attention projection frozen for this
        # pilot and adapt the surrounding GPS/GINE/Fusion linear layers.
        return 0
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            count += 1
        else:
            count += inject_lora(child, rank, alpha, dropout)
    return count


def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def trainable_params(*modules: nn.Module) -> int:
    return int(sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad))


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_or_build_graphs(csv_path: Path, graph_cache: Path):
    if graph_cache.exists():
        obj = torch.load(graph_cache, weights_only=False)
        return obj["df"], obj["g2d"], obj["g3d"]

    df = pd.read_csv(csv_path)
    g2d, g3d, keep = [], [], []
    for i, smi in enumerate(df["smiles"].tolist()):
        a = smiles_to_2d_pyg(smi)
        b = smiles_to_pyg(smi)
        if a is None or b is None:
            continue
        g2d.append(a)
        g3d.append(b)
        keep.append(i)
        if len(keep) % 500 == 0:
            print(f"  graphs {len(keep)}/{len(df)}", flush=True)
    df = df.iloc[keep].reset_index(drop=True)
    graph_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"df": df, "g2d": g2d, "g3d": g3d}, graph_cache)
    print(f"Saved graph cache: {graph_cache} ({len(df)} molecules)", flush=True)
    return df, g2d, g3d


def scaffold_split(smiles: list[str], split_seed: int):
    scaffolds = [murcko_scaffold_smiles(s) or "NONE" for s in smiles]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=split_seed)
    all_idx = np.arange(len(smiles))
    train_val, test = next(gss.split(all_idx, groups=scaffolds))
    train, val = train_test_split(train_val, test_size=0.1, random_state=split_seed)
    return train, val, test, len(set(scaffolds))


def iter_batches(indices: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.RandomState):
    idx = np.array(indices, copy=True)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield idx[start:start + batch_size]


def forward_batch(gps, schnet, fusion, g2d, g3d, indices, device):
    b2 = Batch.from_data_list([g2d[int(i)] for i in indices]).to(device)
    b3 = Batch.from_data_list([g3d[int(i)] for i in indices]).to(device)
    e2 = gps.encode(b2.x, b2.edge_index, b2.edge_attr, b2.batch)
    charges = b3.charges if hasattr(b3, "charges") else None
    e3 = schnet.encode(b3.z, b3.pos, b3.batch, charges=charges)
    return fusion(e2, e3)


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


def evaluate(gps, schnet, fusion, g2d, g3d, y, indices, device, batch_size):
    gps.eval(); schnet.eval(); fusion.eval()
    preds = []
    with torch.no_grad():
        for batch_idx in iter_batches(indices, batch_size, False, np.random.RandomState(0)):
            preds.append(forward_batch(gps, schnet, fusion, g2d, g3d, batch_idx, device).cpu().numpy())
    return np.concatenate(preds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", choices=["gps", "schnet", "fusion"],
                        default=["gps", "schnet", "fusion"])
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--hybrid-key", default="phase7_hybrid")
    parser.add_argument("--csv", type=Path, default=CSV)
    parser.add_argument("--graph-cache", type=Path, default=GRAPH_CACHE)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = resolve_device(args.device)
    targets = tuple(sorted(set(args.targets)))
    name = args.name or "_".join(targets) + f"_r{args.rank}"
    print(
        f"Device: {device} | targets={targets} | rank={args.rank} "
        f"| seed={args.seed} split_seed={args.split_seed}",
        flush=True,
    )

    df, g2d, g3d = load_or_build_graphs(args.csv, args.graph_cache)
    y = torch.tensor(df[[f"gw_{t}" for t in TARGET_COLS]].values, dtype=torch.float32)
    gw = y.numpy()
    raw = df[[f"pred_{t}" for t in TARGET_COLS]].values.astype(np.float32)
    train, val, test, n_scaffolds = scaffold_split(df["smiles"].tolist(), args.split_seed)
    print(
        f"OE62 graphs: n={len(df)} scaffolds={n_scaffolds} "
        f"train/val/test={len(train)}/{len(val)}/{len(test)}",
        flush=True,
    )

    gps, schnet, fusion, _ = load_hybrid(device, key=args.hybrid_key)
    freeze(gps); freeze(schnet); freeze(fusion)
    layer_counts = {}
    if "gps" in targets:
        layer_counts["gps"] = inject_lora(gps, args.rank, args.alpha, args.lora_dropout)
    if "schnet" in targets:
        layer_counts["schnet"] = inject_lora(schnet, args.rank, args.alpha, args.lora_dropout)
    if "fusion" in targets:
        layer_counts["fusion"] = inject_lora(fusion, args.rank, args.alpha, args.lora_dropout)
    gps.to(device); schnet.to(device); fusion.to(device)
    n_trainable = trainable_params(gps, schnet, fusion)
    print(f"LoRA layers={layer_counts} trainable_params={n_trainable:,}", flush=True)

    params = [p for m in (gps, schnet, fusion) for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-6)
    crit = nn.L1Loss()

    best_val = float("inf")
    best_state = None
    wait = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        gps.train(); schnet.train(); fusion.train()
        tr_loss, tr_count = 0.0, 0
        for batch_idx in iter_batches(train, args.batch_size, True, rng):
            pred = forward_batch(gps, schnet, fusion, g2d, g3d, batch_idx, device)
            yy = y[batch_idx].to(device)
            opt.zero_grad()
            loss = crit(pred, yy)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(batch_idx)
            tr_count += len(batch_idx)

        gps.eval(); schnet.eval(); fusion.eval()
        va_loss, va_count = 0.0, 0
        with torch.no_grad():
            for batch_idx in iter_batches(val, args.batch_size, False, rng):
                pred = forward_batch(gps, schnet, fusion, g2d, g3d, batch_idx, device)
                yy = y[batch_idx].to(device)
                va_loss += crit(pred, yy).item() * len(batch_idx)
                va_count += len(batch_idx)
        train_mae = tr_loss / tr_count
        val_mae = va_loss / va_count
        sched.step(val_mae)

        improved = val_mae < best_val
        if improved:
            best_val = val_mae
            best_state = {
                "gps": {k: v.detach().cpu().clone() for k, v in gps.state_dict().items()},
                "schnet": {k: v.detach().cpu().clone() for k, v in schnet.state_dict().items()},
                "fusion": {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()},
            }
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % args.log_every == 0 or improved:
            mark = "*" if improved else " "
            print(
                f"{mark} epoch {epoch:03d} train_mae={train_mae:.4f} "
                f"val_mae={val_mae:.4f} best={best_val:.4f} wait={wait}",
                flush=True,
            )
        if wait >= args.patience:
            print(f"Early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No best state captured")
    gps.load_state_dict(best_state["gps"])
    schnet.load_state_dict(best_state["schnet"])
    fusion.load_state_dict(best_state["fusion"])

    pred_lora = evaluate(gps, schnet, fusion, g2d, g3d, y, test, device, args.batch_size)
    y_test = gw[test]
    raw_test = raw[test]
    delta_train = gw[train] - raw[train]
    const = raw_test + delta_train.mean(axis=0, keepdims=True)
    blocks = {
        "raw_b3lyp": metrics_block(y_test, raw_test),
        "const_delta": metrics_block(y_test, const),
        "encoder_lora_gw": metrics_block(y_test, pred_lora),
    }

    print("\nScaffold-test GW MAE/R2")
    for key in ["raw_b3lyp", "const_delta", "encoder_lora_gw"]:
        b = blocks[key]
        print(
            f"{key:16s} avg MAE={b['average']['mae']:.4f} R2={b['average']['r2']:.4f} | "
            f"H {b['homo']['mae']:.4f} L {b['lumo']['mae']:.4f} G {b['gap']['mae']:.4f}",
            flush=True,
        )

    out_ckpt = MODELS_DIR / f"hybrid_encoder_lora_gw_{name}.pt"
    torch.save(
        {
            "targets": targets,
            "rank": args.rank,
            "alpha": args.alpha,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "layer_counts": layer_counts,
            "trainable_params": n_trainable,
            "best_val_mae": best_val,
            "state": best_state,
        },
        out_ckpt,
    )
    result = {
        "targets": list(targets),
        "hybrid_key": args.hybrid_key,
        "csv": str(args.csv),
        "graph_cache": str(args.graph_cache),
        "rank": args.rank,
        "alpha": args.alpha,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "layer_counts": layer_counts,
        "trainable_params": n_trainable,
        "best_val_mae": float(best_val),
        "train_time_s": float(time.time() - t0),
        "n": int(len(df)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "metrics": blocks,
    }
    out_metrics = PHASE9 / f"encoder_lora_delta_{name}_metrics.json"
    out_metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pred_df = df.iloc[test].reset_index(drop=True).copy()
    for i, target in enumerate(TARGET_COLS):
        pred_df[f"gw_pred_encoder_lora_{target}"] = pred_lora[:, i]
        pred_df[f"gw_pred_const_{target}"] = const[:, i]
    out_pred = PHASE9 / f"encoder_lora_delta_{name}_predictions.csv"
    pred_df.to_csv(out_pred, index=False, encoding="utf-8")
    print(f"\nSaved checkpoint: {out_ckpt}", flush=True)
    print(f"Saved metrics: {out_metrics}", flush=True)
    print(f"Saved predictions: {out_pred}", flush=True)


if __name__ == "__main__":
    main()
