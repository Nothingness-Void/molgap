"""
Phase 9: OOD stress-test for the encoder-LoRA GW adaptation.

The scaffold-disjoint test in train_encoder_lora_delta.py is still OE62-internal
and same-MW. The open question raised in docs/phase9.md is whether encoder LoRA
(130k trainable params) merely overfits the OE62 domain or genuinely generalizes
to molecules unlike its training set.

This script answers it with a COVARIATE-SHIFT split that still has GW labels so
MAE is computable:
  - train pool = low-MW molecules (MW <= q80 of the in-dist OE62 set),
  - OOD test   = high-MW tail (MW > q80) — also near scaffold-disjoint.
It is a proxy for "trained on smaller molecules, deployed on larger/stranger
commercial molecules".

Fair A/B on the SAME shifted split, all retrained on the low-MW pool only:
  raw B3LYP / const Δ / LightGBM Δ / encoder-LoRA (GPS+SchNet+Fusion).

Usage:
  .venv\\Scripts\\python.exe scripts/phase9/train_lora_ood_mwshift.py
  .venv\\Scripts\\python.exe scripts/phase9/train_lora_ood_mwshift.py --seeds 42 1 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch

from molgap.constants import MODELS_DIR, RESULTS_DIR, TARGET_COLS
from molgap.inference import load_hybrid

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
RDLogger.DisableLog("rdApp.*")

PHASE9 = RESULTS_DIR / "phase9"
GRAPH_CACHE = PHASE9 / "delta_oe62_graphs.pt"
NPZ = PHASE9 / "delta_oe62_embeddings.npz"
MW_QUANTILE = 0.8

LGB_PARAMS = dict(
    n_estimators=1500, learning_rate=0.02, num_leaves=31, max_depth=-1,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
    reg_lambda=2.0, min_child_samples=30, n_jobs=-1, verbose=-1,
)


# ── LoRA plumbing (mirrors train_encoder_lora_delta.py) ──────────────────────
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


def forward_batch(gps, schnet, fusion, g2d, g3d, indices, device):
    b2 = Batch.from_data_list([g2d[int(i)] for i in indices]).to(device)
    b3 = Batch.from_data_list([g3d[int(i)] for i in indices]).to(device)
    e2 = gps.encode(b2.x, b2.edge_index, b2.edge_attr, b2.batch)
    charges = b3.charges if hasattr(b3, "charges") else None
    e3 = schnet.encode(b3.z, b3.pos, b3.batch, charges=charges)
    return fusion(e2, e3)


def iter_batches(indices, batch_size, shuffle, rng):
    idx = np.array(indices, copy=True)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield idx[start:start + batch_size]


def metrics_block(y_true, y_pred):
    out = {}
    for i, t in enumerate(TARGET_COLS):
        out[t] = {
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
        }
    out["average"] = {
        "mae": float(np.mean([out[t]["mae"] for t in TARGET_COLS])),
        "r2": float(np.mean([out[t]["r2"] for t in TARGET_COLS])),
    }
    return out


# ── data ─────────────────────────────────────────────────────────────────────
def load_data():
    obj = torch.load(GRAPH_CACHE, weights_only=False)
    df, g2d, g3d = obj["df"], obj["g2d"], obj["g3d"]
    smiles = df["smiles"].tolist()
    mw = np.array([Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)

    # align frozen embeddings (for the LightGBM baseline) by SMILES
    npz = np.load(NPZ, allow_pickle=True)
    emb_by_smi = {s: np.hstack([npz["emb_2d"][i], npz["emb_3d"][i]]).astype(np.float32)
                  for i, s in enumerate(npz["smiles"])}
    X = np.stack([emb_by_smi[s] for s in smiles])  # [n, 384], df-order

    gw = df[[f"gw_{t}" for t in TARGET_COLS]].to_numpy(dtype=np.float32)
    raw = df[[f"pred_{t}" for t in TARGET_COLS]].to_numpy(dtype=np.float32)
    return df, g2d, g3d, mw, X, gw, raw


def mw_shift_split(mw, split_seed):
    thr = float(np.quantile(mw, MW_QUANTILE))
    pool = np.where(mw <= thr)[0]
    test = np.where(mw > thr)[0]
    train, val = train_test_split(pool, test_size=0.1, random_state=split_seed)
    return train, val, test, thr


# ── LoRA train/eval ────────────────────────────────────────────────────────────
def train_lora(g2d, g3d, gw, train, val, test, args, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    y = torch.tensor(gw, dtype=torch.float32)

    gps, schnet, fusion, _ = load_hybrid(device, key="phase7_hybrid")
    freeze(gps); freeze(schnet); freeze(fusion)
    counts = {
        "gps": inject_lora(gps, args.rank, args.alpha, args.lora_dropout),
        "schnet": inject_lora(schnet, args.rank, args.alpha, args.lora_dropout),
        "fusion": inject_lora(fusion, args.rank, args.alpha, args.lora_dropout),
    }
    gps.to(device); schnet.to(device); fusion.to(device)
    n_trainable = trainable_params(gps, schnet, fusion)

    params = [p for m in (gps, schnet, fusion) for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-6)
    crit = nn.L1Loss()

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        gps.train(); schnet.train(); fusion.train()
        for bi in iter_batches(train, args.batch_size, True, rng):
            pred = forward_batch(gps, schnet, fusion, g2d, g3d, bi, device)
            opt.zero_grad()
            crit(pred, y[bi].to(device)).backward()
            opt.step()
        gps.eval(); schnet.eval(); fusion.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            for bi in iter_batches(val, args.batch_size, False, rng):
                vl += crit(forward_batch(gps, schnet, fusion, g2d, g3d, bi, device),
                           y[bi].to(device)).item() * len(bi); vc += len(bi)
        vmae = vl / vc
        sched.step(vmae)
        if vmae < best_val:
            best_val = vmae
            best_state = {
                "gps": {k: v.detach().cpu().clone() for k, v in gps.state_dict().items()},
                "schnet": {k: v.detach().cpu().clone() for k, v in schnet.state_dict().items()},
                "fusion": {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()},
            }
            wait = 0
        else:
            wait += 1
        if wait >= args.patience:
            break

    gps.load_state_dict(best_state["gps"])
    schnet.load_state_dict(best_state["schnet"])
    fusion.load_state_dict(best_state["fusion"])
    gps.eval(); schnet.eval(); fusion.eval()
    preds = []
    with torch.no_grad():
        for bi in iter_batches(test, args.batch_size, False, rng):
            preds.append(forward_batch(gps, schnet, fusion, g2d, g3d, bi, device).cpu().numpy())
    return np.concatenate(preds, axis=0), n_trainable, counts, float(best_val)


def train_lightgbm_delta(X, gw, raw, train, val, test, seed):
    """LightGBM Δ on frozen embeddings, same MW-shift split."""
    delta = gw - raw
    preds = np.zeros((len(test), len(TARGET_COLS)), dtype=np.float32)
    fit_idx = np.concatenate([train, val])  # LightGBM has its own internal val
    for i, t in enumerate(TARGET_COLS):
        Xa, Xv, ya, yv = train_test_split(X[fit_idx], delta[fit_idx, i],
                                          test_size=0.1, random_state=seed)
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(Xa, ya, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        preds[:, i] = raw[test, i] + m.predict(X[test])
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    df, g2d, g3d, mw, X, gw, raw = load_data()
    train, val, test, thr = mw_shift_split(mw, args.split_seed)
    print(f"Device: {device}", flush=True)
    print(f"MW-shift split: thr(q{int(MW_QUANTILE*100)})={thr:.1f} | "
          f"train {len(train)} / val {len(val)} / OOD-test {len(test)}", flush=True)
    print(f"  train MW: {mw[np.concatenate([train,val])].min():.0f}-"
          f"{mw[np.concatenate([train,val])].max():.0f} | "
          f"OOD MW: {mw[test].min():.0f}-{mw[test].max():.0f}", flush=True)

    y_test = gw[test]
    raw_test = raw[test]
    const = raw_test + (gw[np.concatenate([train, val])] -
                        raw[np.concatenate([train, val])]).mean(axis=0, keepdims=True)

    blocks = {
        "raw_b3lyp": metrics_block(y_test, raw_test),
        "const_delta": metrics_block(y_test, const),
    }

    # LightGBM Δ baseline (deterministic-ish; use first seed)
    print("\nTraining LightGBM Δ on low-MW pool ...", flush=True)
    lgb_pred = train_lightgbm_delta(X, gw, raw, train, val, test, args.seeds[0])
    blocks["lightgbm_delta"] = metrics_block(y_test, lgb_pred)

    # encoder LoRA across seeds
    lora_preds, n_trainable, counts, best_vals = [], None, None, []
    for seed in args.seeds:
        t0 = time.time()
        print(f"\nTraining encoder-LoRA (gps+schnet+fusion) seed={seed} ...", flush=True)
        p, n_trainable, counts, bv = train_lora(g2d, g3d, gw, train, val, test, args, device, seed)
        best_vals.append(bv)
        lora_preds.append(p)
        b = metrics_block(y_test, p)
        print(f"  seed {seed}: avg MAE={b['average']['mae']:.4f} "
              f"R2={b['average']['r2']:.4f} (best_val={bv:.4f}, {time.time()-t0:.0f}s)", flush=True)
    lora_mean = np.mean(lora_preds, axis=0)
    blocks["encoder_lora_gw_mean"] = metrics_block(y_test, lora_mean)
    # per-seed for variance
    per_seed = [metrics_block(y_test, p)["average"]["mae"] for p in lora_preds]

    print("\n=== OOD (high-MW tail) GW MAE / R2 ===", flush=True)
    order = ["raw_b3lyp", "const_delta", "lightgbm_delta", "encoder_lora_gw_mean"]
    for key in order:
        b = blocks[key]
        print(f"{key:22s} avg MAE={b['average']['mae']:.4f} R2={b['average']['r2']:.4f} | "
              f"H {b['homo']['mae']:.4f} L {b['lumo']['mae']:.4f} G {b['gap']['mae']:.4f}",
              flush=True)
    if len(per_seed) > 1:
        print(f"  LoRA per-seed avg MAE: {np.mean(per_seed):.4f} ± {np.std(per_seed):.4f} "
              f"{[round(x,4) for x in per_seed]}", flush=True)

    result = {
        "split": "mw_shift", "mw_threshold": thr, "mw_quantile": MW_QUANTILE,
        "n_train": int(len(train)), "n_val": int(len(val)), "n_ood_test": int(len(test)),
        "rank": args.rank, "alpha": args.alpha, "seeds": args.seeds,
        "lora_trainable_params": n_trainable, "lora_layer_counts": counts,
        "lora_best_val_mae": best_vals, "lora_per_seed_avg_mae": per_seed,
        "metrics": blocks,
    }
    out = PHASE9 / "lora_ood_mwshift_metrics.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
