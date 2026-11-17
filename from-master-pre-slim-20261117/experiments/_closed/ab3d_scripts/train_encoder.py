"""A/B step 3: train one 3D encoder end-to-end on the 10k subset.

Controlled comparison — only the 3D encoder changes; subset, scaffold split,
training budget, optimizer/loss/schedule are fixed across arms. Each encoder is
trained end-to-end with its own head (gives a leak-free standalone 3D metric, the
PRIMARY discriminator), then its pooled embeddings are extracted for the fusion
step. Resource cost (wall-clock, sec/epoch, peak GPU memory, param count) is the
"运行压力 / 运行时间" deliverable.

Outputs (results/ab3d/):
  encoder_<name>.json   resources + standalone test MAE/R²
  emb_3d_<name>.pt      [n, hidden] float32, row-matched to graphs

Usage:
  .venv\\Scripts\\python.exe scripts/ab3d/train_encoder.py schnet
  .venv\\Scripts\\python.exe scripts/ab3d/train_encoder.py visnet
  .venv\\Scripts\\python.exe scripts/ab3d/train_encoder.py tensornet
"""
from __future__ import annotations

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from molgap.constants import AB_ENCODERS, RESULTS_DIR, SEED
from molgap.utils import regression_metrics

OUT = RESULTS_DIR / "ab3d"
MAX_EPOCHS = 120
PATIENCE = 20
# Per-encoder default batch (bf16, fits 8 GB with headroom): SchNet/ViSNet 64,
# TensorNet 32 (its [edges,3,3,F] tensors are memory-heavier).
DEFAULT_BATCH = {"schnet": 64, "visnet": 64, "tensornet": 32}
LR = 5e-4
WD = 1e-4
CLIP = 10.0


def build_encoder(name):
    spec = AB_ENCODERS[name]
    kind, params, use_charges = spec["kind"], spec["params"], spec["use_charges"]
    if kind == "schnet":
        from molgap.schnet import SchNetWrapper
        return SchNetWrapper(**params, use_charges=use_charges), use_charges
    if kind == "visnet":
        from molgap.visnet import ViSNetWrapper
        return ViSNetWrapper(**params, use_charges=use_charges), use_charges
    if kind == "tensornet":
        from molgap.tensornet import TensorNetWrapper
        return TensorNetWrapper(**params, use_charges=use_charges), use_charges
    raise ValueError(f"unknown encoder kind: {kind}")


def run_batch(model, b, use_charges):
    charges = b.charges if (use_charges and hasattr(b, "charges")) else None
    return model(b.z, b.pos, b.batch, charges=charges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(AB_ENCODERS.keys()))
    ap.add_argument("--batch", type=int, default=None,
                    help="override per-encoder default batch size")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="wall-clock training cap; stop after this many minutes")
    args = ap.parse_args()
    name = args.name

    from torch_geometric.loader import DataLoader

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 autocast (NOT fp16): Blackwell-native, ~2.3x faster than fp32 on the
    # RTX 5060, and bf16's fp32-range exponent keeps the equivariant/tensor norms
    # numerically safe (fp16 would overflow). bf16 needs no GradScaler.
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    print(f"[{name}] device={device} amp={use_amp} ({amp_dtype})")

    graphs = torch.load(str(OUT / "graphs_3d.pt"), weights_only=False)
    split = json.loads((OUT / "split.json").read_text(encoding="utf-8"))
    tr = [graphs[i] for i in split["train"]]
    va = [graphs[i] for i in split["val"]]
    te = [graphs[i] for i in split["test"]]
    print(f"[{name}] 3D graphs {len(graphs)} | train/val/test {len(tr)}/{len(va)}/{len(te)}")

    model, use_charges = build_encoder(name)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] params={n_params:,} use_charges={use_charges}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1e-6)
    crit = nn.L1Loss()
    # bf16 needs no loss scaling (fp32-range exponent), so no GradScaler.

    bs = args.batch if args.batch else DEFAULT_BATCH[name]
    max_seconds = args.max_minutes * 60 if args.max_minutes else None
    tl = DataLoader(tr, batch_size=bs, shuffle=True)
    vl = DataLoader(va, batch_size=bs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_val, best_state, wait, epochs_ran = float("inf"), None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(MAX_EPOCHS):
        epochs_ran = epoch + 1
        model.train()
        for b in tl:
            b = b.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = crit(run_batch(model, b, use_charges).float(), b.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()

        model.eval()
        vsum, vn = 0.0, 0
        with torch.no_grad():
            for b in vl:
                b = b.to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    out = run_batch(model, b, use_charges)
                vsum += crit(out.float(), b.y).item() * b.num_graphs
                vn += b.num_graphs
        vmae = vsum / vn
        sched.step(vmae)
        if vmae < best_val - 1e-5:
            best_val, best_state, wait = vmae, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if epoch % 5 == 0 or wait == 0:
            print(f"[{name}] epoch {epoch:3d} val_MAE {vmae:.4f} (best {best_val:.4f})")
        if wait >= PATIENCE:
            print(f"[{name}] early stop at epoch {epoch}")
            break
        if max_seconds and (time.perf_counter() - t0) >= max_seconds:
            print(f"[{name}] wall-clock cap hit at epoch {epoch} "
                  f"({(time.perf_counter()-t0)/60:.1f} min)")
            break

    train_seconds = time.perf_counter() - t0
    peak_train_mem = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else None

    if best_state is None:
        print(f"[{name}] WARNING: no epoch improved val (likely diverged to NaN); "
              f"keeping last weights for diagnostics.")
    else:
        model.load_state_dict(best_state)
    model.eval()

    # Standalone test metrics (primary discriminator).
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    P, T = [], []
    with torch.no_grad():
        for b in DataLoader(te, batch_size=bs):
            b = b.to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = run_batch(model, b, use_charges)
            P.append(out.float().cpu().numpy())
            T.append(b.y.cpu().numpy())
    metrics = regression_metrics(np.concatenate(T), np.concatenate(P),
                                 targets=["HOMO", "LUMO", "Gap"])
    peak_infer_mem = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else None

    # Extract embeddings for ALL 10k (row order = graphs order).
    emb = []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=bs):
            b = b.to(device)
            charges = b.charges if (use_charges and hasattr(b, "charges")) else None
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                e = model.encode(b.z, b.pos, b.batch, charges=charges)
            emb.append(e.float().cpu())
    emb = torch.cat(emb)
    torch.save(emb, str(OUT / f"emb_3d_{name}.pt"))

    result = {
        "encoder": name,
        "n_params": n_params,
        "use_charges": use_charges,
        "batch_size": bs,
        "epochs_ran": epochs_ran,
        "train_seconds": round(train_seconds, 1),
        "sec_per_epoch": round(train_seconds / max(1, epochs_ran), 2),
        "peak_train_mem_mb": round(peak_train_mem, 1) if peak_train_mem else None,
        "peak_infer_mem_mb": round(peak_infer_mem, 1) if peak_infer_mem else None,
        "best_val_mae": round(best_val, 4),
        "emb_dim": int(emb.shape[1]),
        "test": metrics,
    }
    (OUT / f"encoder_{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{name}] DONE  Gap MAE {metrics['Gap']['mae']:.4f} R² {metrics['Gap']['r2']:.4f} | "
          f"{train_seconds:.0f}s, {result['sec_per_epoch']}s/ep, peak {peak_train_mem}MB, "
          f"emb_dim {emb.shape[1]}")


if __name__ == "__main__":
    main()
