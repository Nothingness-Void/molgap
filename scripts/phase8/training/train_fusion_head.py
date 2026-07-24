"""
Train Phase 8 embedding fusion heads on aligned encoder embeddings.

The active 1M path uses `--head baseline` with one or two aligned 2D embedding
payloads. `--head moe` is retained only to reproduce the closed head-capacity
experiments; it is not a production candidate.

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/training/train_fusion_head.py
  .venv\\Scripts\\python.exe scripts/phase8/training/train_fusion_head.py --max-samples 2000 --epochs 3
  .venv\\Scripts\\python.exe scripts/phase8/training/train_fusion_head.py --head baseline --out results/phase8/fusion_replacement_300k_metrics.json
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
from torch.utils.data import DataLoader, TensorDataset

from molgap.constants import MODELS_DIR, RESULTS_DIR, SEED
from molgap.fusion import FusionHead, MoEFusionHead
from molgap.utils import ensure_dirs, load_aligned_encoder_embeddings

PHASE8_DIR = RESULTS_DIR / "phase8"
EMB_2D = PHASE8_DIR / "gps_replacement_300k_embeddings.pt"
EMB_3D = PHASE8_DIR / "schnet_replacement_300k_embeddings.pt"
GRAPH_3D = PHASE8_DIR / "pyg_3d_graphs_etkdg_replacement_300k.pt"


def make_split(n: int, max_samples: int | None):
    idx = np.random.RandomState(SEED).permutation(n)
    if max_samples is not None:
        idx = idx[:max_samples]
    n_train, n_val = int(0.8 * len(idx)), int(0.1 * len(idx))
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


def make_loader(h2, h3, y, idx, batch_size, shuffle, sample_weights=None):
    ds = TensorDataset(h2[idx], h3[idx], y[idx])
    if sample_weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights.to(dtype=torch.double),
            num_samples=len(ds),
            replacement=True,
            generator=torch.Generator().manual_seed(SEED),
        )
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)


def eval_metrics(model, h2, h3, y, idx, batch_size, device):
    loader = make_loader(h2, h3, y, idx, batch_size, False)
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for b2, b3, by in loader:
            pred.append(model(b2.to(device), b3.to(device)).float().cpu().numpy())
            true.append(by.numpy())
    pred, true = np.concatenate(pred), np.concatenate(true)
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


def train_one(model, h2, h3, y, split, args, device, train_sample_weights=None):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-6)
    crit = nn.L1Loss()
    train_loader = make_loader(
        h2, h3, y, split["train"], args.batch_size, True,
        sample_weights=train_sample_weights,
    )
    val_loader = make_loader(h2, h3, y, split["val"], 2048, False)
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
            for b2, b3, by in val_loader:
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
        log_rows.append({"epoch": epoch, "val_mae": val, "time_s": time.time() - t0})
        print(f"  ep{epoch:03d} val={val:.4f} best={best_val:.4f}@{best_epoch}", flush=True)
        if wait >= args.patience:
            break
    model.load_state_dict(best_state)
    metrics = eval_metrics(model, h2, h3, y, split["test"], 2048, device)
    metrics["best_val_mae"] = float(best_val)
    metrics["best_epoch"] = int(best_epoch)
    metrics["n_params"] = int(sum(p.numel() for p in model.parameters()))
    metrics["log"] = log_rows
    return model, metrics


def make_replay_weights(source_idx, train_idx, boundary: int | None, old_weight: float):
    """Return per-train-row sampling weights for an old-cache replay prefix."""
    if boundary is None:
        return None, None
    if old_weight <= 0.0:
        raise ValueError("--replay-weight must be positive")
    train_source = source_idx[train_idx].view(-1).cpu().numpy()
    old = train_source < boundary
    if not old.any() or old.all():
        raise ValueError("Replay boundary must split the fusion training rows into old and new pools")
    weights = torch.as_tensor(np.where(old, old_weight, 1.0), dtype=torch.double)
    report = {
        "source_idx_lt": int(boundary),
        "old_train_rows": int(old.sum()),
        "new_train_rows": int((~old).sum()),
        "old_weight": float(old_weight),
        "expected_old_draw_fraction": float(weights[old].sum().item() / weights.sum().item()),
    }
    return weights, report


def main():
    parser = argparse.ArgumentParser(description="Phase 8 baseline/MoE fusion A/B")
    parser.add_argument("--emb-2d", type=Path, default=EMB_2D)
    parser.add_argument("--emb-2d-extra", type=Path, default=None,
                        help="optional second aligned 2D embedding payload to concatenate")
    parser.add_argument("--emb-3d", type=Path, default=EMB_3D)
    parser.add_argument("--graphs-3d", type=Path, default=GRAPH_3D)
    parser.add_argument("--graphs-3d-extra", type=Path, default=None,
                        help="optional non-overlapping second 3D graph cache")
    parser.add_argument("--head", choices=["baseline", "moe", "both"], default="both")
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5.4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--replay-boundary", type=int, default=None,
                        help="source_idx below this value is the replay pool; disabled by default")
    parser.add_argument("--replay-weight", type=float, default=1.0,
                        help="relative sampling weight for rows below --replay-boundary")
    parser.add_argument("--out", type=Path, default=PHASE8_DIR / "moe_replacement_300k_metrics.json")
    parser.add_argument("--baseline-model-out", type=Path,
                        default=MODELS_DIR / "phase8_hybrid_fusion_replacement_300k.pt")
    parser.add_argument("--moe-model-out", type=Path, default=None)
    args = parser.parse_args()

    ensure_dirs(PHASE8_DIR, MODELS_DIR)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    emb_2d_paths = [args.emb_2d] + ([args.emb_2d_extra] if args.emb_2d_extra else [])
    graph_paths = [args.graphs_3d] + ([args.graphs_3d_extra] if args.graphs_3d_extra else [])
    h2, h3, y, source_idx = load_aligned_encoder_embeddings(emb_2d_paths, args.emb_3d, graph_paths)
    split = make_split(h2.shape[0], args.max_samples)
    replay_weights, replay_report = make_replay_weights(
        source_idx, split["train"], args.replay_boundary, args.replay_weight,
    )
    print(f"Aligned N={h2.shape[0]} split={len(split['train'])}/{len(split['val'])}/{len(split['test'])}",
          flush=True)
    if replay_report is not None:
        print(f"Replay sampling: {replay_report}", flush=True)

    suffix = f"_n{args.max_samples}" if args.max_samples else ""
    result = {
        "n_aligned": int(h2.shape[0]),
        "max_samples": args.max_samples,
        "source_idx_min": int(source_idx.min().item()),
        "source_idx_max": int(source_idx.max().item()),
        "head": args.head,
        "embedding_dims": {"2d": int(h2.shape[1]), "3d": int(h3.shape[1])},
        "emb_2d_extra": str(args.emb_2d_extra) if args.emb_2d_extra else None,
        "graphs_3d": [str(path) for path in graph_paths],
        "replay_sampling": replay_report,
    }
    if args.head in {"baseline", "both"}:
        print("\nBaseline FusionHead", flush=True)
        base, base_metrics = train_one(
            FusionHead(
                "gate", args.hidden, 0.0,
                dim_2d=int(h2.shape[1]), dim_3d=int(h3.shape[1]),
            ),
            h2, h3, y, split, args, device, replay_weights,
        )
        baseline_out = (
            args.baseline_model_out
            if not suffix
            else MODELS_DIR / f"phase8_hybrid_fusion_baseline{suffix}.pt"
        )
        torch.save(base.state_dict(), baseline_out)
        result["baseline"] = base_metrics
        result["baseline_model"] = str(baseline_out)
    if args.head in {"moe", "both"}:
        print("\nMoEFusionHead", flush=True)
        moe, moe_metrics = train_one(
            MoEFusionHead(args.hidden, 0.0, n_experts=args.experts),
            h2, h3, y, split, args, device, replay_weights,
        )
        moe_out = args.moe_model_out or MODELS_DIR / f"phase8_hybrid_moe_e{args.experts}{suffix}.pt"
        torch.save(moe.state_dict(), moe_out)
        result["moe"] = moe_metrics
        result["moe_model"] = str(moe_out)
    if "baseline" in result and "moe" in result:
        result["delta_average_mae"] = float(
            result["moe"]["average"]["mae"] - result["baseline"]["average"]["mae"]
        )
        result["delta_gap_mae"] = float(result["moe"]["Gap"]["mae"] - result["baseline"]["Gap"]["mae"])
    elif "baseline" in result:
        result["delta_average_mae"] = None
        result["delta_gap_mae"] = None
    else:
        result["delta_average_mae"] = None
        result["delta_gap_mae"] = None
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nMetrics -> {args.out}", flush=True)
    if "baseline" in result:
        print(
            f"Gap MAE baseline={result['baseline']['Gap']['mae']:.4f} "
            f"avg={result['baseline']['average']['mae']:.4f}",
            flush=True,
        )
    if "moe" in result:
        print(
            f"Gap MAE moe={result['moe']['Gap']['mae']:.4f} "
            f"avg={result['moe']['average']['mae']:.4f}",
            flush=True,
        )
    if result["delta_gap_mae"] is not None:
        print(f"MoE delta Gap={result['delta_gap_mae']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
