"""Train a Phase 8 v3 + tail-pool fusion-head probe.

This is a cheap decision run after fetching the residual-tail pool. It freezes
the v3 GPS/SchNet encoders, concatenates existing expansion500k embeddings with
tail-pool embeddings extracted from the same encoders, and trains only the
standard FusionHead. If this head-only top-up has no common-eval signal, a much
more expensive encoder-level retrain is unlikely to be a good next step.
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
from molgap.fusion import FusionHead
from molgap.utils import ensure_dirs

PHASE8_DIR = RESULTS_DIR / "phase8"
TARGETS = ("HOMO", "LUMO", "Gap")


def _load_payload(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, weights_only=False)
    return payload["embeddings"].float(), payload["source_idx"].long()


def _labels_by_source(graph_path: Path, offset: int = 0) -> dict[int, torch.Tensor]:
    graphs = torch.load(graph_path, weights_only=False)
    labels: dict[int, torch.Tensor] = {}
    for g in graphs:
        source_idx = int(g.source_idx.view(-1)[0].item()) + offset
        labels[source_idx] = g.y.squeeze(0).float()
    return labels


def _offset_payload(h: torch.Tensor, idx: torch.Tensor, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    if offset == 0:
        return h, idx
    return h, idx + int(offset)


def load_aligned(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_h2, base_i2 = _load_payload(args.base_emb_2d)
    base_h3, base_i3 = _load_payload(args.base_emb_3d)
    tail_h2, tail_i2 = _load_payload(args.tail_emb_2d)
    tail_h3, tail_i3 = _load_payload(args.tail_emb_3d)

    base_labels = _labels_by_source(args.base_graph_3d)
    tail_labels = _labels_by_source(args.tail_graph_3d, offset=args.tail_offset)
    labels = {**base_labels, **tail_labels}

    tail_h2, tail_i2 = _offset_payload(tail_h2, tail_i2, args.tail_offset)
    tail_h3, tail_i3 = _offset_payload(tail_h3, tail_i3, args.tail_offset)

    h2 = torch.cat([base_h2, tail_h2], dim=0)
    i2 = torch.cat([base_i2, tail_i2], dim=0)
    h3 = torch.cat([base_h3, tail_h3], dim=0)
    i3 = torch.cat([base_i3, tail_i3], dim=0)

    pos2 = {int(v): i for i, v in enumerate(i2.tolist())}
    pos3 = {int(v): i for i, v in enumerate(i3.tolist())}
    common = sorted(set(pos2).intersection(pos3).intersection(labels))
    ii2 = torch.tensor([pos2[i] for i in common], dtype=torch.long)
    ii3 = torch.tensor([pos3[i] for i in common], dtype=torch.long)
    y = torch.stack([labels[i] for i in common])
    return h2[ii2], h3[ii3], y, torch.tensor(common, dtype=torch.long)


def make_split(n: int, max_samples: int | None) -> dict[str, np.ndarray]:
    idx = np.random.RandomState(SEED).permutation(n)
    if max_samples is not None:
        idx = idx[:max_samples]
    n_train, n_val = int(0.8 * len(idx)), int(0.1 * len(idx))
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


def _loader(h2, h3, y, idx, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(h2[idx], h3[idx], y[idx]),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,
    )


@torch.no_grad()
def eval_metrics(model, h2, h3, y, idx, batch_size: int, device: torch.device) -> dict:
    model.eval()
    pred, true = [], []
    for b2, b3, by in _loader(h2, h3, y, idx, batch_size, False):
        pred.append(model(b2.to(device), b3.to(device)).float().cpu().numpy())
        true.append(by.numpy())
    pred_arr = np.concatenate(pred)
    true_arr = np.concatenate(true)
    out = {}
    for i, name in enumerate(TARGETS):
        out[name] = {
            "mae": float(mean_absolute_error(true_arr[:, i], pred_arr[:, i])),
            "r2": float(r2_score(true_arr[:, i], pred_arr[:, i])),
        }
    out["average"] = {
        "mae": float(np.mean([out[t]["mae"] for t in TARGETS])),
        "r2": float(np.mean([out[t]["r2"] for t in TARGETS])),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-emb-2d", type=Path, default=PHASE8_DIR / "gps_expansion_500k_embeddings.pt")
    parser.add_argument("--base-emb-3d", type=Path, default=PHASE8_DIR / "schnet_expansion_500k_embeddings.pt")
    parser.add_argument("--base-graph-3d", type=Path, default=PHASE8_DIR / "pyg_3d_graphs_etkdg_expansion_500k.pt")
    parser.add_argument("--tail-emb-2d", type=Path, default=PHASE8_DIR / "gps_tail_probe_30k_embeddings.pt")
    parser.add_argument("--tail-emb-3d", type=Path, default=PHASE8_DIR / "schnet_tail_probe_30k_embeddings.pt")
    parser.add_argument("--tail-graph-3d", type=Path, default=PHASE8_DIR / "pyg_3d_graphs_etkdg_tail_probe_30k.pt")
    parser.add_argument("--tail-offset", type=int, default=1_000_000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5.4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", type=Path, default=PHASE8_DIR / "fusion_tail_probe_30k_metrics.json")
    parser.add_argument("--model-out", type=Path, default=MODELS_DIR / "phase8_hybrid_fusion_tail_probe_30k.pt")
    args = parser.parse_args()

    ensure_dirs(PHASE8_DIR, MODELS_DIR)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    h2, h3, y, source_idx = load_aligned(args)
    split = make_split(h2.shape[0], args.max_samples)
    n_tail = int((source_idx >= args.tail_offset).sum().item())
    print(
        f"Aligned N={h2.shape[0]} tail_aligned={n_tail} "
        f"split={len(split['train'])}/{len(split['val'])}/{len(split['test'])}",
        flush=True,
    )

    model = FusionHead("gate", args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-6)
    crit = nn.L1Loss()
    train_loader = _loader(h2, h3, y, split["train"], args.batch_size, True)
    val_loader = _loader(h2, h3, y, split["val"], 2048, False)

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
        val = total / max(1, n)
        sched.step(val)
        improved = val < best_val
        if improved:
            best_val = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        log_rows.append({"epoch": epoch, "val_mae": float(val), "time_s": time.time() - t0})
        mark = " *" if improved else ""
        print(f"ep{epoch:03d} val={val:.5f} best={best_val:.5f}@{best_epoch}{mark}", flush=True)
        if wait >= args.patience:
            print(f"Early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No best state produced")
    model.load_state_dict(best_state)
    torch.save(best_state, args.model_out)
    test = eval_metrics(model, h2, h3, y, split["test"], 2048, device)

    result = {
        "kind": "tail_probe_fusion_head_only",
        "base": "phase8_expansion_500k",
        "tail": "phase8_tail_probe_30k",
        "n_aligned": int(h2.shape[0]),
        "n_tail_aligned": n_tail,
        "max_samples": args.max_samples,
        "best_val_mae": float(best_val),
        "best_epoch": int(best_epoch),
        "test_metrics": test,
        "model_path": str(args.model_out),
        "log": log_rows,
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Model -> {args.model_out}", flush=True)
    print(f"Metrics -> {args.out}", flush=True)
    print(
        f"Test avg MAE={test['average']['mae']:.5f} Gap MAE={test['Gap']['mae']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
