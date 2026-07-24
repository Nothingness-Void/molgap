"""Train a GPS7+GPS9-only late-fusion head from frozen 2D embeddings.

This is a controlled data-mixture gate before allocating 3D compute.  The head
sees no coordinates, SchNet embedding, or 3D graph cache.  Data-only v1/v2
comparisons must use uniform sampling.  Targeted rescue runs may explicitly
warm-start and replay-weight an appended suffix, which is recorded in metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from molgap.constants import MODELS_DIR, RESULTS_DIR, SEED, TARGET_COLS
from molgap.fusion import DualGPSFusionHead
from molgap.utils import ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gps7-emb", type=Path, required=True)
    parser.add_argument("--gps9-emb", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True,
                        help="Training CSV used to build the two 2D caches.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5.4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--replay-boundary", type=int, default=None)
    parser.add_argument("--replay-weight", type=float, default=1.0)
    return parser.parse_args()


def atomic_torch_save(value: object, path: Path) -> None:
    ensure_dirs(path.parent)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_write(value: dict, path: Path) -> None:
    ensure_dirs(path.parent)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_labels(csv_path: Path) -> torch.Tensor:
    table = pd.read_csv(csv_path)
    for column in TARGET_COLS:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna(subset=["smiles", *TARGET_COLS])
    table = table.loc[table["gap"] > 0].reset_index(drop=True)
    return torch.from_numpy(table[TARGET_COLS].to_numpy(dtype=np.float32, copy=True))


def load_embeddings(gps7_path: Path, gps9_path: Path, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gps7 = torch.load(gps7_path, map_location="cpu", weights_only=False)
    gps9 = torch.load(gps9_path, map_location="cpu", weights_only=False)
    h7, source7 = gps7["embeddings"].float(), gps7["source_idx"].long()
    h9, source9 = gps9["embeddings"].float(), gps9["source_idx"].long()
    if h7.ndim != 2 or h9.ndim != 2 or h7.shape[1] != 192 or h9.shape[1] != 192:
        raise ValueError(f"Expected two (N, 192) embedding tensors, got {tuple(h7.shape)} and {tuple(h9.shape)}")
    if not torch.equal(source7, source9):
        raise ValueError("GPS7/GPS9 source_idx differs")
    expected = torch.arange(len(labels), dtype=torch.long)
    if not torch.equal(source7, expected):
        raise ValueError("Embedding source_idx does not cover the filtered CSV rows exactly")
    return h7, h9, labels


def make_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(("HOMO", "LUMO", "Gap")):
        result[name] = {
            "mae_eV": float(mean_absolute_error(target[:, index], prediction[:, index])),
            "r2": float(r2_score(target[:, index], prediction[:, index])),
        }
    result["average"] = {
        "mae_eV": float(np.mean([result[name]["mae_eV"] for name in ("HOMO", "LUMO", "Gap")])),
        "r2": float(np.mean([result[name]["r2"] for name in ("HOMO", "LUMO", "Gap")])),
    }
    return result


def evaluate(model, h7, h9, labels, indices, batch_size, device) -> dict:
    loader = DataLoader(TensorDataset(h7[indices], h9[indices], labels[indices]), batch_size=batch_size, shuffle=False)
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch7, batch9, target in loader:
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                prediction = model(batch7.to(device), batch9.to(device))
            predictions.append(prediction.float().cpu().numpy())
            targets.append(target.numpy())
    return make_metrics(np.concatenate(predictions), np.concatenate(targets))


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out.parent, args.model_out.parent, args.checkpoint_out.parent, MODELS_DIR, RESULTS_DIR / "phase8")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_labels(args.csv)
    h7, h9, labels = load_embeddings(args.gps7_emb, args.gps9_emb, labels)
    permutation = np.random.RandomState(SEED).permutation(len(labels))
    n_train, n_val = int(0.8 * len(labels)), int(0.1 * len(labels))
    split = {
        "train": torch.from_numpy(permutation[:n_train]).long(),
        "val": torch.from_numpy(permutation[n_train:n_train + n_val]).long(),
        "test": torch.from_numpy(permutation[n_train + n_val:]).long(),
    }
    print(f"Device={device}; aligned={len(labels):,}; split={n_train:,}/{n_val:,}/{len(split['test']):,}", flush=True)

    model = DualGPSFusionHead(hidden=args.hidden).to(device)
    if args.init_from is not None and args.resume_from is not None:
        raise ValueError("--init-from and --resume-from are mutually exclusive")
    if args.init_from is not None:
        model.load_state_dict(torch.load(args.init_from, map_location=device, weights_only=True))
        print(f"Warm-started from {args.init_from}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    criterion = nn.L1Loss()
    start_epoch, best_val, best_epoch, wait, best_state, log = 0, float("inf"), -1, 0, None, []
    if args.resume_from is not None:
        state = torch.load(args.resume_from, map_location=device, weights_only=False)
        if state.get("tag") != args.tag or state.get("n_aligned") != len(labels):
            raise ValueError("Resume checkpoint does not match this data/head run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_val, best_epoch, wait = state["next_epoch"], state["best_val"], state["best_epoch"], state["wait"]
        best_state, log = state["best_state"], state["log"]
        print(f"Resuming from epoch {start_epoch}", flush=True)

    sampler = None
    sampling: dict[str, object] = {"mode": "uniform"}
    if args.replay_boundary is not None:
        if args.replay_weight <= 0:
            raise ValueError("--replay-weight must be positive")
        old = split["train"] < args.replay_boundary
        if not old.any() or old.all():
            raise ValueError("Replay boundary must split the head training rows")
        weights = torch.where(
            old,
            torch.full_like(old, args.replay_weight, dtype=torch.float64),
            torch.ones_like(old, dtype=torch.float64),
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        old_mass = float(old.sum()) * args.replay_weight
        new_mass = float((~old).sum())
        sampling = {
            "mode": "weighted_suffix",
            "source_idx_lt": args.replay_boundary,
            "old_train_rows": int(old.sum()),
            "new_train_rows": int((~old).sum()),
            "old_weight": args.replay_weight,
            "expected_new_draw_fraction": new_mass / (old_mass + new_mass),
        }
        print(f"Replay sampling: {sampling}", flush=True)
    train_loader = DataLoader(
        TensorDataset(h7[split["train"]], h9[split["train"]], labels[split["train"]]),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        pin_memory=True,
        num_workers=0,
    )
    val_loader = DataLoader(TensorDataset(h7[split["val"]], h9[split["val"]], labels[split["val"]]),
                            batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=0)
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        model.train()
        total, count = 0.0, 0
        for batch7, batch9, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss = criterion(model(batch7.to(device, non_blocking=True), batch9.to(device, non_blocking=True)), target.to(device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(target)
            count += len(target)
        model.eval()
        val_total, val_count = 0.0, 0
        with torch.no_grad():
            for batch7, batch9, target in val_loader:
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    loss = criterion(model(batch7.to(device, non_blocking=True), batch9.to(device, non_blocking=True)), target.to(device, non_blocking=True))
                val_total += loss.item() * len(target)
                val_count += len(target)
        val_mae = val_total / max(val_count, 1)
        scheduler.step(val_mae)
        if val_mae < best_val:
            best_val, best_epoch, wait = val_mae, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
        row = {"epoch": epoch, "train_mae": total / max(count, 1), "val_mae": val_mae, "best_val_mae": best_val,
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - started}
        log.append(row)
        atomic_torch_save({
            "tag": args.tag, "n_aligned": len(labels), "next_epoch": epoch + 1, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_val": best_val, "best_epoch": best_epoch, "wait": wait, "best_state": best_state,
            "sampling": sampling, "log": log,
        }, args.checkpoint_out)
        atomic_json_write({"complete": False, "tag": args.tag, "n_aligned": len(labels), "best_val_mae_eV": best_val,
                           "best_epoch": best_epoch, "log": log}, args.out)
        print(f"ep{epoch:03d} train={row['train_mae']:.5f} val={val_mae:.5f} best={best_val:.5f}@{best_epoch} {row['seconds']:.1f}s{' *' if wait == 0 else ''}", flush=True)
        if wait >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite checkpoint was produced")
    model.load_state_dict(best_state)
    atomic_torch_save(best_state, args.model_out)
    result = {
        "tag": args.tag, "n_aligned": len(labels), "split": {"seed": SEED, **{name: len(indices) for name, indices in split.items()}},
        "sampling": sampling, "embedding_dims": {"gps7": int(h7.shape[1]), "gps9": int(h9.shape[1])},
        "best_val_mae_eV": best_val, "best_epoch": best_epoch, "test_metrics": evaluate(model, h7, h9, labels, split["test"], args.eval_batch_size, device),
        "gps7_embedding": str(args.gps7_emb), "gps9_embedding": str(args.gps9_emb), "csv": str(args.csv), "log": log,
    }
    atomic_json_write(result, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
