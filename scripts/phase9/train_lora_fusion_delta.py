"""
Phase 9: LoRA-adapt the Phase 7 FusionHead from B3LYP to GW on OE62.

This keeps the B3LYP production model intact:
  - load Phase 7 Hybrid FusionHead weights;
  - freeze all original Linear weights;
  - inject low-rank LoRA adapters into FusionHead Linear layers;
  - train only LoRA parameters on OE62 GW labels using frozen 2D/3D embeddings.

It is a fast feasibility test for model-side Δ adaptation. The comparison target
is the existing LightGBM Δ model in results/phase9/delta_model_metrics.json.

Usage:
  .venv\\Scripts\\python.exe scripts/phase9/train_lora_fusion_delta.py --rank 8
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
from torch.utils.data import DataLoader, TensorDataset

from molgap.constants import MODELS_DIR, RESULTS_DIR, TARGET_COLS
from molgap.inference import load_hybrid
from molgap.utils import murcko_scaffold_smiles

PHASE9 = RESULTS_DIR / "phase9"
CSV = PHASE9 / "delta_oe62.csv"
NPZ = PHASE9 / "delta_oe62_embeddings.npz"
DEFAULT_CKPT = MODELS_DIR / "hybrid_fusion_lora_gw_r8.pt"
DEFAULT_METRICS = PHASE9 / "lora_fusion_delta_metrics.json"
DEFAULT_PREDS = PHASE9 / "lora_fusion_delta_predictions.csv"
SEED = 42
TEST_FRAC = 0.2


class LoRALinear(nn.Module):
    """Frozen Linear layer plus trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_b(self.lora_a(self.dropout(x)))


def inject_lora(module: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    """Recursively replace Linear children with LoRALinear. Returns layer count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            count += 1
        else:
            count += inject_lora(child, rank=rank, alpha=alpha, dropout=dropout)
    return count


def resolve_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    df = pd.read_csv(CSV)
    npz = np.load(NPZ, allow_pickle=True)
    e2d = torch.tensor(npz["emb_2d"], dtype=torch.float32)
    e3d = torch.tensor(npz["emb_3d"], dtype=torch.float32)
    smiles = npz["smiles"]
    if not (smiles == df["smiles"].to_numpy()).all():
        raise RuntimeError("smiles order mismatch between csv and npz")
    y_gw = torch.tensor(df[[f"gw_{t}" for t in TARGET_COLS]].values, dtype=torch.float32)
    pred_b3 = df[[f"pred_{t}" for t in TARGET_COLS]].values.astype(np.float32)
    gw = y_gw.numpy()
    return df, e2d, e3d, y_gw, pred_b3, gw


def scaffold_split(smiles: list[str]):
    scaffolds = [murcko_scaffold_smiles(s) or "NONE" for s in smiles]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    all_idx = np.arange(len(smiles))
    train_val, test = next(gss.split(all_idx, groups=scaffolds))
    train, val = train_test_split(train_val, test_size=0.1, random_state=SEED)
    return train, val, test, len(set(scaffolds))


def make_loader(e2d, e3d, y, idx, batch_size, shuffle):
    ds = TensorDataset(e2d[idx], e3d[idx], y[idx])
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


def count_trainable(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def evaluate(model, e2d, e3d, y_true, idx, device, batch_size=1024):
    model.eval()
    preds = []
    loader = make_loader(e2d, e3d, y_true, idx, batch_size, False)
    with torch.no_grad():
        for h2, h3, _ in loader:
            preds.append(model(h2.to(device), h3.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--metrics", type=str, default=str(DEFAULT_METRICS))
    parser.add_argument("--predictions", type=str, default=str(DEFAULT_PREDS))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = resolve_device(args.device)
    print(f"Device: {device}", flush=True)

    df, e2d, e3d, y_gw, pred_b3, gw = load_data()
    train, val, test, n_scaffolds = scaffold_split(df["smiles"].tolist())
    print(
        f"OE62 GW: n={len(df)} scaffolds={n_scaffolds} "
        f"train/val/test={len(train)}/{len(val)}/{len(test)}",
        flush=True,
    )

    _, _, fusion, _ = load_hybrid(device, key="phase7_hybrid")
    for p in fusion.parameters():
        p.requires_grad_(False)
    n_lora = inject_lora(fusion, rank=args.rank, alpha=args.alpha, dropout=args.lora_dropout)
    fusion.to(device)
    trainable = count_trainable(fusion)
    print(f"Injected LoRA into {n_lora} Linear layers; trainable params={trainable:,}", flush=True)

    opt = torch.optim.AdamW(
        [p for p in fusion.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=12, factor=0.5, min_lr=1e-6
    )
    crit = nn.L1Loss()
    train_loader = make_loader(e2d, e3d, y_gw, train, args.batch_size, True)
    val_loader = make_loader(e2d, e3d, y_gw, val, 1024, False)

    best_val = float("inf")
    best_state = None
    wait = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        fusion.train()
        tr_loss = 0.0
        tr_count = 0
        for h2, h3, y in train_loader:
            h2, h3, y = h2.to(device), h3.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(fusion(h2, h3), y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * y.size(0)
            tr_count += y.size(0)

        fusion.eval()
        va_loss = 0.0
        va_count = 0
        with torch.no_grad():
            for h2, h3, y in val_loader:
                h2, h3, y = h2.to(device), h3.to(device), y.to(device)
                va_loss += crit(fusion(h2, h3), y).item() * y.size(0)
                va_count += y.size(0)
        train_mae = tr_loss / tr_count
        val_mae = va_loss / va_count
        sched.step(val_mae)

        improved = val_mae < best_val
        if improved:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()}
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
        raise RuntimeError("No best LoRA state captured")
    fusion.load_state_dict(best_state)
    pred_lora = evaluate(fusion, e2d, e3d, y_gw, test, device)
    y_test = gw[test]
    raw = pred_b3[test]
    delta_train = gw[train] - pred_b3[train]
    const = raw + delta_train.mean(axis=0, keepdims=True)

    blocks = {
        "raw_b3lyp": metrics_block(y_test, raw),
        "const_delta": metrics_block(y_test, const),
        "lora_gw": metrics_block(y_test, pred_lora),
    }
    lgbm_path = PHASE9 / "delta_model_metrics.json"
    if lgbm_path.exists():
        blocks["lightgbm_delta_reference"] = json.loads(lgbm_path.read_text())

    print("\nScaffold-test GW MAE/R2")
    for name in ["raw_b3lyp", "const_delta", "lora_gw"]:
        b = blocks[name]
        print(
            f"{name:12s} avg MAE={b['average']['mae']:.4f} R2={b['average']['r2']:.4f} | "
            f"H {b['homo']['mae']:.4f} L {b['lumo']['mae']:.4f} G {b['gap']['mae']:.4f}",
            flush=True,
        )

    result = {
        "n": int(len(df)),
        "n_scaffolds": int(n_scaffolds),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "rank": args.rank,
        "alpha": args.alpha,
        "trainable_params": trainable,
        "best_val_mae": float(best_val),
        "train_time_s": float(time.time() - t0),
        "metrics": blocks,
    }

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = MODELS_DIR / ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "rank": args.rank,
            "alpha": args.alpha,
            "lora_dropout": args.lora_dropout,
            "trainable_params": trainable,
            "best_val_mae": best_val,
        },
        ckpt_path,
    )

    pred_df = df.iloc[test].reset_index(drop=True).copy()
    for i, target in enumerate(TARGET_COLS):
        pred_df[f"gw_pred_lora_{target}"] = pred_lora[:, i]
        pred_df[f"gw_pred_const_{target}"] = const[:, i]
    pred_path = Path(args.predictions)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved checkpoint: {ckpt_path}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)
    print(f"Saved predictions: {pred_path}", flush=True)


if __name__ == "__main__":
    main()
