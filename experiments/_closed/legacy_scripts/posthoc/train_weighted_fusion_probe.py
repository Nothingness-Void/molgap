"""Probe tail-aware B3LYP FusionHead fine-tuning on v3 embeddings.

The v3 encoders stay frozen. Starting from the selected expansion500k FusionHead,
this script fine-tunes only the FusionHead with simple low-gap / high-MW sample
weights, then checks whether the new heads improve the external Phase 8 common
eval. This is a cheap B3LYP-level alternative to a full encoder retrain.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from eval_full_replacement_common import _build_graphs, _load_trio, _metric_blocks, _predict
from molgap.constants import MODELS_DIR, RAW_DIR, RESULTS_DIR, SEED
from molgap.fusion import FusionHead

PHASE8 = RESULTS_DIR / "phase8"
TARGETS = ("homo", "lumo", "gap")
DISPLAY_TARGETS = ("HOMO", "LUMO", "Gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=RAW_DIR / "phase8_expansion_500k.csv")
    archive = PHASE8 / "archive" / "legacy" / "head_posthoc"
    parser.add_argument(
        "--common-csv",
        type=Path,
        default=PHASE8 / "archive" / "legacy" / "pilots_30k" / "common_eval_30k_predictions.csv",
    )
    parser.add_argument("--emb-2d", type=Path, default=PHASE8 / "gps_expansion_500k_embeddings.pt")
    parser.add_argument("--emb-3d", type=Path, default=PHASE8 / "schnet_expansion_500k_embeddings.pt")
    parser.add_argument("--base-fusion", type=Path, default=MODELS_DIR / "phase8_hybrid_fusion_expansion_500k.pt")
    parser.add_argument("--out-json", type=Path, default=archive / "weighted_fusion_probe_metrics.json")
    parser.add_argument("--out-md", type=Path, default=archive / "weighted_fusion_probe_decision.md")
    parser.add_argument("--predictions", type=Path, default=archive / "weighted_fusion_probe_common_predictions.csv")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--bs-2d", type=int, default=256)
    parser.add_argument("--bs-3d", type=int, default=128)
    parser.add_argument("--bs-fusion", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    return parser.parse_args()


def load_embedding_payload(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    return payload["embeddings"].float(), payload["source_idx"].long()


def load_aligned(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    h2, idx2 = load_embedding_payload(args.emb_2d)
    h3, idx3 = load_embedding_payload(args.emb_3d)
    pos2 = {int(v): i for i, v in enumerate(idx2.tolist())}
    pos3 = {int(v): i for i, v in enumerate(idx3.tolist())}
    common = np.array(sorted(set(pos2).intersection(pos3)), dtype=np.int64)
    ii2 = torch.tensor([pos2[int(i)] for i in common], dtype=torch.long)
    ii3 = torch.tensor([pos3[int(i)] for i in common], dtype=torch.long)
    df = pd.read_csv(args.train_csv).iloc[common].reset_index(drop=True)
    y = torch.tensor(df[list(TARGETS)].to_numpy(dtype=np.float32))
    return h2[ii2], h3[ii3], y, df


def make_split(n: int) -> dict[str, np.ndarray]:
    idx = np.random.RandomState(SEED).permutation(n)
    n_train, n_val = int(0.8 * len(idx)), int(0.1 * len(idx))
    return {"train": idx[:n_train], "val": idx[n_train:n_train + n_val], "test": idx[n_train + n_val:]}


def sample_weights(df: pd.DataFrame, mode: str) -> np.ndarray:
    gap = df["gap"].to_numpy(dtype=np.float32)
    mw = df["mw"].to_numpy(dtype=np.float32)
    weights = np.ones(len(df), dtype=np.float32)
    if mode == "lowgap":
        weights += 1.0 * (gap < 4.0) + 2.0 * (gap < 3.0)
    elif mode == "lowgap_mw":
        weights += 1.0 * (gap < 4.0) + 2.0 * (gap < 3.0) + 1.5 * (mw > 800.0)
    else:
        raise ValueError(f"unknown weighting mode: {mode}")
    return weights / weights.mean()


def make_loader(h2, h3, y, w, idx, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(h2[idx], h3[idx], y[idx], torch.tensor(w[idx], dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0,
    )


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for i, name in enumerate(DISPLAY_TARGETS):
        out[name] = {
            "mae": float(mean_absolute_error(y_true[:, i], pred[:, i])),
            "r2": float(r2_score(y_true[:, i], pred[:, i])),
        }
    out["average"] = {
        "mae": float(np.mean([out[t]["mae"] for t in DISPLAY_TARGETS])),
        "r2": float(np.mean([out[t]["r2"] for t in DISPLAY_TARGETS])),
    }
    return out


@torch.no_grad()
def evaluate(model, h2, h3, y, idx, batch_size: int, device: torch.device) -> dict:
    pred, true = [], []
    model.eval()
    dummy_w = np.ones(len(y), dtype=np.float32)
    for b2, b3, by, _ in make_loader(h2, h3, y, dummy_w, idx, batch_size, False):
        pred.append(model(b2.to(device), b3.to(device)).float().cpu().numpy())
        true.append(by.numpy())
    return metrics(np.concatenate(true), np.concatenate(pred))


def train_one(args, h2, h3, y, df, split, mode: str, device: torch.device) -> tuple[Path, dict]:
    weights = sample_weights(df, mode)
    model = FusionHead("gate", 192, 0.0).to(device)
    model.load_state_dict(torch.load(args.base_fusion, weights_only=True, map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5, min_lr=1e-6)
    train_loader = make_loader(h2, h3, y, weights, split["train"], args.batch_size, True)
    val_loader = make_loader(h2, h3, y, weights, split["val"], args.batch_size, False)
    best_val, best_state, best_epoch, wait = float("inf"), None, -1, 0
    log = []
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        for b2, b3, by, bw in train_loader:
            opt.zero_grad()
            pred = model(b2.to(device), b3.to(device))
            per_row = torch.abs(pred - by.to(device)).mean(dim=1)
            loss = (per_row * bw.to(device)).mean()
            loss.backward()
            opt.step()
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for b2, b3, by, bw in val_loader:
                pred = model(b2.to(device), b3.to(device))
                per_row = torch.abs(pred - by.to(device)).mean(dim=1)
                loss = (per_row * bw.to(device)).sum()
                total += float(loss.item())
                n += by.size(0)
        val = total / max(n, 1)
        sched.step(val)
        improved = val < best_val
        if improved:
            best_val = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        log.append({"epoch": epoch, "weighted_val_mae": float(val), "time_s": time.time() - t0})
        print(f"{mode} ep{epoch:03d} weighted_val={val:.5f} best={best_val:.5f}@{best_epoch}", flush=True)
        if wait >= args.patience:
            break
    model.load_state_dict(best_state)
    out = MODELS_DIR / f"phase8_hybrid_fusion_expansion_500k_weighted_{mode}.pt"
    torch.save(best_state, out)
    result = {
        "mode": mode,
        "model": str(out),
        "best_weighted_val_mae": float(best_val),
        "best_epoch": int(best_epoch),
        "internal_test": evaluate(model, h2, h3, y, split["test"], args.batch_size, device),
        "weight_summary": {
            "min": float(weights.min()),
            "mean": float(weights.mean()),
            "max": float(weights.max()),
        },
        "log": log,
    }
    return out, result


def write_decision(path: Path, result: dict) -> None:
    common = result["common_eval"]
    baseline = common["baseline"]["all"]
    best = min((name for name in common if name != "baseline"), key=lambda n: common[n]["all"]["average"]["mae"])
    best_row = common[best]["all"]
    delta_avg = best_row["average"]["mae"] - baseline["average"]["mae"]
    delta_gap = best_row["Gap"]["mae"] - baseline["Gap"]["mae"]
    verdict = "positive" if delta_avg < -0.001 and delta_gap <= 0 else "negative"
    lines = [
        "# Phase 8 Weighted FusionHead Probe",
        "",
        "Date: 2026-07-06",
        "",
        "## Setup",
        "",
        "- Base: v3 expansion500k GPS/SchNet encoders frozen.",
        "- Starting point: selected v3 FusionHead checkpoint.",
        "- Target: B3LYP HOMO/LUMO/Gap labels only.",
        "- Probe: low-gap and low-gap+high-MW weighted L1 fine-tuning of FusionHead.",
        "",
        "## Common Eval MAE",
        "",
        "| model | HOMO | LUMO | Gap | avg |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, blocks in common.items():
        row = blocks["all"]
        lines.append(
            f"| {name} | {row['HOMO']['mae']:.4f} | {row['LUMO']['mae']:.4f} | "
            f"{row['Gap']['mae']:.4f} | {row['average']['mae']:.4f} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"Probe verdict: **{verdict}**. Best model `{best}` changes common-eval avg/GAP MAE by `{delta_avg:+.5f}/{delta_gap:+.5f}` eV versus v3.",
    ])
    if verdict == "negative":
        lines.append("Do not promote weighted FusionHead fine-tuning; keep the selected v3 FusionHead.")
    else:
        lines.append("Keep this as a B3LYP-level candidate and run a second validation slice before changing defaults.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    h2, h3, y, train_df = load_aligned(args)
    split = make_split(len(train_df))
    print(f"Aligned split {len(split['train'])}/{len(split['val'])}/{len(split['test'])}", flush=True)

    trained = {}
    model_paths = {}
    for mode in ("lowgap", "lowgap_mw"):
        path, block = train_one(args, h2, h3, y, train_df, split, mode, device)
        trained[mode] = block
        model_paths[mode] = path

    print("Building common eval graphs", flush=True)
    eval_df = pd.read_csv(args.common_csv)
    eval_df, graphs_2d, graphs_3d = _build_graphs(eval_df)
    common = {}
    pred_df = eval_df.copy()
    specs = {"baseline": args.base_fusion, **model_paths}
    for name, fusion_path in specs.items():
        print(f"Common eval {name}", flush=True)
        trio = _load_trio(
            MODELS_DIR / "phase8_gps_expansion_500k.pt",
            MODELS_DIR / "phase8_schnet_expansion_500k.pt",
            fusion_path,
            device,
        )
        preds = _predict(*trio, graphs_2d, graphs_3d, args, device)["hybrid"]
        common[name] = _metric_blocks(eval_df, preds)
        for i, target in enumerate(TARGETS):
            pred_df[f"{name}_{target}"] = preds[:, i]
    pred_df.to_csv(args.predictions, index=False, encoding="utf-8")

    result = {
        "kind": "weighted_fusion_head_probe",
        "base": "phase8_expansion_hybrid",
        "split": {k: int(len(v)) for k, v in split.items()},
        "trained": trained,
        "common_eval": common,
    }
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_decision(args.out_md, result)
    print(f"Metrics -> {args.out_json}", flush=True)
    print(f"Decision -> {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
