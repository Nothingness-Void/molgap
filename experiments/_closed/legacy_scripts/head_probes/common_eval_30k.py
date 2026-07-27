"""
Common-evaluate Phase 8 old30k vs replacement30k models.

This isolates the data-coverage question: both 30k model families are evaluated
on the same external OOD/hard molecules, using the standard single FusionHead.

Usage:
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/head_probes/common_eval_30k.py
  .venv\\Scripts\\python.exe scripts/phase8/archive/legacy/head_probes/common_eval_30k.py --max-hard 100 --fusion-epochs 3
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm

from molgap.constants import MODELS_DIR, RAW_DIR, RESULTS_DIR, SEED
from molgap.fusion import FusionHead
from molgap.gps import GPSWrapper
from molgap.graphs import smiles_to_2d_pyg, smiles_to_pyg
from molgap.schnet import SchNetWrapper
from molgap.constants import PARAMS_GPS_2D, PARAMS_SCHNET_300K
from molgap.utils import ensure_dirs, require_rdkit

PHASE8_DIR = RESULTS_DIR / "phase8"
PHASE7_OOD = RESULTS_DIR / "phase7" / "ood_1000" / "ood_molecules_1000.csv"
OLD_CSV = RAW_DIR / "phase7_chonsfcl_mw200_1000_300k.csv"
REPLACEMENT_CSV = RAW_DIR / "phase8_replacement_300k.csv"
HARD_CSV = RAW_DIR / "phase8_targeted_replacement_balanced_50k.csv"
TARGETS = ["homo", "lumo", "gap"]
DISPLAY_TARGETS = ["HOMO", "LUMO", "Gap"]
TAGS = ["old30k", "replacement30k"]


def _canonicalize(smiles: str) -> str | None:
    require_rdkit()
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _load_training_canonicals() -> set[str]:
    old = pd.read_csv(OLD_CSV, nrows=30000, usecols=["canonical_smiles"])
    new = pd.read_csv(REPLACEMENT_CSV, nrows=30000, usecols=["canonical_smiles"])
    return set(old["canonical_smiles"].dropna()) | set(new["canonical_smiles"].dropna())


def _build_eval_frame(max_ood: int | None, max_hard: int, hard_seed: int) -> pd.DataFrame:
    ood = pd.read_csv(PHASE7_OOD)
    if max_ood is not None:
        ood = ood.head(max_ood).copy()
    ood = ood[["cid", "mw", "formula", "smiles", *TARGETS]].copy()
    ood["eval_set"] = "ood1000"
    ood["bucket"] = "phase7_ood"
    ood["canonical_smiles"] = [
        _canonicalize(smi) for smi in tqdm(ood["smiles"], desc="canonical OOD")
    ]

    train_can = _load_training_canonicals()
    used_can = set(ood["canonical_smiles"].dropna())

    hard = pd.read_csv(HARD_CSV)
    hard = hard[~hard["canonical_smiles"].isin(train_can | used_can)].copy()
    if max_hard > 0 and len(hard) > max_hard:
        per_bucket = max(1, math.ceil(max_hard / hard["bucket"].nunique()))
        hard = (
            hard.groupby("bucket", group_keys=False)
            .sample(n=per_bucket, replace=False, random_state=hard_seed)
            .head(max_hard)
            .copy()
        )
    hard = hard[["bucket", "cid", "mw", "formula", "smiles", "canonical_smiles", *TARGETS]].copy()
    hard["eval_set"] = "p8_targeted_hard"

    cols = ["eval_set", "bucket", "cid", "mw", "formula", "smiles", "canonical_smiles", *TARGETS]
    out = pd.concat([ood[cols], hard[cols]], ignore_index=True)
    out = out.drop_duplicates("canonical_smiles", keep="first").reset_index(drop=True)
    return out


def _build_one_eval(row: tuple[int, str]):
    i, smi = row
    try:
        g2d = smiles_to_2d_pyg(smi)
        g3d = smiles_to_pyg(smi)
        if g2d is None or g3d is None:
            return None
        return i, g2d, g3d
    except Exception:
        return None


def _build_eval_graphs(eval_df: pd.DataFrame, n_jobs: int):
    rows = list(enumerate(eval_df["smiles"].tolist()))
    if n_jobs <= 1:
        built = [_build_one_eval(row) for row in tqdm(rows, desc="build eval graphs")]
    else:
        with mp.Pool(processes=n_jobs) as pool:
            built = list(tqdm(
                pool.imap_unordered(_build_one_eval, rows, chunksize=8),
                total=len(rows),
                desc="build eval graphs",
            ))
    built = [x for x in built if x is not None]
    built.sort(key=lambda x: x[0])
    valid_idx = [x[0] for x in built]
    g2d = [x[1] for x in built]
    g3d = [x[2] for x in built]
    return eval_df.iloc[valid_idx].reset_index(drop=True), g2d, g3d


def _load_embedding_payload(path: Path):
    payload = torch.load(path, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain embeddings + source_idx")
    return payload["embeddings"].float(), payload["source_idx"].long()


def _load_aligned_for_fusion(tag: str):
    h2, idx2 = _load_embedding_payload(PHASE8_DIR / f"gps_{tag}_embeddings.pt")
    h3, idx3 = _load_embedding_payload(PHASE8_DIR / f"schnet_{tag}_embeddings.pt")
    graphs = torch.load(PHASE8_DIR / f"pyg_3d_graphs_etkdg_{tag}.pt", weights_only=False)
    labels_by_idx = {
        int(g.source_idx.view(-1)[0].item()): g.y.squeeze(0).float()
        for g in graphs
    }
    pos2 = {int(v): i for i, v in enumerate(idx2.tolist())}
    pos3 = {int(v): i for i, v in enumerate(idx3.tolist())}
    common = sorted(set(pos2).intersection(pos3).intersection(labels_by_idx))
    ii2 = torch.tensor([pos2[i] for i in common], dtype=torch.long)
    ii3 = torch.tensor([pos3[i] for i in common], dtype=torch.long)
    y = torch.stack([labels_by_idx[i] for i in common])
    return h2[ii2], h3[ii3], y, torch.tensor(common, dtype=torch.long)


def _make_split(n: int):
    idx = np.random.RandomState(SEED).permutation(n)
    n_train, n_val = int(0.8 * len(idx)), int(0.1 * len(idx))
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


def _make_fusion_loader(h2, h3, y, idx, batch_size, shuffle):
    ds = TensorDataset(h2[idx], h3[idx], y[idx])
    return TorchDataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)


def _eval_fusion_loss(model, h2, h3, y, idx, device):
    loader = _make_fusion_loader(h2, h3, y, idx, 2048, False)
    crit = nn.L1Loss()
    total, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for b2, b3, by in loader:
            loss = crit(model(b2.to(device), b3.to(device)), by.to(device))
            total += loss.item() * by.size(0)
            n += by.size(0)
    return total / max(n, 1)


def _train_fusion_head(tag: str, args, device):
    out = MODELS_DIR / f"phase8_hybrid_fusion_{tag}_common_eval.pt"
    if out.exists() and not args.retrain_fusion:
        print(f"[skip] fusion head exists: {out}", flush=True)
        return out, None

    h2, h3, y, source_idx = _load_aligned_for_fusion(tag)
    split = _make_split(h2.shape[0])
    model = FusionHead("gate", 192, 0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-6)
    crit = nn.L1Loss()
    train_loader = _make_fusion_loader(h2, h3, y, split["train"], args.batch_size, True)

    best_val, best_state, best_epoch, wait = float("inf"), None, -1, 0
    log_rows = []
    for epoch in range(args.fusion_epochs):
        t0 = time.time()
        model.train()
        for b2, b3, by in train_loader:
            opt.zero_grad()
            loss = crit(model(b2.to(device), b3.to(device)), by.to(device))
            loss.backward()
            opt.step()
        val = _eval_fusion_loss(model, h2, h3, y, split["val"], device)
        sched.step(val)
        if val < best_val:
            best_val = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        log_rows.append({"epoch": epoch, "val_mae": float(val), "time_s": time.time() - t0})
        print(f"{tag} fusion ep{epoch:03d} val={val:.4f} best={best_val:.4f}@{best_epoch}", flush=True)
        if wait >= args.fusion_patience:
            break

    if best_state is None:
        raise RuntimeError(f"No fusion checkpoint produced for {tag}")
    torch.save(best_state, out)
    return out, {
        "n_aligned": int(h2.shape[0]),
        "source_idx_min": int(source_idx.min().item()),
        "source_idx_max": int(source_idx.max().item()),
        "best_val_mae": float(best_val),
        "best_epoch": int(best_epoch),
        "log": log_rows,
    }


def _load_tag_models(tag: str, fusion_path: Path, device):
    gps = GPSWrapper(**PARAMS_GPS_2D).to(device)
    gps.load_state_dict(torch.load(PHASE8_DIR / f"gps_{tag}.pt", weights_only=True, map_location=device))
    gps.eval()

    schnet = SchNetWrapper(**PARAMS_SCHNET_300K, use_charges=True).to(device)
    schnet.load_state_dict(torch.load(PHASE8_DIR / f"schnet_{tag}.pt", weights_only=True, map_location=device))
    schnet.eval()

    fusion = FusionHead("gate", 192, 0.0).to(device)
    fusion.load_state_dict(torch.load(fusion_path, weights_only=True, map_location=device))
    fusion.eval()
    return gps, schnet, fusion


@torch.no_grad()
def _predict_tag(tag: str, fusion_path: Path, g2d, g3d, device, args):
    gps, schnet, fusion = _load_tag_models(tag, fusion_path, device)

    emb2, pred2 = [], []
    for batch in GeometricDataLoader(g2d, batch_size=args.bs_2d, shuffle=False):
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            emb = gps.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = gps(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        emb2.append(emb.float().cpu())
        pred2.append(pred.float().cpu())
    emb2 = torch.cat(emb2)
    pred2 = torch.cat(pred2).numpy()

    emb3, pred3 = [], []
    for batch in GeometricDataLoader(g3d, batch_size=args.bs_3d, shuffle=False):
        batch = batch.to(device)
        charges = batch.charges if hasattr(batch, "charges") else None
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            emb = schnet.encode(batch.z, batch.pos, batch.batch, charges=charges)
            pred = schnet(batch.z, batch.pos, batch.batch, charges=charges)
        emb3.append(emb.float().cpu())
        pred3.append(pred.float().cpu())
    emb3 = torch.cat(emb3)
    pred3 = torch.cat(pred3).numpy()

    hybrid = []
    ds = TensorDataset(emb2, emb3)
    for b2, b3 in TorchDataLoader(ds, batch_size=args.bs_fusion, shuffle=False):
        pred = fusion(b2.to(device), b3.to(device))
        hybrid.append(pred.float().cpu())
    pred_h = torch.cat(hybrid).numpy()
    return {"gps_2d": pred2, "schnet_3d": pred3, "hybrid": pred_h}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
    out = {}
    for i, name in enumerate(DISPLAY_TARGETS):
        out[name] = {
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
        }
    out["average"] = {
        "mae": float(np.mean([out[k]["mae"] for k in DISPLAY_TARGETS])),
        "r2": float(np.mean([out[k]["r2"] for k in DISPLAY_TARGETS])),
    }
    return out


def _metric_blocks(eval_df: pd.DataFrame, pred: np.ndarray):
    y = eval_df[TARGETS].to_numpy(dtype=np.float32)
    blocks = {"all": _metrics(y, pred)}
    for eval_set in sorted(eval_df["eval_set"].unique()):
        mask = eval_df["eval_set"].to_numpy() == eval_set
        blocks[eval_set] = _metrics(y[mask], pred[mask])
    return blocks


def main():
    parser = argparse.ArgumentParser(description="Common-evaluate old30k vs replacement30k")
    parser.add_argument("--max-ood", type=int, default=None)
    parser.add_argument("--max-hard", type=int, default=1000)
    parser.add_argument("--hard-seed", type=int, default=SEED)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--fusion-epochs", type=int, default=200)
    parser.add_argument("--fusion-patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5.4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--retrain-fusion", action="store_true")
    parser.add_argument("--bs-2d", type=int, default=256)
    parser.add_argument("--bs-3d", type=int, default=128)
    parser.add_argument("--bs-fusion", type=int, default=2048)
    archive = PHASE8_DIR / "archive" / "legacy" / "pilots_30k"
    parser.add_argument("--out", type=Path, default=archive / "common_eval_30k_metrics.json")
    parser.add_argument("--predictions", type=Path, default=archive / "common_eval_30k_predictions.csv")
    args = parser.parse_args()

    ensure_dirs(PHASE8_DIR, MODELS_DIR)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    fusion_info = {}
    fusion_paths = {}
    for tag in TAGS:
        path, info = _train_fusion_head(tag, args, device)
        fusion_paths[tag] = path
        if info is not None:
            fusion_info[tag] = info

    eval_df = _build_eval_frame(args.max_ood, args.max_hard, args.hard_seed)
    eval_df, g2d, g3d = _build_eval_graphs(eval_df, args.n_jobs)
    print(f"Common eval valid N={len(eval_df)} sets={eval_df['eval_set'].value_counts().to_dict()}", flush=True)

    all_metrics = {
        "n_eval": int(len(eval_df)),
        "eval_set_counts": {k: int(v) for k, v in eval_df["eval_set"].value_counts().items()},
        "fusion_training": fusion_info,
        "models": {},
    }
    pred_df = eval_df.copy()
    for tag in TAGS:
        preds = _predict_tag(tag, fusion_paths[tag], g2d, g3d, device, args)
        all_metrics["models"][tag] = {}
        for name, pred in preds.items():
            all_metrics["models"][tag][name] = _metric_blocks(eval_df, pred)
            for i, target in enumerate(TARGETS):
                pred_df[f"{tag}_{name}_{target}"] = pred[:, i]

    old = all_metrics["models"]["old30k"]["hybrid"]
    new = all_metrics["models"]["replacement30k"]["hybrid"]
    deltas = {}
    for block in old:
        deltas[block] = {
            "avg_mae": float(new[block]["average"]["mae"] - old[block]["average"]["mae"]),
            "gap_mae": float(new[block]["Gap"]["mae"] - old[block]["Gap"]["mae"]),
        }
    all_metrics["replacement_minus_old_hybrid"] = deltas

    args.out.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    pred_df.to_csv(args.predictions, index=False, encoding="utf-8")
    print(f"Metrics -> {args.out}", flush=True)
    print(f"Predictions -> {args.predictions}", flush=True)
    for block, vals in deltas.items():
        print(f"{block}: replacement-old hybrid Δavg={vals['avg_mae']:+.4f} Δgap={vals['gap_mae']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
