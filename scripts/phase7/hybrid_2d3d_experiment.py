"""
Phase 7: Hybrid 2D+3D experiment.

Train SchNet + RDKit 2D descriptors on the Phase 6 44.8k dataset.
Compare with Phase 6 baseline (3D-only, avg MAE=0.162, R²=0.882).

Usage:
  .venv\Scripts\python.exe scripts/phase7/hybrid_2d3d_experiment.py
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from molgap.constants import (
    RESULTS_DIR, MODELS_DIR, TARGET_COLS, DATA_PHASE3, DATA_PHASE6_LARGE, GRAPHS_PHASE6,
)
from molgap.utils import (
    create_split_indices, regression_metrics, save_json, ensure_dirs,
    calc_rdkit_descriptors, canonicalize_smiles,
)
from molgap.schnet import SchNetWrapper

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

GRAPH_PATH = GRAPHS_PHASE6
OUT_DIR = RESULTS_DIR / "phase7" / "hybrid_2d3d"

PARAMS = {
    "hidden_channels": 192, "num_filters": 256, "num_interactions": 6,
    "num_gaussians": 100, "cutoff": 5.0, "dropout": 0.1,
    "lr": 1.56e-3, "weight_decay": 2.06e-5,
    "batch_size": 256, "scheduler": "plateau",
}

EPOCHS = 300
PATIENCE = 40


def compute_2d_descriptors(csv_paths):
    """Load CSVs, compute RDKit 2D descriptors for all molecules."""
    from rdkit import Chem

    dfs = []
    for p in csv_paths:
        if p.exists():
            dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    for col in ["homo", "lumo", "gap", "mw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["homo", "lumo", "gap", "smiles"])
    df = df[df["gap"] > 0]
    df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"])
    df = df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)

    print(f"Computing 2D descriptors for {len(df)} molecules...", flush=True)
    desc_rows = []
    for i, smi in enumerate(df["canonical_smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            desc_rows.append(calc_rdkit_descriptors(mol))
        else:
            desc_rows.append({})
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(df)}", flush=True)

    desc_df = pd.DataFrame(desc_rows)
    desc_df = desc_df.apply(pd.to_numeric, errors="coerce")
    desc_df = desc_df.fillna(0.0)

    std = desc_df.std()
    keep = std[std > 1e-8].index.tolist()
    desc_df = desc_df[keep]
    print(f"  {len(keep)} descriptors after dropping constants", flush=True)

    return df, desc_df


def main():
    ensure_dirs(OUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load graphs
    print("Loading 3D graphs...", flush=True)
    data_list = torch.load(GRAPH_PATH, weights_only=False)
    print(f"Loaded {len(data_list)} graphs", flush=True)

    # Compute 2D descriptors
    csv_paths = [DATA_PHASE3, DATA_PHASE6_LARGE]
    df, desc_df = compute_2d_descriptors(csv_paths)

    n_desc = desc_df.shape[1]
    print(f"2D descriptors: {n_desc} features", flush=True)

    if len(desc_df) > len(data_list):
        desc_df = desc_df.iloc[:len(data_list)]
    elif len(desc_df) < len(data_list):
        data_list = data_list[:len(desc_df)]

    # Normalize descriptors
    desc_arr = desc_df.values.astype(np.float32)
    desc_mean = desc_arr.mean(axis=0)
    desc_std = desc_arr.std(axis=0)
    desc_std[desc_std < 1e-8] = 1.0
    desc_arr = (desc_arr - desc_mean) / desc_std

    # Inject desc into PyG data
    for i, d in enumerate(data_list):
        d.desc = torch.tensor(desc_arr[i], dtype=torch.float32)

    # Split
    train_idx, valid_idx, test_idx = create_split_indices(len(data_list), random_state=SEED)
    train_y = np.stack([data_list[i].y.squeeze(0).numpy() for i in train_idx])
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    for d in data_list:
        d.y = (d.y - torch.tensor(y_mean)) / torch.tensor(y_std)

    train_data = [data_list[i] for i in train_idx]
    valid_data = [data_list[i] for i in valid_idx]
    test_data  = [data_list[i] for i in test_idx]
    del data_list

    print(f"Split: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")
    print(f"y_mean={y_mean}, y_std={y_std}")

    # ── Train 3D+2D hybrid ──
    from torch_geometric.loader import DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    has_charges = hasattr(train_data[0], 'charges')
    bs = PARAMS["batch_size"]
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=bs)
    test_loader = DataLoader(test_data, batch_size=bs)

    model = SchNetWrapper(
        hidden_channels=PARAMS["hidden_channels"],
        num_filters=PARAMS["num_filters"],
        num_interactions=PARAMS["num_interactions"],
        num_gaussians=PARAMS["num_gaussians"],
        cutoff=PARAMS["cutoff"],
        dropout=PARAMS["dropout"],
        use_charges=has_charges,
        n_desc=n_desc,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  SchNet 3D + {n_desc} RDKit 2D descriptors")
    print(f"  params={n_params:,}, cutoff={PARAMS['cutoff']}, bs={bs}")
    print(f"{'='*60}\n", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=max(5, PATIENCE // 3), min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    log_rows = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss, n = 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                charges = getattr(batch, 'charges', None)
                desc = getattr(batch, 'desc', None)
                out = model(batch.z, batch.pos, batch.batch,
                            charges=charges, desc=desc)
                loss = F.l1_loss(out, batch.y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs
        train_loss = total_loss / n

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                with torch.amp.autocast("cuda"):
                    charges = getattr(batch, 'charges', None)
                    desc = getattr(batch, 'desc', None)
                    out = model(batch.z, batch.pos, batch.batch,
                                charges=charges, desc=desc)
                preds.append(out.cpu().numpy())
                trues.append(batch.y.cpu().numpy())
        val_pred = np.vstack(preds) * y_std + y_mean
        val_true = np.vstack(trues) * y_std + y_mean
        val_mae = float(np.mean(np.abs(val_pred - val_true)))

        scheduler.step(val_mae)
        elapsed = time.time() - t0

        log_rows.append({"epoch": epoch, "train_loss": train_loss,
                         "val_mae": val_mae, "lr": optimizer.param_groups[0]["lr"],
                         "time_s": elapsed})

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or is_best:
            marker = " *" if is_best else ""
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | val_MAE={val_mae:.4f} | "
                  f"best={best_val_mae:.4f}@{best_epoch} | {elapsed:.1f}s{marker}", flush=True)

        if epoch - best_epoch >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (best={best_epoch})", flush=True)
            break

    # ── Test ──
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            with torch.amp.autocast("cuda"):
                charges = getattr(batch, 'charges', None)
                desc = getattr(batch, 'desc', None)
                out = model(batch.z, batch.pos, batch.batch,
                            charges=charges, desc=desc)
            preds.append(out.cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    test_pred = np.vstack(preds) * y_std + y_mean
    test_true = np.vstack(trues) * y_std + y_mean
    m = regression_metrics(test_true, test_pred)

    # Phase 6 baseline
    p6_mae = 0.1620
    p6_r2 = 0.8823

    print(f"\n{'='*60}")
    print(f"  RESULTS: 3D+2D Hybrid vs Phase 6 (3D-only)")
    print(f"{'='*60}")
    for t in TARGET_COLS:
        print(f"  {t:5s}: MAE={m[t]['mae']:.4f}  R2={m[t]['r2']:.4f}")
    print(f"  avg  : MAE={m['average']['mae']:.4f}  R2={m['average']['r2']:.4f}")
    print(f"\n  Phase 6 (3D-only): avg MAE={p6_mae:.4f}  R2={p6_r2:.4f}")
    print(f"  Hybrid (3D+2D):   avg MAE={m['average']['mae']:.4f}  R2={m['average']['r2']:.4f}")
    print(f"  Delta: R2 {m['average']['r2'] - p6_r2:+.4f}, MAE {p6_mae - m['average']['mae']:+.4f}")

    pd.DataFrame(log_rows).to_csv(OUT_DIR / "hybrid_train_log.csv", index=False)

    save_json({
        "experiment": "hybrid_2d3d",
        "n_data": len(train_data) + len(valid_data) + len(test_data),
        "n_desc": n_desc,
        "params": PARAMS,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "metrics": m,
        "phase6_baseline": {"avg_mae": p6_mae, "avg_r2": p6_r2},
        "delta_avg_r2": m['average']['r2'] - p6_r2,
        "delta_avg_mae": p6_mae - m['average']['mae'],
        "desc_mean": desc_mean.tolist(),
        "desc_std": desc_std.tolist(),
        "desc_names": desc_df.columns.tolist(),
    }, OUT_DIR / "hybrid_comparison.json")

    if m['average']['r2'] > p6_r2:
        torch.save(best_state, MODELS_DIR / "gnn_schnet_3d2d_hybrid.pt")
        print(f"\n  Hybrid model saved to models/gnn_schnet_3d2d_hybrid.pt")

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
