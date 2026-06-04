"""
Phase 4 Step 4: GNN (AttentiveFP) for HOMO/LUMO/gap prediction.
Converts SMILES to molecular graphs, trains on GPU with early stopping.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import (
    RESULTS_DIR,
    MODELS_DIR,
    RAW_DIR,
    TARGET_COLS,
    create_split_indices,
    ensure_dirs,
    regression_metrics,
    save_json,
)

OUT_DIR = RESULTS_DIR / "phase4"
RAW_CSV = RAW_DIR / "phase3_chonsfcl_mw200_500_30k.csv"
SEED = 42

HIDDEN = 128
NUM_LAYERS = 3
NUM_TIMESTEPS = 2
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 200
PATIENCE = 25


def smiles_to_graph_data(smiles_list, targets):
    """Convert SMILES to PyG Data objects."""
    from torch_geometric.utils import from_smiles

    data_list = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        try:
            data = from_smiles(smi)
            if data.x is None or data.x.size(0) == 0:
                continue
            data.y = torch.tensor(targets[i], dtype=torch.float32).unsqueeze(0)
            data_list.append(data)
            valid_idx.append(i)
        except Exception:
            continue

    print(f"  Converted {len(data_list)}/{len(smiles_list)} molecules to graphs")
    return data_list, valid_idx


def build_model(in_channels, edge_dim):
    from torch_geometric.nn.models import AttentiveFP

    model = AttentiveFP(
        in_channels=in_channels,
        hidden_channels=HIDDEN,
        out_channels=3,
        edge_dim=edge_dim,
        num_layers=NUM_LAYERS,
        num_timesteps=NUM_TIMESTEPS,
        dropout=0.2,
    )
    return model


def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(batch.x.float(), batch.edge_index,
                       batch.edge_attr.float(), batch.batch)
            loss = F.l1_loss(out, batch.y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.amp.autocast("cuda"):
            out = model(batch.x.float(), batch.edge_index,
                       batch.edge_attr.float(), batch.batch)
        preds.append(out.cpu().numpy())
        trues.append(batch.y.cpu().numpy())
    return np.vstack(preds), np.vstack(trues)


def main():
    from torch_geometric.loader import DataLoader

    ensure_dirs(OUT_DIR, MODELS_DIR)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase 4 Step 4: GNN AttentiveFP ===")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load raw data
    df = pd.read_csv(RAW_CSV)
    print(f"  Raw data: {len(df)} rows")

    # Canonicalize SMILES
    from rdkit import Chem
    def canon(s):
        try:
            m = Chem.MolFromSmiles(s)
            return Chem.MolToSmiles(m) if m else None
        except Exception:
            return None

    df["canonical_smiles"] = df["smiles"].apply(canon)
    df = df.dropna(subset=["canonical_smiles"])
    df = df[df["gap"] > 0].reset_index(drop=True)

    smiles_list = df["canonical_smiles"].tolist()
    targets = df[TARGET_COLS].values.astype(np.float32)

    # Convert to graphs
    print("\n  Converting SMILES to graphs...")
    t0 = time.time()
    data_list, valid_idx = smiles_to_graph_data(smiles_list, targets)
    print(f"  Conversion time: {time.time()-t0:.1f}s")

    if len(data_list) < 100:
        print("  ERROR: Too few valid graphs. Aborting.")
        return

    # Split
    train_idx, valid_idx_split, test_idx = create_split_indices(
        len(data_list), random_state=SEED)

    train_data = [data_list[i] for i in train_idx]
    valid_data = [data_list[i] for i in valid_idx_split]
    test_data = [data_list[i] for i in test_idx]

    # Compute target stats from training set for standardization
    train_y = np.stack([d.y.numpy() for d in train_data])
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    for d in train_data + valid_data + test_data:
        d.y = (d.y - torch.tensor(y_mean)) / torch.tensor(y_std)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Build model
    sample = data_list[0]
    in_channels = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1) if sample.edge_attr is not None else 0
    print(f"\n  Node features: {in_channels}, Edge features: {edge_dim}")
    print(f"  Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    model = build_model(in_channels, edge_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    print(f"\n  Training {EPOCHS} epochs (patience={PATIENCE})...\n")
    best_val_mae = float("inf")
    best_epoch = 0
    log_rows = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        val_pred, val_true = evaluate(model, valid_loader, device)

        # Inverse standardize for MAE
        val_pred_real = val_pred * y_std + y_mean
        val_true_real = val_true * y_std + y_mean
        val_mae = float(np.mean(np.abs(val_pred_real - val_true_real)))

        scheduler.step(val_mae)
        elapsed = time.time() - t0

        log_rows.append({"epoch": epoch, "train_loss": train_loss,
                         "val_mae": val_mae, "lr": optimizer.param_groups[0]["lr"],
                         "time_s": elapsed})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), MODELS_DIR / "gnn_attentivefp.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | val_MAE={val_mae:.4f} | "
                  f"best={best_val_mae:.4f}@{best_epoch} | lr={optimizer.param_groups[0]['lr']:.1e} | {elapsed:.1f}s")

        if epoch - best_epoch >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # Save training log
    pd.DataFrame(log_rows).to_csv(OUT_DIR / "gnn_training_log.csv", index=False)

    # Load best model and evaluate on test
    model.load_state_dict(torch.load(MODELS_DIR / "gnn_attentivefp.pt", weights_only=True))
    test_pred, test_true = evaluate(model, test_loader, device)
    test_pred_real = test_pred * y_std + y_mean
    test_true_real = test_true * y_std + y_mean

    m = regression_metrics(test_true_real, test_pred_real)

    print(f"\n{'='*50}")
    print(f"  GNN AttentiveFP Test Results")
    print(f"{'='*50}")
    for t in TARGET_COLS:
        print(f"  {t:5s}: MAE={m[t]['mae']:.4f} R2={m[t]['r2']:.4f}")
    print(f"  avg  : MAE={m['average']['mae']:.4f} R2={m['average']['r2']:.4f}")

    save_json({
        "model": "AttentiveFP",
        "hidden": HIDDEN,
        "num_layers": NUM_LAYERS,
        "num_timesteps": NUM_TIMESTEPS,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "epochs_trained": len(log_rows),
        "metrics": m,
    }, OUT_DIR / "gnn_metrics.json")

    print(f"\n  Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
