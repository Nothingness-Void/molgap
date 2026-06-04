"""
Phase 4 Step 4b: SchNet (3D) for HOMO/LUMO/gap prediction.
Generates 3D conformers via RDKit ETKDG, builds radius graphs,
trains SchNet on GPU with Gaussian basis distance encoding.
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
PROCESSED_GRAPHS = RESULTS_DIR / "phase4" / "pyg_3d_graphs.pt"
SEED = 42

HIDDEN = 256
NUM_FILTERS = 256
NUM_INTERACTIONS = 5
NUM_GAUSSIANS = 50
CUTOFF = 10.0
BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 300
PATIENCE = 30


def generate_3d_coords(mol):
    """Generate 3D conformer for a molecule using ETKDG."""
    from rdkit.Chem import AllChem
    mol_h = AllChem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    if result != 0:
        result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    if result != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass
    return mol_h


def mol_to_pyg_data(mol_3d, targets, remove_h=True):
    """Convert RDKit mol with 3D coords to PyG Data object."""
    from torch_geometric.data import Data
    from rdkit import Chem

    if remove_h:
        mol_3d = Chem.RemoveHs(mol_3d)

    conf = mol_3d.GetConformer()
    n_atoms = mol_3d.GetNumAtoms()
    if n_atoms == 0:
        return None

    # Atom features: atomic number (used as embedding index by SchNet)
    z = torch.tensor([atom.GetAtomicNum() for atom in mol_3d.GetAtoms()],
                     dtype=torch.long)

    # 3D positions
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)

    data = Data(z=z, pos=pos)
    data.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(0)
    return data


def build_graph_dataset(smiles_list, targets_array):
    """Convert SMILES list to 3D PyG Data objects."""
    from rdkit import Chem

    data_list = []
    failed = 0

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue

        mol_3d = generate_3d_coords(mol)
        if mol_3d is None:
            failed += 1
            continue

        data = mol_to_pyg_data(mol_3d, targets_array[i])
        if data is None:
            failed += 1
            continue

        data_list.append(data)

        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(smiles_list)} done ({failed} failed)")

    print(f"  Total: {len(data_list)} graphs, {failed} failed")
    return data_list


class SchNetWrapper(torch.nn.Module):
    """SchNet with multi-target output head."""
    def __init__(self, hidden_channels, num_filters, num_interactions,
                 num_gaussians, cutoff, n_targets=3):
        super().__init__()
        from torch_geometric.nn.models import SchNet

        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels // 2, n_targets),
        )

    def forward(self, z, pos, batch):
        # SchNet returns per-atom embeddings; we need graph-level
        from torch_geometric.nn import global_mean_pool

        # Get atom-level features from SchNet's representation network
        h = self.schnet.embedding(z)
        edge_index, edge_weight = self._radius_graph(pos, batch)
        edge_attr = self.schnet.distance_expansion(edge_weight)

        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = global_mean_pool(h, batch)
        return self.head(h)

    def _radius_graph(self, pos, batch):
        from torch_geometric.nn.models.schnet import radius_graph
        edge_index = radius_graph(pos, r=self.schnet.cutoff, batch=batch,
                                  max_num_neighbors=32)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(batch.z, batch.pos, batch.batch)
            loss = F.l1_loss(out, batch.y)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
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
            out = model(batch.z, batch.pos, batch.batch)
        preds.append(out.cpu().numpy())
        trues.append(batch.y.cpu().numpy())
    return np.vstack(preds), np.vstack(trues)


def main():
    from torch_geometric.loader import DataLoader

    ensure_dirs(OUT_DIR, MODELS_DIR)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase 4 Step 4b: SchNet 3D ===")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load or build graph dataset
    if PROCESSED_GRAPHS.exists():
        print(f"\n  Loading cached graphs from {PROCESSED_GRAPHS}...")
        data_list = torch.load(PROCESSED_GRAPHS, weights_only=False)
        print(f"  Loaded {len(data_list)} graphs")
    else:
        df = pd.read_csv(RAW_CSV)
        print(f"  Raw data: {len(df)} rows")

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

        print(f"\n  Generating 3D conformers + building graphs...")
        t0 = time.time()
        data_list = build_graph_dataset(smiles_list, targets)
        elapsed = time.time() - t0
        print(f"  3D generation time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

        print(f"  Caching graphs to {PROCESSED_GRAPHS}...")
        torch.save(data_list, PROCESSED_GRAPHS)

    # Split
    train_idx, valid_idx, test_idx = create_split_indices(
        len(data_list), random_state=SEED)

    train_data = [data_list[i] for i in train_idx]
    valid_data = [data_list[i] for i in valid_idx]
    test_data = [data_list[i] for i in test_idx]

    # Target standardization
    train_y = np.stack([d.y.squeeze(0).numpy() for d in train_data])
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    for d in train_data + valid_data + test_data:
        d.y = (d.y - torch.tensor(y_mean)) / torch.tensor(y_std)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Build model
    print(f"\n  SchNet config: hidden={HIDDEN}, filters={NUM_FILTERS}, "
          f"interactions={NUM_INTERACTIONS}, gaussians={NUM_GAUSSIANS}, cutoff={CUTOFF}")
    print(f"  Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    model = SchNetWrapper(
        hidden_channels=HIDDEN,
        num_filters=NUM_FILTERS,
        num_interactions=NUM_INTERACTIONS,
        num_gaussians=NUM_GAUSSIANS,
        cutoff=CUTOFF,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=12, min_lr=1e-6)
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
            torch.save(model.state_dict(), MODELS_DIR / "gnn_schnet_3d.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | val_MAE={val_mae:.4f} | "
                  f"best={best_val_mae:.4f}@{best_epoch} | lr={optimizer.param_groups[0]['lr']:.1e} | {elapsed:.1f}s")

        if epoch - best_epoch >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    pd.DataFrame(log_rows).to_csv(OUT_DIR / "schnet_training_log.csv", index=False)

    # Load best and evaluate
    model.load_state_dict(torch.load(MODELS_DIR / "gnn_schnet_3d.pt", weights_only=True))
    test_pred, test_true = evaluate(model, test_loader, device)
    test_pred_real = test_pred * y_std + y_mean
    test_true_real = test_true * y_std + y_mean

    m = regression_metrics(test_true_real, test_pred_real)

    print(f"\n{'='*50}")
    print(f"  SchNet 3D Test Results")
    print(f"{'='*50}")
    for t in TARGET_COLS:
        print(f"  {t:5s}: MAE={m[t]['mae']:.4f} R2={m[t]['r2']:.4f}")
    print(f"  avg  : MAE={m['average']['mae']:.4f} R2={m['average']['r2']:.4f}")

    # Compare with AttentiveFP
    afp_path = OUT_DIR / "gnn_metrics.json"
    if afp_path.exists():
        import json
        with open(afp_path) as f:
            afp = json.load(f)
        afp_mae = afp["metrics"]["average"]["mae"]
        afp_r2 = afp["metrics"]["average"]["r2"]
        print(f"\n  vs AttentiveFP(2D): MAE={afp_mae:.4f} R2={afp_r2:.4f}")
        diff = m["average"]["r2"] - afp_r2
        print(f"  Improvement: R2 {'+'if diff>=0 else ''}{diff:.4f}")

    save_json({
        "model": "SchNet_3D",
        "hidden": HIDDEN,
        "num_filters": NUM_FILTERS,
        "num_interactions": NUM_INTERACTIONS,
        "num_gaussians": NUM_GAUSSIANS,
        "cutoff": CUTOFF,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "epochs_trained": len(log_rows),
        "metrics": m,
    }, OUT_DIR / "schnet_metrics.json")

    print(f"\n  Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
