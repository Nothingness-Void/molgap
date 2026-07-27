"""
Phase 4 Step 4b: SchNet (3D) for HOMO/LUMO/gap prediction.

Supports two 3D coordinate sources:
  - PM6 optimized geometry from PubChemQC (via parquet, preferred)
  - RDKit ETKDG fallback when PM6 coords unavailable

Optionally injects per-atom Gasteiger partial charges.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

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
from molgap.schnet import SchNetWrapper

OUT_DIR = RESULTS_DIR / "phase4"
RAW_CSV = RAW_DIR / "phase3_chonsfcl_mw200_1000_30k.csv"
PROCESSED_GRAPHS = RESULTS_DIR / "phase4" / "pyg_3d_graphs_etkdg.pt"
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
USE_CHARGES = True


def load_pm6_lookup():
    """Load PM6 coordinate lookup from parquet, keyed by CID."""
    if not COORDS_PARQUET.exists():
        return {}
    import pyarrow.parquet as pq
    table = pq.read_table(COORDS_PARQUET)
    df = table.to_pandas()
    lookup = {}
    for _, row in df.iterrows():
        lookup[int(row["cid"])] = (row["atomic_numbers"], row["coordinates"])
    return lookup


def generate_3d_coords(mol):
    """Generate 3D conformer for a molecule using ETKDG (fallback)."""
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


def compute_gasteiger_charges(mol):
    """Compute Gasteiger partial charges, return list of floats."""
    from rdkit.Chem import AllChem
    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    for atom in mol.GetAtoms():
        c = atom.GetDoubleProp('_GasteigerCharge')
        charges.append(0.0 if np.isnan(c) or np.isinf(c) else c)
    return charges


def mol_to_pyg_data_from_pm6(atomic_numbers, coordinates, targets,
                              smiles=None, use_charges=True):
    """Build PyG Data from PM6 coordinates (includes hydrogens)."""
    from torch_geometric.data import Data
    from rdkit import Chem

    z = torch.tensor(atomic_numbers, dtype=torch.long)
    pos = torch.tensor(coordinates, dtype=torch.float64).reshape(-1, 3).float()

    data = Data(z=z, pos=pos)
    data.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(0)

    if use_charges and smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            from rdkit.Chem import AllChem
            mol_h = AllChem.AddHs(mol)
            charges = compute_gasteiger_charges(mol_h)
            if len(charges) == len(atomic_numbers):
                data.charges = torch.tensor(charges, dtype=torch.float32)

    return data


def mol_to_pyg_data_from_etkdg(mol_3d, targets, use_charges=True, remove_h=False):
    """Convert RDKit mol with 3D coords to PyG Data object."""
    from torch_geometric.data import Data
    from rdkit import Chem

    if remove_h:
        mol_3d = Chem.RemoveHs(mol_3d)

    conf = mol_3d.GetConformer()
    n_atoms = mol_3d.GetNumAtoms()
    if n_atoms == 0:
        return None

    z = torch.tensor([atom.GetAtomicNum() for atom in mol_3d.GetAtoms()],
                     dtype=torch.long)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)

    data = Data(z=z, pos=pos)
    data.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(0)

    if use_charges:
        charges = compute_gasteiger_charges(mol_3d)
        data.charges = torch.tensor(charges, dtype=torch.float32)

    return data


def build_graph_dataset(smiles_list, targets_array, cid_list=None,
                         pm6_lookup=None, use_charges=True):
    """Convert SMILES list to 3D PyG Data objects, preferring PM6 coords."""
    from rdkit import Chem

    data_list = []
    failed = 0
    n_pm6 = 0
    n_etkdg = 0

    for i, smi in enumerate(smiles_list):
        cid = int(cid_list[i]) if cid_list is not None else None

        if pm6_lookup and cid and cid in pm6_lookup:
            atomic_nums, coords = pm6_lookup[cid]
            data = mol_to_pyg_data_from_pm6(
                atomic_nums, coords, targets_array[i],
                smiles=smi, use_charges=use_charges)
            if data is not None:
                data_list.append(data)
                n_pm6 += 1
                if (i + 1) % 5000 == 0:
                    print(f"    {i+1}/{len(smiles_list)} done (pm6={n_pm6}, etkdg={n_etkdg}, fail={failed})")
                continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue

        mol_3d = generate_3d_coords(mol)
        if mol_3d is None:
            failed += 1
            continue

        data = mol_to_pyg_data_from_etkdg(mol_3d, targets_array[i],
                                            use_charges=use_charges)
        if data is None:
            failed += 1
            continue

        data_list.append(data)
        n_etkdg += 1

        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(smiles_list)} done (pm6={n_pm6}, etkdg={n_etkdg}, fail={failed})")

    print(f"  Total: {len(data_list)} graphs (PM6={n_pm6}, ETKDG={n_etkdg}), {failed} failed")
    return data_list


def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            charges = getattr(batch, 'charges', None)
            out = model(batch.z, batch.pos, batch.batch, charges=charges)
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
            charges = getattr(batch, 'charges', None)
            out = model(batch.z, batch.pos, batch.batch, charges=charges)
        preds.append(out.cpu().numpy())
        trues.append(batch.y.cpu().numpy())
    return np.vstack(preds), np.vstack(trues)


def main():
    from torch_geometric.loader import DataLoader

    ensure_dirs(OUT_DIR, MODELS_DIR)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase 4 Step 4b: SchNet 3D (ETKDG + Gasteiger) ===")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

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

        print(f"\n  Building ETKDG graphs (use_charges={USE_CHARGES})...")
        t0 = time.time()
        data_list = build_graph_dataset(
            smiles_list, targets, cid_list=None,
            pm6_lookup=None, use_charges=USE_CHARGES)
        elapsed = time.time() - t0
        print(f"  Graph build time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

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

    has_charges = hasattr(data_list[0], 'charges')
    print(f"\n  SchNet config: hidden={HIDDEN}, filters={NUM_FILTERS}, "
          f"interactions={NUM_INTERACTIONS}, gaussians={NUM_GAUSSIANS}, cutoff={CUTOFF}")
    print(f"  Gasteiger charges: {has_charges}")
    print(f"  Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    model = SchNetWrapper(
        hidden_channels=HIDDEN,
        num_filters=NUM_FILTERS,
        num_interactions=NUM_INTERACTIONS,
        num_gaussians=NUM_GAUSSIANS,
        cutoff=CUTOFF,
        use_charges=has_charges,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=12, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")

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

    model.load_state_dict(torch.load(MODELS_DIR / "gnn_schnet_3d.pt", weights_only=True))
    test_pred, test_true = evaluate(model, test_loader, device)
    test_pred_real = test_pred * y_std + y_mean
    test_true_real = test_true * y_std + y_mean

    m = regression_metrics(test_true_real, test_pred_real)

    print(f"\n{'='*50}")
    print(f"  SchNet 3D Test Results (ETKDG + Gasteiger)")
    print(f"{'='*50}")
    for t in TARGET_COLS:
        print(f"  {t:5s}: MAE={m[t]['mae']:.4f} R2={m[t]['r2']:.4f}")
    print(f"  avg  : MAE={m['average']['mae']:.4f} R2={m['average']['r2']:.4f}")

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
        "model": "SchNet_3D_ETKDG_Gasteiger",
        "hidden": HIDDEN,
        "num_filters": NUM_FILTERS,
        "num_interactions": NUM_INTERACTIONS,
        "num_gaussians": NUM_GAUSSIANS,
        "cutoff": CUTOFF,
        "use_charges": has_charges,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "epochs_trained": len(log_rows),
        "metrics": m,
    }, OUT_DIR / "schnet_metrics.json")

    print(f"\n  Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
