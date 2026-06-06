"""Predict HOMO/LUMO/Gap for 10 commercial OLED molecules using tuned SchNet."""
import sys, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"D:\文档\molgap\src")
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from molgap.utils import (
    MODELS_DIR, RESULTS_DIR, TARGET_COLS, create_split_indices, ensure_dirs,
    compute_gasteiger_charges,
    generate_pm6_coords_mopac,
)
from molgap.schnet import SchNetWrapper

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = RESULTS_DIR / "phase5" / "commercial"
ensure_dirs(OUT_DIR)

df = pd.read_csv(r"D:\文档\molgap\data\commercial\gaussian_validation_10.csv")
print(f"Loaded {len(df)} commercial molecules")

def smiles_to_pyg(smi, name=""):
    # PM6 优化几何（与训练一致）
    pm6_result = generate_pm6_coords_mopac(smi)
    if pm6_result is not None:
        atomic_nums, coords = pm6_result
        z = torch.tensor(atomic_nums, dtype=torch.long)
        pos = torch.tensor(coords, dtype=torch.float64).reshape(-1, 3).float()
        mol = Chem.MolFromSmiles(smi)
        mol_h = AllChem.AddHs(mol)
        charges = compute_gasteiger_charges(mol_h)
        if len(charges) == len(atomic_nums):
            data = Data(z=z, pos=pos, charges=torch.tensor(charges, dtype=torch.float32))
        else:
            data = Data(z=z, pos=pos)
        print(f"  {name}: {len(atomic_nums)} atoms (PM6)")
        return data

    # ETKDG fallback
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol_h = AllChem.AddHs(mol)
    if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) != 0:
        if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        pass
    n = mol_h.GetNumAtoms()
    conf = mol_h.GetConformer()
    z = torch.tensor([mol_h.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)], dtype=torch.long)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    charges = compute_gasteiger_charges(mol_h)
    data = Data(z=z, pos=pos, charges=torch.tensor(charges, dtype=torch.float32))
    print(f"  {name}: {n} atoms (ETKDG fallback)")
    return data

pyg_list = []
valid_idx = []
for i, row in df.iterrows():
    d = smiles_to_pyg(row["smiles"], row["name"])
    if d is not None:
        pyg_list.append(d)
        valid_idx.append(i)
    else:
        print(f"  {row['name']}: FAILED")

print(f"\n3D success: {len(pyg_list)}/{len(df)}")

# Load y_mean/y_std from training
graphs_path = RESULTS_DIR / "phase4" / "pyg_3d_graphs_pm6.pt"
data_list = torch.load(graphs_path, weights_only=False)
train_idx, _, _ = create_split_indices(len(data_list), random_state=SEED)
train_y = np.stack([data_list[i].y.squeeze(0).numpy() for i in train_idx])
y_mean = train_y.mean(axis=0)
y_std = train_y.std(axis=0)
y_std[y_std < 1e-6] = 1.0
del data_list

model = SchNetWrapper(
    hidden_channels=192, num_filters=256, num_interactions=6,
    num_gaussians=100, cutoff=6.0, dropout=0.2, use_charges=True
).to(device)
model.load_state_dict(torch.load(MODELS_DIR / "gnn_schnet_3d_tuned.pt",
                                  weights_only=True, map_location=device))
model.eval()

loader = DataLoader(pyg_list, batch_size=len(pyg_list))
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.z, batch.pos, batch.batch, charges=batch.charges)
        preds = out.cpu().numpy() * y_std + y_mean

result = df.loc[valid_idx].reset_index(drop=True).copy()
for i, t in enumerate(TARGET_COLS):
    result[f"{t}_pred_eV"] = preds[:, i]

print(f"\n{'='*70}")
print(f"  Commercial OLED Molecules - SchNet Predictions")
print(f"{'='*70}")
print(f"{'Name':20s} {'Category':18s} {'HOMO(eV)':>10s} {'LUMO(eV)':>10s} {'Gap(eV)':>10s}")
print("-" * 70)
for _, row in result.iterrows():
    print(f"{row['name']:20s} {row['category']:18s} {row['homo_pred_eV']:10.4f} {row['lumo_pred_eV']:10.4f} {row['gap_pred_eV']:10.4f}")

result.to_csv(OUT_DIR / "commercial_predictions.csv", index=False, encoding="utf-8")
print(f"\nSaved to {OUT_DIR / 'commercial_predictions.csv'}")
