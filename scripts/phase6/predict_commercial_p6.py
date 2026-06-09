"""Predict commercial OLED molecules with Phase 6 expanded SchNet model."""
import sys, json, numpy as np, pandas as pd, torch
sys.path.insert(0, r"D:\文档\molgap\src")
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from molgap.utils import TARGET_COLS, create_split_indices, compute_gasteiger_charges
from molgap.schnet import SchNetWrapper

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r"D:\文档\molgap\data\commercial\gaussian_validation_10.csv")
print(f"Loaded {len(df)} commercial molecules")

def smiles_to_pyg(smi, name=""):
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
    if n == 0:
        return None
    conf = mol_h.GetConformer()
    z = torch.tensor([mol_h.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)], dtype=torch.long)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    charges = compute_gasteiger_charges(mol_h)
    data = Data(z=z, pos=pos, charges=torch.tensor(charges, dtype=torch.float32))
    print(f"  {name}: {n} atoms")
    return data

pyg_list, valid_idx = [], []
for i, row in df.iterrows():
    d = smiles_to_pyg(row["smiles"], row["name"])
    if d is not None:
        pyg_list.append(d)
        valid_idx.append(i)

print(f"3D success: {len(pyg_list)}/{len(df)}")

# y_mean/y_std from Phase 6 expanded training data
graphs = torch.load(r"D:\文档\molgap\results\phase6\pyg_3d_graphs_etkdg_expanded.pt", weights_only=False)
train_idx, _, _ = create_split_indices(len(graphs), random_state=SEED)
train_y = np.stack([graphs[i].y.squeeze(0).numpy() for i in train_idx])
y_mean = train_y.mean(axis=0)
y_std = train_y.std(axis=0)
y_std[y_std < 1e-6] = 1.0
del graphs
print(f"y_mean={y_mean}, y_std={y_std}")

# Load Phase 6 model
params = {"hidden_channels": 192, "num_filters": 256, "num_interactions": 6,
          "num_gaussians": 100, "cutoff": 8.0, "dropout": 0.1}
model = SchNetWrapper(**params, use_charges=True).to(device)
model.load_state_dict(torch.load(r"D:\文档\molgap\models\gnn_schnet_3d_optuna_expanded.pt",
                                  weights_only=True, map_location=device))
model.eval()

loader = DataLoader(pyg_list, batch_size=len(pyg_list))
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.z, batch.pos, batch.batch, charges=batch.charges)
        preds = out.cpu().numpy() * y_std + y_mean

# Load Phase 4 predictions from Gaussian validation (canonical P4 source)
p4 = pd.read_csv(r"D:\文档\molgap\results\phase5\gaussian_validation\ml_vs_gaussian.csv")

result = df.loc[valid_idx].reset_index(drop=True).copy()
for i, t in enumerate(TARGET_COLS):
    result[f"{t}_pred_eV"] = preds[:, i]

print()
print("=" * 100)
print("  Phase 6 vs Phase 4 vs Gaussian B3LYP (eV)")
print("=" * 100)
fmt = "{:15s} {:12s} | {:>7s} {:>7s} {:>7s} | {:>7s} {:>7s} {:>7s} | {:>7s} {:>7s} {:>7s}"
print(fmt.format("Name", "Category", "G HOMO", "P4", "P6", "G LUMO", "P4", "P6", "G Gap", "P4", "P6"))
print("-" * 100)
for idx, row in result.iterrows():
    name = row["name"]
    g = p4[p4["name"] == name].iloc[0]
    print(fmt.format(name, row["category"],
                     f"{g['homo_gaussian']:.3f}", f"{g['homo_pred']:.3f}", f"{row['homo_pred_eV']:.3f}",
                     f"{g['lumo_gaussian']:.3f}", f"{g['lumo_pred']:.3f}", f"{row['lumo_pred_eV']:.3f}",
                     f"{g['gap_gaussian']:.3f}", f"{g['gap_pred']:.3f}", f"{row['gap_pred_eV']:.3f}"))

result.to_csv(r"D:\文档\molgap\results\phase6\commercial_predictions_p6.csv", index=False)
print(f"\nSaved to results/phase6/commercial_predictions_p6.csv")
