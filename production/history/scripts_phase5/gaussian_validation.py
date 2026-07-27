"""
Phase 5: Gaussian validation pipeline.

1. Load 10 commercial molecules
2. Predict HOMO/LUMO/gap with SchNet tuned model
3. Generate Gaussian 16 input files (.gjf)

Usage:
  python scripts/phase5/gaussian_validation.py

Output:
  results/phase5/gaussian_validation/predictions.csv
  results/phase5/gaussian_validation/gjf/*.gjf
  results/phase5/gaussian_validation/submit_all.sh
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from molgap.utils import (
    MODELS_DIR,
    RESULTS_DIR,
    TARGET_COLS,
    canonicalize_smiles,
    ensure_dirs,
    save_json,
)
from molgap.schnet import SchNetWrapper
from molgap.constants import COMMERCIAL_DIR
INPUT_CSV = COMMERCIAL_DIR / "gaussian_validation_10.csv"
OUT_DIR = RESULTS_DIR / "phase5" / "gaussian_validation"
GJF_DIR = OUT_DIR / "gjf"
SCHNET_WEIGHTS = MODELS_DIR / "gnn_schnet_3d_tuned.pt"

BEST_PARAMS = {
    "hidden_channels": 256,
    "num_filters": 192,
    "num_interactions": 7,
    "num_gaussians": 50,
    "cutoff": 7.0,
    "dropout": 0.0,
}

PROCESSED_GRAPHS = RESULTS_DIR / "phase4" / "pyg_3d_graphs.pt"
SEED = 42


# ── 3D conformer generation ───────────────────────────────────

def smiles_to_3d_mol(smiles):
    """SMILES -> RDKit mol with 3D coordinates (ETKDG + MMFF opt)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_h = AllChem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    if result != 0:
        result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    if result != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        pass
    return mol_h


def mol_to_pyg_data(mol_3d):
    """RDKit mol with 3D coords -> PyG Data (no targets)."""
    from rdkit import Chem
    from torch_geometric.data import Data

    mol_noh = Chem.RemoveHs(mol_3d)
    conf = mol_noh.GetConformer()
    n_atoms = mol_noh.GetNumAtoms()
    if n_atoms == 0:
        return None

    z = torch.tensor([a.GetAtomicNum() for a in mol_noh.GetAtoms()], dtype=torch.long)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    return Data(z=z, pos=pos)


# ── Gaussian input file generation ────────────────────────────

def mol_to_gjf(mol_3d, name, charge=0, multiplicity=1):
    """Generate Gaussian 16 input file content from RDKit mol with 3D coords."""
    from rdkit import Chem

    conf = mol_3d.GetConformer()
    lines = [
        f"%chk={name}.chk",
        "#p B3LYP/6-31G(d) opt freq",
        "",
        f"{name} - B3LYP/6-31G(d) optimization",
        "",
        f"{charge} {multiplicity}",
    ]

    for atom in mol_3d.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        symbol = atom.GetSymbol()
        lines.append(f" {symbol:<2s}  {pos.x:14.8f}  {pos.y:14.8f}  {pos.z:14.8f}")

    lines.append("")
    lines.append("")
    return "\n".join(lines)


# ── SchNet prediction ─────────────────────────────────────────

def get_y_stats():
    """Get y_mean and y_std from training data (same as training script)."""
    from molgap.utils import create_split_indices

    data_list = torch.load(PROCESSED_GRAPHS, weights_only=False)
    train_idx, _, _ = create_split_indices(len(data_list), random_state=SEED)
    train_y = np.stack([data_list[i].y.squeeze(0).numpy() for i in train_idx])
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std[y_std < 1e-6] = 1.0
    return y_mean, y_std


@torch.no_grad()
def predict_schnet(model, data_list, y_mean, y_std, device):
    """Predict with SchNet, return de-standardized predictions."""
    from torch_geometric.loader import DataLoader

    model.eval()
    loader = DataLoader(data_list, batch_size=len(data_list))
    for batch in loader:
        batch = batch.to(device)
        with torch.amp.autocast("cuda"):
            out = model(batch.z, batch.pos, batch.batch)
        pred = out.cpu().numpy()
    return pred * y_std + y_mean


# ── Main ──────────────────────────────────────────────────────

def main():
    from rdkit import Chem

    ensure_dirs(OUT_DIR, GJF_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Phase 5: Gaussian Validation Pipeline ===", flush=True)
    print(f"  Device: {device}", flush=True)

    # Load molecules
    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df)} commercial molecules", flush=True)

    # Canonicalize SMILES and generate 3D
    print(f"\n  Generating 3D conformers...", flush=True)
    results = []
    pyg_data_list = []
    mol_3d_list = []

    for _, row in df.iterrows():
        name = row["name"]
        smiles = row["smiles"]
        can = canonicalize_smiles(smiles)

        if can is None:
            print(f"    {name}: invalid SMILES, skipping", flush=True)
            results.append({**row.to_dict(), "canonical_smiles": None, "status": "invalid_smiles"})
            continue

        mol_3d = smiles_to_3d_mol(can)
        if mol_3d is None:
            print(f"    {name}: 3D generation failed, skipping", flush=True)
            results.append({**row.to_dict(), "canonical_smiles": can, "status": "3d_failed"})
            continue

        pyg = mol_to_pyg_data(mol_3d)
        if pyg is None:
            print(f"    {name}: graph conversion failed, skipping", flush=True)
            results.append({**row.to_dict(), "canonical_smiles": can, "status": "graph_failed"})
            continue

        pyg_data_list.append(pyg)
        mol_3d_list.append((name, mol_3d, can))
        results.append({**row.to_dict(), "canonical_smiles": can, "status": "ok",
                        "pyg_idx": len(pyg_data_list) - 1})
        print(f"    {name}: OK ({Chem.RemoveHs(mol_3d).GetNumAtoms()} heavy atoms)", flush=True)

    # Load SchNet model
    print(f"\n  Loading SchNet tuned model...", flush=True)
    y_mean, y_std = get_y_stats()

    model = SchNetWrapper(**BEST_PARAMS).to(device)
    state = torch.load(SCHNET_WEIGHTS, weights_only=True, map_location=device)
    model.load_state_dict(state)
    print(f"  Model loaded", flush=True)

    # Predict
    if pyg_data_list:
        preds = predict_schnet(model, pyg_data_list, y_mean, y_std, device)
        print(f"\n  Predictions (SchNet tuned):", flush=True)
        print(f"  {'Name':<25s} {'HOMO':>8s} {'LUMO':>8s} {'Gap':>8s}", flush=True)
        print(f"  {'-'*51}", flush=True)

        for r in results:
            if r["status"] == "ok":
                idx = r["pyg_idx"]
                r["homo_pred"] = float(preds[idx, 0])
                r["lumo_pred"] = float(preds[idx, 1])
                r["gap_pred"] = float(preds[idx, 2])
                print(f"  {r['name']:<25s} {r['homo_pred']:8.4f} {r['lumo_pred']:8.4f} {r['gap_pred']:8.4f}",
                      flush=True)

    # Generate Gaussian input files
    print(f"\n  Generating Gaussian input files...", flush=True)
    gjf_names = []
    for name, mol_3d, can in mol_3d_list:
        safe_name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        gjf_content = mol_to_gjf(mol_3d, safe_name)
        gjf_path = GJF_DIR / f"{safe_name}.gjf"
        gjf_path.write_text(gjf_content, encoding="utf-8")
        gjf_names.append(safe_name)
        print(f"    {gjf_path.name}", flush=True)

    # Generate submit script
    submit_lines = ["#!/bin/bash", f"# Submit all {len(gjf_names)} Gaussian jobs", ""]
    for gn in gjf_names:
        submit_lines.append(f"g16sub {gn}.gjf")
    submit_lines.append("")
    submit_lines.append("echo 'All jobs submitted.'")
    submit_path = OUT_DIR / "submit_all.sh"
    submit_path.write_text("\n".join(submit_lines), encoding="utf-8")
    print(f"\n  Submit script: {submit_path}", flush=True)

    # Save predictions CSV
    out_df = pd.DataFrame(results)
    cols = ["name", "supplier", "smiles", "canonical_smiles", "mw_approx", "category",
            "homo_pred", "lumo_pred", "gap_pred", "status"]
    for c in cols:
        if c not in out_df.columns:
            out_df[c] = np.nan
    out_df[cols].to_csv(OUT_DIR / "predictions.csv", index=False, encoding="utf-8")

    save_json({
        "n_molecules": len(df),
        "n_predicted": len(pyg_data_list),
        "n_failed": len(df) - len(pyg_data_list),
        "model": "SchNet_3D_optuna",
        "gjf_dir": str(GJF_DIR),
    }, OUT_DIR / "validation_summary.json")

    print(f"\n  Results: {OUT_DIR / 'predictions.csv'}", flush=True)
    print(f"  GJF files: {GJF_DIR}/", flush=True)
    print(f"\n  Next: upload gjf/ to IMS server and run submit_all.sh", flush=True)


if __name__ == "__main__":
    main()
