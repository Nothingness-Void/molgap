"""Inject 2D RDKit descriptors into existing PyG graph cache for Kaggle upload."""
import numpy as np
import pandas as pd
import torch

from molgap.constants import RESULTS_DIR, DATA_PHASE3, DATA_PHASE6_LARGE, GRAPHS_PHASE6
from molgap.utils import calc_rdkit_descriptors, canonicalize_smiles, ensure_dirs, save_json

GRAPH_PATH = GRAPHS_PHASE6
OUT_PATH = RESULTS_DIR / "phase7" / "pyg_3d_graphs_etkdg_expanded_with_desc.pt"

def main():
    from rdkit import Chem
    ensure_dirs(OUT_PATH.parent)

    # Load CSVs
    dfs = []
    for p in [DATA_PHASE3, DATA_PHASE6_LARGE]:
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

    print(f"Computing 2D descriptors for {len(df)} molecules...")
    desc_rows = []
    for i, smi in enumerate(df["canonical_smiles"]):
        mol = Chem.MolFromSmiles(smi)
        desc_rows.append(calc_rdkit_descriptors(mol) if mol else {})
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(df)}")

    desc_df = pd.DataFrame(desc_rows).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    std = desc_df.std()
    keep = std[std > 1e-8].index.tolist()
    desc_df = desc_df[keep]
    print(f"  {len(keep)} descriptors after dropping constants")

    desc_arr = desc_df.values.astype(np.float32)
    desc_mean = desc_arr.mean(axis=0)
    desc_std = desc_arr.std(axis=0)
    desc_std[desc_std < 1e-8] = 1.0
    desc_arr = (desc_arr - desc_mean) / desc_std

    # Load graphs and inject
    print("Loading graphs...")
    data_list = torch.load(GRAPH_PATH, weights_only=False)
    print(f"  {len(data_list)} graphs")

    n = min(len(data_list), len(desc_arr))
    for i in range(n):
        data_list[i].desc = torch.tensor(desc_arr[i], dtype=torch.float32)
    data_list = data_list[:n]

    # Save desc normalization stats
    torch.save(data_list, OUT_PATH)
    print(f"Saved {len(data_list)} graphs with {len(keep)} desc to {OUT_PATH}")
    print(f"File size: {OUT_PATH.stat().st_size / 1e9:.2f} GB")

    # Save desc stats for inference
    save_json({
        "n_desc": len(keep),
        "desc_names": keep,
        "desc_mean": desc_mean.tolist(),
        "desc_std": desc_std.tolist(),
    }, RESULTS_DIR / "phase7" / "desc_normalization.json")
    print("Saved desc normalization stats")

if __name__ == "__main__":
    main()
