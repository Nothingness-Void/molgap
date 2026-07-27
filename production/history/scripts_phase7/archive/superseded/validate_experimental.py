"""
Validate SchNet 300k against experimental OLED molecule data.
SMILES are hardcoded in the CSV for known molecules.

Usage:
  .venv\Scripts\python.exe scripts/phase7/validate_experimental.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from molgap.constants import MODELS_DIR, RESULTS_DIR, COMMERCIAL_DIR
from molgap.graphs import smiles_to_pyg
from molgap.schnet import SchNetWrapper

OUT_DIR = RESULTS_DIR / "phase7" / "experimental_validation"
DATA_PATH = COMMERCIAL_DIR / "oled_experimental_v2.csv"

MODEL_300K = MODELS_DIR / "gnn_schnet_3d_300k.pt"
PARAMS_300K = {
    "hidden_channels": 192,
    "num_filters": 192,
    "num_interactions": 6,
    "num_gaussians": 50,
    "cutoff": 6.0,
    "dropout": 0.0,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} molecules with SMILES")

    # Generate 3D graphs
    print("\n--- Generating ETKDG conformers ---")
    pyg_list, valid_idx = [], []
    for i, row in df.iterrows():
        g = smiles_to_pyg(row["smiles"])
        if g is not None:
            pyg_list.append(g)
            valid_idx.append(i)
            print(f"  {row['name']}: OK")
        else:
            print(f"  {row['name']}: 3D FAILED")
    print(f"  Success: {len(pyg_list)}/{len(df)}")

    # Load model
    print("\n--- Loading SchNet 300k model ---")
    model = SchNetWrapper(**PARAMS_300K, use_charges=True).to(device)
    model.load_state_dict(
        torch.load(str(MODEL_300K), weights_only=False, map_location=device)
    )
    model.eval()

    # Predict
    loader = DataLoader(pyg_list, batch_size=64)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            charges = batch.charges if hasattr(batch, "charges") else None
            with torch.amp.autocast("cuda"):
                out = model(batch.z, batch.pos, batch.batch, charges=charges)
            preds.append(out.cpu().numpy())
    preds = np.vstack(preds)

    # Compare
    df_result = df.loc[valid_idx].reset_index(drop=True)
    df_result["homo_pred"] = preds[:, 0]
    df_result["lumo_pred"] = preds[:, 1]
    df_result["gap_pred"] = preds[:, 2]
    df_result["homo_err"] = df_result["homo_pred"] - df_result["homo_exp"]
    df_result["lumo_err"] = df_result["lumo_pred"] - df_result["lumo_exp"]
    df_result["gap_err"] = df_result["gap_pred"] - df_result["gap_exp"]

    print(f"\n{'='*90}")
    print(f"  Experimental Validation: SchNet 300k vs Literature ({len(df_result)} molecules)")
    print(f"{'='*90}")
    print(f"  {'Name':<16s} {'H_exp':>6s} {'H_pred':>6s} {'err':>6s}  "
          f"{'L_exp':>6s} {'L_pred':>6s} {'err':>6s}  "
          f"{'G_exp':>5s} {'G_pred':>6s} {'err':>6s}")
    print(f"  {'-'*78}")

    for _, row in df_result.iterrows():
        print(f"  {row['name']:<16s} {row['homo_exp']:6.2f} {row['homo_pred']:6.2f} {row['homo_err']:+6.2f}  "
              f"{row['lumo_exp']:6.2f} {row['lumo_pred']:6.2f} {row['lumo_err']:+6.2f}  "
              f"{row['gap_exp']:5.2f} {row['gap_pred']:6.2f} {row['gap_err']:+6.2f}")

    # Summary
    print(f"\n--- Summary (B3LYP prediction vs experimental) ---")
    for prop in ["homo", "lumo", "gap"]:
        errs = df_result[f"{prop}_err"].values
        mae = np.mean(np.abs(errs))
        me = np.mean(errs)
        rmse = np.sqrt(np.mean(errs**2))
        print(f"  {prop.upper():5s}: MAE={mae:.3f} eV, ME={me:+.3f} eV, RMSE={rmse:.3f} eV")

    homo_bias = df_result["homo_err"].mean()
    lumo_bias = df_result["lumo_err"].mean()
    print(f"\n--- B3LYP vs Experimental Systematic Bias ---")
    print(f"  HOMO: B3LYP {homo_bias:+.3f} eV vs experiment (literature: +0.5 to +0.7 eV)")
    print(f"  LUMO: B3LYP {lumo_bias:+.3f} eV vs experiment (literature: +1.3 to +2.1 eV)")
    print(f"  Gap is more comparable since systematic errors partially cancel.")

    # Save
    df_result.to_csv(OUT_DIR / "experimental_comparison_300k.csv", index=False)
    summary = {}
    for prop in ["homo", "lumo", "gap"]:
        errs = df_result[f"{prop}_err"].values
        summary[prop] = {
            "mae": float(np.mean(np.abs(errs))),
            "me": float(np.mean(errs)),
            "rmse": float(np.sqrt(np.mean(errs**2))),
        }
    summary["n_molecules"] = len(df_result)
    with open(OUT_DIR / "experimental_summary_300k.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
