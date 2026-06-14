"""
Validate SchNet 300k against HOPV15 experimental dataset (small molecules).

Usage:
  .venv\Scripts\python.exe scripts/phase7/validate_hopv.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

from molgap.constants import MODELS_DIR, RESULTS_DIR, COMMERCIAL_DIR
from molgap.graphs import smiles_to_pyg
from molgap.schnet import SchNetWrapper

OUT_DIR = RESULTS_DIR / "phase7" / "hopv_validation"
HOPV_PATH = COMMERCIAL_DIR / "HOPV_15_revised_2.data"

MODEL_300K = MODELS_DIR / "gnn_schnet_3d_300k.pt"
PARAMS_300K = {
    "hidden_channels": 192,
    "num_filters": 192,
    "num_interactions": 6,
    "num_gaussians": 50,
    "cutoff": 6.0,
    "dropout": 0.0,
}


def parse_hopv(path):
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if (line and not line.startswith("InChI") and "," not in line
                and not line[0].isdigit() and not line.startswith("Conformer")):
            smiles = line
            if i + 2 < len(lines):
                parts = lines[i + 2].strip().split(",")
                if len(parts) == 13:
                    records.append({
                        "smiles": smiles,
                        "doi": parts[0],
                        "inchikey": parts[1],
                        "type": parts[2],
                        "homo_exp": parts[5],
                        "lumo_exp": parts[6],
                        "gap_exp": parts[7],
                        "optical_gap": parts[8],
                    })
        i += 1

    # Filter small molecules with valid HOMO+LUMO+Gap
    valid = []
    for r in records:
        if r["type"] != "molecule":
            continue
        try:
            h = float(r["homo_exp"])
            l = float(r["lumo_exp"])
            g = float(r["gap_exp"])
            if np.isnan(h) or np.isnan(l) or np.isnan(g):
                continue
            r["homo_exp"] = h
            r["lumo_exp"] = l
            r["gap_exp"] = g
            r["optical_gap"] = float(r["optical_gap"]) if r["optical_gap"] != "nan" else np.nan
            valid.append(r)
        except ValueError:
            continue

    return pd.DataFrame(valid)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Parse HOPV
    df = parse_hopv(HOPV_PATH)
    print(f"HOPV15 small molecules with HOMO+LUMO+Gap: {len(df)}")
    print(f"  HOMO range: [{df['homo_exp'].min():.2f}, {df['homo_exp'].max():.2f}] eV")
    print(f"  LUMO range: [{df['lumo_exp'].min():.2f}, {df['lumo_exp'].max():.2f}] eV")
    print(f"  Gap range:  [{df['gap_exp'].min():.2f}, {df['gap_exp'].max():.2f}] eV")

    # Generate 3D conformers
    print(f"\n--- Generating ETKDG conformers ---")
    pyg_list, valid_idx = [], []
    for i, row in df.iterrows():
        g = smiles_to_pyg(row["smiles"])
        if g is not None:
            pyg_list.append(g)
            valid_idx.append(i)
        else:
            print(f"  FAILED: {row['smiles'][:60]}")
    print(f"  3D success: {len(pyg_list)}/{len(df)}")

    # Load model
    print(f"\n--- Loading SchNet 300k model ---")
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

    # Build result
    df_result = df.loc[valid_idx].reset_index(drop=True)
    df_result["homo_pred"] = preds[:, 0]
    df_result["lumo_pred"] = preds[:, 1]
    df_result["gap_pred"] = preds[:, 2]
    df_result["homo_err"] = df_result["homo_pred"] - df_result["homo_exp"]
    df_result["lumo_err"] = df_result["lumo_pred"] - df_result["lumo_exp"]
    df_result["gap_err"] = df_result["gap_pred"] - df_result["gap_exp"]

    # Per-molecule table
    print(f"\n{'='*90}")
    print(f"  HOPV15 Validation: SchNet 300k ({len(df_result)} small molecules)")
    print(f"{'='*90}")
    print(f"  {'#':>3s}  {'H_exp':>6s} {'H_pred':>6s} {'err':>6s}  "
          f"{'L_exp':>6s} {'L_pred':>6s} {'err':>6s}  "
          f"{'G_exp':>5s} {'G_pred':>6s} {'err':>6s}")
    print(f"  {'-'*78}")

    for idx, row in df_result.iterrows():
        print(f"  {idx+1:3d}  {row['homo_exp']:6.2f} {row['homo_pred']:6.2f} {row['homo_err']:+6.2f}  "
              f"{row['lumo_exp']:6.2f} {row['lumo_pred']:6.2f} {row['lumo_err']:+6.2f}  "
              f"{row['gap_exp']:5.2f} {row['gap_pred']:6.2f} {row['gap_err']:+6.2f}")

    # Summary
    print(f"\n--- Summary ---")
    for prop in ["homo", "lumo", "gap"]:
        errs = df_result[f"{prop}_err"].values
        mae = np.mean(np.abs(errs))
        me = np.mean(errs)
        rmse = np.sqrt(np.mean(errs**2))
        r2 = r2_score(df_result[f"{prop}_exp"], df_result[f"{prop}_pred"])
        print(f"  {prop.upper():5s}: MAE={mae:.3f} eV, ME={me:+.3f} eV, RMSE={rmse:.3f} eV, R2={r2:.3f}")

    # Bias analysis
    print(f"\n--- B3LYP vs Experimental Systematic Bias ---")
    print(f"  HOMO: B3LYP {df_result['homo_err'].mean():+.3f} eV vs experiment")
    print(f"  LUMO: B3LYP {df_result['lumo_err'].mean():+.3f} eV vs experiment")
    print(f"  Gap:  B3LYP {df_result['gap_err'].mean():+.3f} eV vs experiment")

    # Corrected metrics (subtract systematic bias)
    print(f"\n--- After Linear Bias Correction ---")
    for prop in ["homo", "lumo", "gap"]:
        bias = df_result[f"{prop}_err"].mean()
        corrected = df_result[f"{prop}_pred"] - bias
        mae = mean_absolute_error(df_result[f"{prop}_exp"], corrected)
        r2 = r2_score(df_result[f"{prop}_exp"], corrected)
        print(f"  {prop.upper():5s}: MAE={mae:.3f} eV, R2={r2:.3f}")

    # Save
    df_result.to_csv(OUT_DIR / "hopv_comparison_300k.csv", index=False)
    summary = {"n_molecules": len(df_result)}
    for prop in ["homo", "lumo", "gap"]:
        errs = df_result[f"{prop}_err"].values
        summary[prop] = {
            "mae": float(np.mean(np.abs(errs))),
            "me": float(np.mean(errs)),
            "rmse": float(np.sqrt(np.mean(errs**2))),
            "r2": float(r2_score(df_result[f"{prop}_exp"], df_result[f"{prop}_pred"])),
        }
    with open(OUT_DIR / "hopv_summary_300k.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
