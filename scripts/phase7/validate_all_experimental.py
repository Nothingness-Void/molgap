"""
Combined experimental validation: OLED literature + HOPV15 + additional molecules.
Runs SchNet 300k predictions on all, outputs a unified results table.

Usage:
  .venv\Scripts\python.exe scripts/phase7/validate_all_experimental.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

from molgap.constants import MODELS_DIR, RESULTS_DIR, COMMERCIAL_DIR
from molgap.graphs import smiles_to_pyg
from molgap.schnet import SchNetWrapper

OUT_DIR = RESULTS_DIR / "phase7" / "combined_validation"
OLED_CSV = COMMERCIAL_DIR / "oled_experimental_v2.csv"
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

ALLOWED_ELEMENTS = {"C", "H", "O", "N", "S", "F", "Cl"}

EXTRA_MOLECULES = [
    {
        "name": "DMAC-DPS",
        "smiles": "O=S(=O)(c1ccc(N2c3ccccc3C(C)(C)c3ccccc32)cc1)c1ccc(N2c3ccccc3C(C)(C)c3ccccc32)cc1",
        "homo_exp": -5.92, "lumo_exp": -2.92, "gap_exp": 3.00,
        "source": "Ossila/Literature", "doi": "10.1002/anie.201501521",
    },
]


def check_elements(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ELEMENTS:
            return False
    return True


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
                    try:
                        h = float(parts[5])
                        l = float(parts[6])
                        g = float(parts[7])
                        if not (np.isnan(h) or np.isnan(l) or np.isnan(g)):
                            if parts[2] == "molecule":
                                records.append({
                                    "name": f"HOPV_{len(records)+1:03d}",
                                    "smiles": smiles,
                                    "homo_exp": h, "lumo_exp": l, "gap_exp": g,
                                    "source": "HOPV15",
                                    "doi": parts[0],
                                })
                    except ValueError:
                        pass
        i += 1
    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Collect all molecules ──
    all_records = []

    # OLED literature
    oled_df = pd.read_csv(OLED_CSV)
    for _, row in oled_df.iterrows():
        all_records.append({
            "name": row["name"],
            "smiles": row["smiles"],
            "homo_exp": row["homo_exp"],
            "lumo_exp": row["lumo_exp"],
            "gap_exp": row["gap_exp"],
            "source": "OLED_Literature",
            "doi": row.get("doi", ""),
        })
    print(f"OLED literature: {len(oled_df)} molecules")

    # HOPV15
    hopv_records = parse_hopv(HOPV_PATH)
    all_records.extend(hopv_records)
    print(f"HOPV15 small molecules: {len(hopv_records)} molecules")

    # Extra molecules
    valid_extra = []
    for mol_info in EXTRA_MOLECULES:
        mol = Chem.MolFromSmiles(mol_info["smiles"])
        if mol is None:
            print(f"  SKIP {mol_info['name']}: invalid SMILES")
            continue
        if not check_elements(mol):
            elems = {a.GetSymbol() for a in mol.GetAtoms()}
            print(f"  SKIP {mol_info['name']}: unsupported elements {elems - ALLOWED_ELEMENTS}")
            continue
        mw = Descriptors.ExactMolWt(mol)
        if mw < 200 or mw > 1000:
            print(f"  SKIP {mol_info['name']}: MW={mw:.0f} out of range")
            continue
        valid_extra.append(mol_info)
    all_records.extend(valid_extra)
    print(f"Extra molecules: {len(valid_extra)} molecules")

    df = pd.DataFrame(all_records)
    print(f"\nTotal molecules: {len(df)}")

    # ── 2. Generate 3D conformers ──
    print(f"\n--- Generating ETKDG conformers ---")
    pyg_list, valid_idx = [], []
    for i, row in df.iterrows():
        g = smiles_to_pyg(row["smiles"])
        if g is not None:
            pyg_list.append(g)
            valid_idx.append(i)
        else:
            print(f"  FAILED: {row['name']} ({row['smiles'][:50]})")
    print(f"  3D success: {len(pyg_list)}/{len(df)}")

    # ── 3. Load model & predict ──
    print(f"\n--- Loading SchNet 300k ---")
    model = SchNetWrapper(**PARAMS_300K, use_charges=True).to(device)
    model.load_state_dict(
        torch.load(str(MODEL_300K), weights_only=False, map_location=device)
    )
    model.eval()

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

    # ── 4. Build result table ──
    df_result = df.loc[valid_idx].reset_index(drop=True)
    df_result["homo_pred"] = preds[:, 0]
    df_result["lumo_pred"] = preds[:, 1]
    df_result["gap_pred"] = preds[:, 2]
    df_result["homo_err"] = df_result["homo_pred"] - df_result["homo_exp"]
    df_result["lumo_err"] = df_result["lumo_pred"] - df_result["lumo_exp"]
    df_result["gap_err"] = df_result["gap_pred"] - df_result["gap_exp"]

    # ── 5. Print full table ──
    print(f"\n{'='*100}")
    print(f"  Combined Experimental Validation: SchNet 300k ({len(df_result)} molecules)")
    print(f"{'='*100}")
    print(f"  {'#':>3s} {'Name':<16s} {'Source':<16s} "
          f"{'H_exp':>6s} {'H_pred':>6s} {'err':>6s}  "
          f"{'L_exp':>6s} {'L_pred':>6s} {'err':>6s}  "
          f"{'G_exp':>5s} {'G_pred':>6s} {'err':>6s}")
    print(f"  {'-'*95}")

    for idx, row in df_result.iterrows():
        print(f"  {idx+1:3d} {row['name']:<16s} {row['source']:<16s} "
              f"{row['homo_exp']:6.2f} {row['homo_pred']:6.2f} {row['homo_err']:+6.2f}  "
              f"{row['lumo_exp']:6.2f} {row['lumo_pred']:6.2f} {row['lumo_err']:+6.2f}  "
              f"{row['gap_exp']:5.2f} {row['gap_pred']:6.2f} {row['gap_err']:+6.2f}")

    # ── 6. Per-source summary ──
    print(f"\n{'='*80}")
    print(f"  Per-Source Summary")
    print(f"{'='*80}")
    for source in df_result["source"].unique():
        sub = df_result[df_result["source"] == source]
        print(f"\n  [{source}] ({len(sub)} molecules)")
        for prop in ["homo", "lumo", "gap"]:
            errs = sub[f"{prop}_err"].values
            mae = np.mean(np.abs(errs))
            me = np.mean(errs)
            rmse = np.sqrt(np.mean(errs**2))
            exp_vals = sub[f"{prop}_exp"].values
            pred_vals = sub[f"{prop}_pred"].values
            r2 = r2_score(exp_vals, pred_vals) if len(sub) > 1 else float("nan")
            print(f"    {prop.upper():5s}: MAE={mae:.3f}  ME={me:+.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

    # ── 7. Overall summary ──
    print(f"\n{'='*80}")
    print(f"  Overall ({len(df_result)} molecules)")
    print(f"{'='*80}")
    for prop in ["homo", "lumo", "gap"]:
        errs = df_result[f"{prop}_err"].values
        mae = np.mean(np.abs(errs))
        me = np.mean(errs)
        rmse = np.sqrt(np.mean(errs**2))
        r2 = r2_score(df_result[f"{prop}_exp"], df_result[f"{prop}_pred"])
        print(f"  {prop.upper():5s}: MAE={mae:.3f}  ME={me:+.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

    # ── 8. Bias correction ──
    print(f"\n--- After Linear Bias Correction (overall) ---")
    for prop in ["homo", "lumo", "gap"]:
        bias = df_result[f"{prop}_err"].mean()
        corrected = df_result[f"{prop}_pred"] - bias
        mae = mean_absolute_error(df_result[f"{prop}_exp"], corrected)
        r2 = r2_score(df_result[f"{prop}_exp"], corrected)
        print(f"  {prop.upper():5s}: bias={bias:+.3f}  corrected MAE={mae:.3f}  R2={r2:.3f}")

    # ── 9. Save ──
    df_result.to_csv(OUT_DIR / "all_experimental_comparison.csv", index=False)

    summary = {
        "total_molecules": len(df_result),
        "sources": {},
        "overall": {},
        "bias_corrected": {},
    }
    for source in df_result["source"].unique():
        sub = df_result[df_result["source"] == source]
        summary["sources"][source] = {"n": len(sub)}
        for prop in ["homo", "lumo", "gap"]:
            errs = sub[f"{prop}_err"].values
            summary["sources"][source][prop] = {
                "mae": float(np.mean(np.abs(errs))),
                "me": float(np.mean(errs)),
            }
    for prop in ["homo", "lumo", "gap"]:
        errs = df_result[f"{prop}_err"].values
        bias = float(np.mean(errs))
        corrected = df_result[f"{prop}_pred"] - bias
        summary["overall"][prop] = {
            "mae": float(np.mean(np.abs(errs))),
            "me": bias,
            "rmse": float(np.sqrt(np.mean(errs**2))),
            "r2": float(r2_score(df_result[f"{prop}_exp"], df_result[f"{prop}_pred"])),
        }
        summary["bias_corrected"][prop] = {
            "mae": float(mean_absolute_error(df_result[f"{prop}_exp"], corrected)),
            "r2": float(r2_score(df_result[f"{prop}_exp"], corrected)),
        }

    with open(OUT_DIR / "combined_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
