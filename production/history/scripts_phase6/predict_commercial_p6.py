"""Predict commercial OLED molecules with Phase 6 expanded SchNet model."""
import json

import numpy as np
import pandas as pd
import torch

from molgap.constants import (
    COMMERCIAL_DIR, RESULTS_DIR, MODELS_DIR,
    MODEL_PHASE6, GRAPHS_PHASE6, PARAMS_PHASE6, TARGET_COLS, SEED,
)
from molgap.graphs import smiles_to_pyg
from molgap.inference import load_model, predict_graphs
from molgap.utils import create_split_indices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(COMMERCIAL_DIR / "gaussian_validation_10.csv")
print(f"Loaded {len(df)} commercial molecules")

pyg_list, valid_idx = [], []
for i, row in df.iterrows():
    d = smiles_to_pyg(row["smiles"])
    if d is not None:
        pyg_list.append(d)
        valid_idx.append(i)
        print(f"  {row['name']}: {d.z.shape[0]} atoms")

print(f"3D success: {len(pyg_list)}/{len(df)}")

model, y_mean, y_std, device = load_model()
preds = predict_graphs(model, pyg_list, y_mean, y_std, device)

p4_path = RESULTS_DIR / "phase5" / "gaussian_validation" / "ml_vs_gaussian.csv"
p4 = pd.read_csv(p4_path)

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

out_path = RESULTS_DIR / "phase6" / "commercial_predictions_p6.csv"
result.to_csv(out_path, index=False)
print(f"\nSaved to {out_path.relative_to(RESULTS_DIR.parent)}")
