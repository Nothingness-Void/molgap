"""
Phase 7: full model comparison — GPS 2D vs SchNet 3D vs Hybrid (Optuna-tuned)
on TWO datasets:
  1. OOD 1000  (PubChemQC unseen, B3LYP labels — pure model quality)
  2. Experimental (OLED literature + HOPV15 + extra — real measured values,
     so B3LYP has systematic bias; report raw + bias-corrected)

Uses the Optuna-tuned fusion head (models/hybrid_fusion_optuna.pt), whose
architecture (hidden) is read from fusion_optuna_metrics.json.

Usage:
  .venv\\Scripts\\python.exe scripts/phase7/compare_models_full.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

from molgap.constants import RESULTS_DIR, TARGET_COLS
from molgap.graphs import smiles_to_pyg, smiles_to_2d_pyg
from molgap.inference import load_hybrid

from validate_all_experimental import (
    parse_hopv, check_elements, EXTRA_MOLECULES, OLED_CSV, HOPV_PATH,
)

OUT_DIR = RESULTS_DIR / "phase7" / "full_comparison"
OOD_CSV = RESULTS_DIR / "phase7" / "ood_1000" / "ood_molecules_1000.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_all(smiles_list, gps, schnet, fusion):
    """Build 2D+3D graphs (keep 3D-valid), return (valid_idx, pred_2d, pred_3d, pred_hy)."""
    g2d_list, g3d_list, valid_idx = [], [], []
    for i, smi in enumerate(smiles_list):
        g3d = smiles_to_pyg(smi)
        if g3d is None:
            continue
        g2d = smiles_to_2d_pyg(smi)
        if g2d is None:
            continue
        g3d_list.append(g3d); g2d_list.append(g2d); valid_idx.append(i)

    pred_2d, emb_2d = [], []
    with torch.no_grad():
        for b in DataLoader(g2d_list, batch_size=256):
            b = b.to(device)
            e = gps.encode(b.x, b.edge_index, b.edge_attr, b.batch)
            emb_2d.append(e.cpu()); pred_2d.append(gps.head(e).cpu().numpy())
    pred_2d = np.concatenate(pred_2d); emb_2d = torch.cat(emb_2d)

    pred_3d, emb_3d = [], []
    with torch.no_grad():
        for b in DataLoader(g3d_list, batch_size=128):
            b = b.to(device)
            charges = b.charges if hasattr(b, "charges") else None
            e = schnet.encode(b.z, b.pos, b.batch, charges=charges)
            emb_3d.append(e.cpu()); pred_3d.append(schnet.head(e).cpu().numpy())
    pred_3d = np.concatenate(pred_3d); emb_3d = torch.cat(emb_3d)

    with torch.no_grad():
        pred_hy = fusion(emb_2d.to(device), emb_3d.to(device)).cpu().numpy()

    return np.array(valid_idx), pred_2d, pred_3d, pred_hy


def metrics_block(y_true, y_pred):
    out = {}
    for i, t in enumerate(TARGET_COLS):
        out[t] = {"mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
                  "r2": float(r2_score(y_true[:, i], y_pred[:, i]))}
    out["average"] = {"mae": float(np.mean([out[t]["mae"] for t in TARGET_COLS])),
                      "r2": float(np.mean([out[t]["r2"] for t in TARGET_COLS]))}
    return out


def print_table(title, y_true, preds, corrected=False):
    print(f"\n{'='*82}\n  {title}\n{'='*82}")
    print(f"  {'':5s}  {'GPS 2D':^16s}  {'SchNet 3D':^16s}  {'Hybrid(tuned)':^16s}")
    print(f"  {'':5s}  {'MAE':>7s} {'R2':>7s}  {'MAE':>7s} {'R2':>7s}  {'MAE':>7s} {'R2':>7s}")
    blocks = {}
    for name, p in preds.items():
        yp = p
        if corrected:  # subtract each model's own mean bias per target
            yp = p - (p - y_true).mean(axis=0, keepdims=True)
        blocks[name] = metrics_block(y_true, yp)
    for t in TARGET_COLS + ["average"]:
        row = f"  {t:5s}  "
        for name in ["2d", "3d", "hybrid"]:
            b = blocks[name][t]
            row += f"{b['mae']:7.4f} {b['r2']:7.4f}  "
        print(row)
    return blocks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print("Loading models...")
    gps, schnet, fusion, _ = load_hybrid(device, key="phase7_hybrid")
    results = {}

    # ── Dataset 1: OOD 1000 (B3LYP labels) ──
    print("\n[1/2] OOD 1000 ...")
    ood = pd.read_csv(OOD_CSV)
    vi, p2, p3, ph = predict_all(ood["smiles"].tolist(), gps, schnet, fusion)
    yt = ood.iloc[vi][["homo", "lumo", "gap"]].values.astype(np.float32)
    print(f"  valid: {len(vi)}/{len(ood)}")
    b = print_table(f"OOD {len(vi)} molecules (B3LYP labels)", yt,
                    {"2d": p2, "3d": p3, "hybrid": ph})
    results["ood"] = {"n": len(vi), **{k: b[k] for k in ["2d", "3d", "hybrid"]}}

    # ── Dataset 2: Experimental (measured values) ──
    print("\n[2/2] Experimental (OLED + HOPV15 + extra) ...")
    recs = []
    oled = pd.read_csv(OLED_CSV)
    for _, r in oled.iterrows():
        recs.append({"name": r["name"], "smiles": r["smiles"],
                     "homo": r["homo_exp"], "lumo": r["lumo_exp"], "gap": r["gap_exp"],
                     "source": "OLED"})
    for r in parse_hopv(HOPV_PATH):
        recs.append({"name": r["name"], "smiles": r["smiles"],
                     "homo": r["homo_exp"], "lumo": r["lumo_exp"], "gap": r["gap_exp"],
                     "source": "HOPV15"})
    for m in EXTRA_MOLECULES:
        mol = Chem.MolFromSmiles(m["smiles"])
        if mol and check_elements(mol):
            recs.append({"name": m["name"], "smiles": m["smiles"],
                         "homo": m["homo_exp"], "lumo": m["lumo_exp"], "gap": m["gap_exp"],
                         "source": "Extra"})
    exp = pd.DataFrame(recs)
    vi, p2, p3, ph = predict_all(exp["smiles"].tolist(), gps, schnet, fusion)
    yt = exp.iloc[vi][["homo", "lumo", "gap"]].values.astype(np.float32)
    print(f"  valid: {len(vi)}/{len(exp)}")
    b_raw = print_table(f"Experimental {len(vi)} molecules — RAW B3LYP vs measured",
                        yt, {"2d": p2, "3d": p3, "hybrid": ph})
    b_cor = print_table(f"Experimental {len(vi)} molecules — bias-corrected",
                        yt, {"2d": p2, "3d": p3, "hybrid": ph}, corrected=True)
    results["experimental_raw"] = {"n": len(vi), **{k: b_raw[k] for k in ["2d", "3d", "hybrid"]}}
    results["experimental_corrected"] = {"n": len(vi), **{k: b_cor[k] for k in ["2d", "3d", "hybrid"]}}

    # Save per-molecule experimental table
    exp_v = exp.iloc[vi].reset_index(drop=True)
    for i, t in enumerate(TARGET_COLS):
        exp_v[f"{t}_2d"] = p2[:, i]; exp_v[f"{t}_3d"] = p3[:, i]; exp_v[f"{t}_hybrid"] = ph[:, i]
    exp_v.to_csv(OUT_DIR / "experimental_3models.csv", index=False, encoding="utf-8")
    with open(OUT_DIR / "full_comparison_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
