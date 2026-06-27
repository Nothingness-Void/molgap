"""
Phase 9 geometry-noise diagnostic: how much of the residual Δ error is caused by
the ETKDG-vs-PBE geometry mismatch?

Same Δ-model pipeline as train_delta.py, but the 3D graph (SchNet input) is built
from OE62's own PBE-relaxed geometry (xyz_pbe_relaxed) instead of ETKDG. The 2D
graph is unchanged (geometry-independent). If the Δ-model MAE drops a lot vs the
ETKDG baseline (0.197/0.217/0.303), geometry mismatch is a major noise source and
better inference geometry is worth pursuing; if it barely moves, the residual is
the GW/label floor and geometry is not the bottleneck.

NOTE: diagnostic only — PBE geometry is NOT available for commercial molecules at
inference (they only have ETKDG), so this measures the geometry-noise ceiling, it
is not a production path.

Usage:
  .venv\\Scripts\\python.exe scripts/phase9/diagnose_geometry.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from molgap.constants import RESULTS_DIR
from molgap.inference import load_hybrid
from molgap.graphs import smiles_to_2d_pyg
from molgap.utils import compute_gasteiger_charges, murcko_scaffold_smiles

from probe_oe62_indist import (
    gw_homo_lumo, molecule_elements, ALLOWED_ELEMENTS, MW_MIN, MW_MAX,
)
from train_delta import fit_lgbm, TARGETS, SEED, TEST_FRAC

OE62 = "data/raw/oe62_df_5k.json"
PHASE9 = RESULTS_DIR / "phase9"
ETKDG_BASELINE = {"homo": 0.197, "lumo": 0.217, "gap": 0.303}  # from delta_model_metrics


def xyz_to_3d_data(xyz_str):
    """Build a SchNet 3D Data(z, pos, charges) from a standard-XYZ geometry."""
    if not isinstance(xyz_str, str) or not xyz_str.strip():
        return None
    mol = Chem.MolFromXYZBlock(xyz_str)
    if mol is None:
        return None
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        return None
    n = mol.GetNumAtoms()
    if n == 0 or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    z = torch.tensor([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)],
                     dtype=torch.long)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    try:
        charges = compute_gasteiger_charges(mol)
    except Exception:
        return None
    return Data(z=z, pos=pos, charges=torch.tensor(charges, dtype=torch.float32))


def main():
    df = pd.read_json(OE62, orient="split")
    print(f"Loaded {len(df)} OE62 rows")

    # ── In-distribution candidates with GW + PBE geometry ──
    rows = []
    for _, row in df.iterrows():
        hl = gw_homo_lumo(row)
        if hl is None:
            continue
        smi = row.get("canonical_smiles")
        if not isinstance(smi, str) or not smi:
            continue
        els, mw = molecule_elements(smi)
        if els is None or (els - ALLOWED_ELEMENTS) or not (MW_MIN <= mw <= MW_MAX):
            continue
        rows.append({"smiles": smi, "xyz": row.get("xyz_pbe_relaxed"),
                     "gw_homo": hl[0], "gw_lumo": hl[1], "gw_gap": hl[1] - hl[0]})
    cand = pd.DataFrame(rows)
    print(f"In-distribution candidates: {len(cand)}")

    # ── Build 2D (unchanged) + 3D-from-PBE graphs, keep both-valid ──
    gps, schnet, fusion, device = load_hybrid(key="phase7_hybrid")
    g2d, g3d, keep = [], [], []
    for i, r in cand.iterrows():
        d3 = xyz_to_3d_data(r["xyz"])
        if d3 is None:
            continue
        d2 = smiles_to_2d_pyg(r["smiles"])
        if d2 is None:
            continue
        g3d.append(d3); g2d.append(d2); keep.append(i)
    cv = cand.iloc[keep].reset_index(drop=True)
    print(f"PBE-geometry valid (XYZ->bonds + 2D ok): {len(cv)}/{len(cand)}\n")

    # ── Encode (emb_2d from SMILES, emb_3d from PBE geometry) + fuse ──
    emb_2d = []
    with torch.no_grad():
        for b in DataLoader(g2d, batch_size=256):
            b = b.to(device)
            emb_2d.append(gps.encode(b.x, b.edge_index, b.edge_attr, b.batch).cpu())
    emb_2d = torch.cat(emb_2d)

    emb_3d = []
    with torch.no_grad():
        for b in DataLoader(g3d, batch_size=128):
            b = b.to(device)
            charges = b.charges if hasattr(b, "charges") else None
            emb_3d.append(schnet.encode(b.z, b.pos, b.batch, charges=charges).cpu())
    emb_3d = torch.cat(emb_3d)

    with torch.no_grad():
        preds = fusion(emb_2d.to(device), emb_3d.to(device)).cpu().numpy()

    X = torch.cat([emb_2d, emb_3d], dim=1).numpy().astype(np.float32)
    for k, t in enumerate(TARGETS):
        cv[f"pred_{t}"] = preds[:, k]
        cv[f"delta_{t}"] = cv[f"gw_{t}"] - cv[f"pred_{t}"]

    # ── Scaffold split + train Δ model on PBE-geometry features ──
    scaffolds = [murcko_scaffold_smiles(s) or "NONE" for s in cv["smiles"]]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    tr, te = next(gss.split(X, groups=scaffolds))
    print(f"train {len(tr)} / test {len(te)} (scaffold split)\n")

    print(f"  {'tgt':4s} {'ETKDG base':>10s} {'PBE-geom':>9s} {'const':>7s} {'Δ vs ETKDG':>11s}")
    print(f"  {'-'*52}")
    results = {"n": int(len(cv)), "n_train": int(len(tr)), "n_test": int(len(te))}
    for t in TARGETS:
        y = cv[f"delta_{t}"].to_numpy()
        pred_b3 = cv[f"pred_{t}"].to_numpy()
        gw_te = cv[f"gw_{t}"].to_numpy()[te]
        model = fit_lgbm(X[tr], y[tr])
        mae_pbe = mean_absolute_error(gw_te, pred_b3[te] + model.predict(X[te]))
        mae_const = mean_absolute_error(gw_te, pred_b3[te] + y[tr].mean())
        base = ETKDG_BASELINE[t]
        delta = mae_pbe - base
        results[t] = {"mae_pbe_geom": float(mae_pbe), "mae_const": float(mae_const),
                      "etkdg_baseline": base, "improvement": float(-delta)}
        print(f"  {t:4s} {base:10.3f} {mae_pbe:9.3f} {mae_const:7.3f} {delta:+11.3f}")

    print("\n  'Δ vs ETKDG' negative = PBE geometry reduced error = geometry WAS noise.")
    print("  Near zero = geometry is not the bottleneck (residual is GW/label floor).")
    print("  Caveat: different valid-molecule set than ETKDG run; read the trend, not 3rd decimal.")

    (PHASE9 / "geometry_diagnostic.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved geometry_diagnostic.json to {PHASE9}")


if __name__ == "__main__":
    main()
