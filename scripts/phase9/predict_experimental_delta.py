"""
Phase 9: apply the FULL pipeline (B3LYP surrogate + Δ correction → near-GW) to the
experimental molecules and compare against measured values.

Earlier sounding compared raw B3LYP vs experiment (off by ~1-3 eV, the B3LYP
ceiling). Now we add the trained Δ model and ask: does B3LYP + Δ (≈ gas-phase GW)
land closer to experiment? GW ≈ gas-phase photoemission, so for gas-phase-like
measurements it should improve; residual offset (ME) hints at solid-state/solvent
effects the gas-phase target cannot capture.

For each molecule: B3LYP = hybrid prediction; Δ = LightGBM on the 192+192 embedding;
near-GW = B3LYP + Δ. Molecules outside the in-dist screen (elements/MW) get an
OOD flag — their Δ is extrapolated and less trustworthy.

Usage:
  .venv\\Scripts\\python.exe scripts/phase9/predict_experimental_delta.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import Descriptors

from molgap.constants import RESULTS_DIR, TARGET_COLS
from molgap.inference import load_hybrid, predict_smiles_batch_hybrid

# Experimental-data loaders live in scripts/phase7.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase7"))
from validate_all_experimental import (  # noqa: E402
    parse_hopv, EXTRA_MOLECULES, OLED_CSV, HOPV_PATH, ALLOWED_ELEMENTS,
)

PHASE9 = RESULTS_DIR / "phase9"
TARGETS = ("homo", "lumo", "gap")
MW_MIN, MW_MAX = 200.0, 1000.0


def load_delta_models():
    """Load LightGBM Δ models via model_str (model_file fails on the 文档 path)."""
    out = {}
    for t in TARGETS:
        out[t] = lgb.Booster(model_str=(PHASE9 / f"delta_lgbm_{t}.txt").read_text())
    return out


def in_distribution(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    els = {a.GetSymbol() for a in mol.GetAtoms()} | (
        {"H"} if any(a.GetTotalNumHs() for a in mol.GetAtoms()) else set())
    return not (els - ALLOWED_ELEMENTS) and MW_MIN <= Descriptors.MolWt(mol) <= MW_MAX


def collect_experimental():
    recs = []
    oled = pd.read_csv(OLED_CSV)
    for _, r in oled.iterrows():
        recs.append({"name": r["name"], "smiles": r["smiles"], "source": "OLED",
                     "homo": r["homo_exp"], "lumo": r["lumo_exp"], "gap": r["gap_exp"]})
    for r in parse_hopv(HOPV_PATH):
        recs.append({"name": r["name"], "smiles": r["smiles"], "source": "HOPV15",
                     "homo": r["homo_exp"], "lumo": r["lumo_exp"], "gap": r["gap_exp"]})
    for m in EXTRA_MOLECULES:
        recs.append({"name": m["name"], "smiles": m["smiles"], "source": "Extra",
                     "homo": m["homo_exp"], "lumo": m["lumo_exp"], "gap": m["gap_exp"]})
    return pd.DataFrame(recs)


def stats(y_exp, y_pred):
    err = y_pred - y_exp
    return float(np.mean(np.abs(err))), float(np.mean(err))  # MAE, ME(signed bias)


def main():
    exp = collect_experimental()
    print(f"Experimental molecules: {len(exp)}")

    models = load_hybrid(key="phase7_hybrid")
    vi, preds, e2d, e3d = predict_smiles_batch_hybrid(
        exp["smiles"].tolist(), models=models, return_embeddings=True)
    ev = exp.iloc[vi].reset_index(drop=True)
    print(f"Predicted (3D valid): {len(ev)}/{len(exp)}")

    X = np.hstack([e2d, e3d]).astype(np.float32)
    dmodels = load_delta_models()
    ev["in_dist"] = [in_distribution(s) for s in ev["smiles"]]
    for k, t in enumerate(TARGETS):
        ev[f"b3lyp_{t}"] = preds[:, k]
        ev[f"delta_{t}"] = dmodels[t].predict(X)
        ev[f"neargw_{t}"] = ev[f"b3lyp_{t}"] + ev[f"delta_{t}"]

    print(f"In-distribution (Δ trustworthy): {int(ev['in_dist'].sum())}/{len(ev)}\n")

    results = {}
    for subset, mask in (("ALL", np.ones(len(ev), bool)), ("IN-DIST", ev["in_dist"].to_numpy())):
        sub = ev[mask]
        if len(sub) == 0:
            continue
        print(f"{'='*64}\n  {subset}  (n={len(sub)})   pred vs experiment\n{'='*64}")
        print(f"  {'':5s} {'rawB3LYP MAE':>12s} {'ME':>7s}   {'nearGW MAE':>11s} {'ME':>7s}")
        results[subset] = {"n": int(len(sub))}
        for t in TARGETS:
            ye = sub[t].to_numpy(dtype=float)
            mae_b, me_b = stats(ye, sub[f"b3lyp_{t}"].to_numpy())
            mae_g, me_g = stats(ye, sub[f"neargw_{t}"].to_numpy())
            results[subset][t] = {"raw_mae": mae_b, "raw_me": me_b,
                                  "neargw_mae": mae_g, "neargw_me": me_g}
            print(f"  {t:5s} {mae_b:12.3f} {me_b:+7.3f}   {mae_g:11.3f} {me_g:+7.3f}")

    print("\n  MAE down from raw→nearGW = Δ moved predictions toward experiment.")
    print("  Residual ME (nearGW) = leftover bias, likely gas-phase GW vs solid/solution.")

    ev.to_csv(PHASE9 / "experimental_delta_comparison.csv", index=False, encoding="utf-8")
    (PHASE9 / "experimental_delta_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved experimental_delta_comparison.csv + _metrics.json to {PHASE9}")


if __name__ == "__main__":
    main()
