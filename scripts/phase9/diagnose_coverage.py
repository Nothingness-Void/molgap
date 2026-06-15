"""
Phase 9 diagnostic: is the high PCQM4Mv2 error (gap MAE 0.26) caused by molecules
whose chemistry our 30万 training set never covered?

Same PCQM4Mv2 in-dist non-overlap subset as benchmark_pcqm4mv2.py. For each molecule
we compute its nearest-neighbor Tanimoto similarity (Morgan) to the training set,
predict its gap with the Hybrid, then bin by similarity and report gap MAE per bin.

  · Low-similarity bins have HIGH MAE, high-similarity bins LOW MAE
        → error IS driven by unseen chemistry (training coverage gap) → adding
          more diverse training data should help.
  · MAE roughly flat across bins
        → not a coverage problem (geometry / method issue instead).

Prereq: PCQM4Mv2 already downloaded by benchmark_pcqm4mv2.py.

Usage:
  .venv\\Scripts\\python.exe scripts/phase9/diagnose_coverage.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from sklearn.metrics import mean_absolute_error

from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.inference import load_hybrid, predict_smiles_batch_hybrid
from molgap.utils import canonicalize_smiles

DATA_CSV = RAW_DIR / "pcqm4m-v2" / "raw" / "data.csv.gz"
SPLIT_PT = RAW_DIR / "pcqm4m-v2" / "split_dict.pt"
TRAIN_CSV = RAW_DIR / "phase7_chonsfcl_mw200_1000_300k.csv"
ALLOWED = {"C", "H", "O", "N", "S", "F", "Cl"}
MW_MIN, MW_MAX = 200.0, 1000.0
N_SAMPLE = 3000
SEED = 42
OUT = RESULTS_DIR / "phase9" / "coverage_diagnostic.json"


def fp_of(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def in_dist(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    els = {a.GetSymbol() for a in mol.GetAtoms()}
    if any(a.GetTotalNumHs() for a in mol.GetAtoms()):
        els.add("H")
    return not (els - ALLOWED) and MW_MIN <= Descriptors.MolWt(mol) <= MW_MAX


def main():
    if not DATA_CSV.exists():
        raise SystemExit("PCQM4Mv2 not found — run benchmark_pcqm4mv2.py first to download.")

    df = pd.read_csv(DATA_CSV)
    split = torch.load(SPLIT_PT, weights_only=False)
    valid = df.iloc[np.array(split["valid"])].reset_index(drop=True)
    train = pd.read_csv(TRAIN_CSV)
    train_canon = set(train["canonical_smiles"].dropna())

    # Same subset as the benchmark (in-dist, non-overlap, same sample).
    rows = []
    for _, r in valid.iterrows():
        smi = r["smiles"]
        can = canonicalize_smiles(smi)
        if can is not None and can in train_canon:
            continue
        if not in_dist(smi):
            continue
        rows.append({"smiles": smi, "gap": float(r["homolumogap"])})
    sub = pd.DataFrame(rows)
    if len(sub) > N_SAMPLE:
        sub = sub.sample(N_SAMPLE, random_state=np.random.RandomState(SEED)).reset_index(drop=True)
    print(f"Evaluating {len(sub)} molecules (same as benchmark)")

    # Training-set fingerprints (full 300k for accurate nearest neighbor).
    print("Computing training-set fingerprints (300k)...")
    train_fps = [fp for fp in (fp_of(s) for s in train["canonical_smiles"].dropna()) if fp is not None]
    print(f"  {len(train_fps)} training fingerprints")

    # Predict gap with Hybrid.
    print("Predicting gap with Hybrid...")
    models = load_hybrid()
    vi, preds = predict_smiles_batch_hybrid(sub["smiles"].tolist(), models=models)
    sv = sub.iloc[vi].reset_index(drop=True)
    sv["gap_pred"] = preds[:, 2]
    sv["err"] = np.abs(sv["gap_pred"] - sv["gap"])

    # Nearest-neighbor Tanimoto similarity to the training set.
    print("Computing nearest-neighbor similarity to training set...")
    max_sims = []
    for smi in sv["smiles"]:
        fp = fp_of(smi)
        if fp is None:
            max_sims.append(np.nan)
            continue
        max_sims.append(float(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))))
    sv["max_sim"] = max_sims

    # Bin by similarity, report gap MAE per bin.
    bins = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)]
    print(f"\n{'='*52}")
    print(f"  similarity bin |   n  | gap MAE")
    print(f"{'='*52}")
    layers = []
    for lo, hi in bins:
        m = (sv["max_sim"] >= lo) & (sv["max_sim"] < hi)
        n = int(m.sum())
        mae = float(sv.loc[m, "err"].mean()) if n else None
        layers.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": n, "gap_mae": mae})
        if n:
            print(f"  [{lo:.1f}, {hi:.1f})     | {n:4d} | {mae:.3f}")

    overall = float(sv["err"].mean())
    print(f"{'='*52}")
    print(f"  overall MAE {overall:.3f} | mean max_sim {sv['max_sim'].mean():.3f}")
    print("\n  低相似度 MAE 高、高相似度 MAE 低 → 覆盖不足(未见化学结构)是主因")
    print("  各层 MAE 接近 → 不是覆盖问题(几何/方法)")

    OUT.write_text(json.dumps({"overall_mae": overall,
                               "mean_max_sim": float(sv["max_sim"].mean()),
                               "layers": layers}, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
