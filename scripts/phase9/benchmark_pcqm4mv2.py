"""
Phase 9: evaluate our Hybrid model on the PCQM4Mv2 VALIDATION set — the same
evaluation set the OGB-LSC leaderboard uses — so the gap MAE is directly comparable
to the leaderboard (~0.07 eV), instead of our own held-out.

PCQM4Mv2 predicts the B3LYP HOMO-LUMO gap (same target / same source as our data).
We:
  1. download PCQM4Mv2 (official OGB), take the validation split,
  2. drop molecules that overlap our training set (canonical SMILES) — no leakage,
  3. keep only in-distribution molecules (elements ⊆ CHONSFCl, MW 200-1000),
     since our model is only trained/trustworthy there,
  4. predict gap with the Hybrid and report MAE.

Honest caveats printed at the end: this is the in-dist, non-overlap subset of the
official valid set (not the full valid the leaderboard averages over), and PCQM4Mv2
gaps use DFT geometry while we use ETKDG.

Usage:
  .venv\\Scripts\\python.exe scripts/phase9/benchmark_pcqm4mv2.py
"""
from __future__ import annotations

import json
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_absolute_error

from molgap.constants import RAW_DIR, RESULTS_DIR
from molgap.inference import load_hybrid, predict_smiles_batch_hybrid
from molgap.utils import canonicalize_smiles

URL = "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip"
ZIP = RAW_DIR / "pcqm4m-v2.zip"
BASE = RAW_DIR / "pcqm4m-v2"
DATA_CSV = BASE / "raw" / "data.csv.gz"
SPLIT_PT = BASE / "split_dict.pt"

TRAIN_CSV = RAW_DIR / "phase7_chonsfcl_mw200_1000_300k.csv"
ALLOWED = {"C", "H", "O", "N", "S", "F", "Cl"}
MW_MIN, MW_MAX = 200.0, 1000.0
N_SAMPLE = 3000
SEED = 42
OUT = RESULTS_DIR / "phase9" / "pcqm4mv2_benchmark.json"


def ensure_data():
    if DATA_CSV.exists() and SPLIT_PT.exists():
        return
    if not ZIP.exists():
        print(f"Downloading PCQM4Mv2 (~60 MB) from {URL} ...")
        urllib.request.urlretrieve(URL, ZIP)
    print("Extracting ...")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(RAW_DIR)


def in_dist(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    els = {a.GetSymbol() for a in mol.GetAtoms()}
    if any(a.GetTotalNumHs() for a in mol.GetAtoms()):
        els.add("H")
    return not (els - ALLOWED) and MW_MIN <= Descriptors.MolWt(mol) <= MW_MAX


def main():
    ensure_data()
    df = pd.read_csv(DATA_CSV)  # columns: idx, smiles, homolumogap
    split = torch.load(SPLIT_PT, weights_only=False)
    valid_idx = np.array(split["valid"])
    valid = df.iloc[valid_idx].reset_index(drop=True)
    print(f"PCQM4Mv2 validation set: {len(valid)} molecules")

    train = pd.read_csv(TRAIN_CSV)
    train_canon = set(train["canonical_smiles"].dropna())
    print(f"Our training set: {len(train_canon)} canonical SMILES")

    rows, n_overlap, n_ood = [], 0, 0
    for _, r in valid.iterrows():
        smi = r["smiles"]
        can = canonicalize_smiles(smi)
        if can is not None and can in train_canon:
            n_overlap += 1
            continue
        if not in_dist(smi):
            n_ood += 1
            continue
        rows.append({"smiles": smi, "gap": float(r["homolumogap"])})
    sub = pd.DataFrame(rows)
    print(f"  dropped {n_overlap} training overlaps, {n_ood} out-of-distribution")
    print(f"  usable (in-dist, non-overlap): {len(sub)}")

    if len(sub) > N_SAMPLE:
        sub = sub.sample(N_SAMPLE, random_state=np.random.RandomState(SEED)).reset_index(drop=True)
        print(f"  sampled {len(sub)} for evaluation")

    print("\nPredicting gap with Hybrid (ETKDG + 2D/3D + fusion) ...")
    models = load_hybrid(key="phase7_hybrid")
    vi, preds = predict_smiles_batch_hybrid(sub["smiles"].tolist(), models=models)
    sv = sub.iloc[vi].reset_index(drop=True)
    gap_pred = preds[:, 2]  # TARGET_COLS = [homo, lumo, gap]
    gap_true = sv["gap"].to_numpy()
    mae = float(mean_absolute_error(gap_true, gap_pred))

    print(f"\n{'='*60}")
    print(f"  Our Hybrid on PCQM4Mv2 validation")
    print(f"  (in-dist, non-overlap subset, n={len(sv)})")
    print(f"{'='*60}")
    print(f"  gap MAE = {mae:.4f} eV")
    print(f"  leaderboard SOTA ~0.07 eV (full valid, 370万 trained)")
    print(f"\n  Caveats: in-dist non-overlap subset (not full valid);")
    print(f"           PCQM4Mv2 uses DFT geometry, we use ETKDG.")

    OUT.write_text(json.dumps({
        "n_valid_total": int(len(valid)), "n_overlap": n_overlap, "n_ood": n_ood,
        "n_evaluated": int(len(sv)), "gap_mae": mae, "leaderboard_sota": 0.07,
    }, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
