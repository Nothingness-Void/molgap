"""
Phase 7: OOD validation — fetch 300 unseen molecules (MW 200-1000, CHONSFCl)
from PubChemQC, predict with P6 (44.8k) and P7 SchNet 300k, compare.

Usage:
  .venv\Scripts\python.exe scripts/phase7/ood_validation_300k.py
"""
from __future__ import annotations

import io
import json
import re
import time
import urllib.request
import urllib.error
import warnings

import ijson
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

warnings.filterwarnings("ignore")

from molgap.constants import (
    MODELS_DIR, RAW_DIR, RESULTS_DIR, TARGET_COLS, SEED,
    MODEL_PHASE6, PARAMS_PHASE6, GRAPHS_PHASE6,
)
from molgap.utils import ensure_dirs, regression_metrics, save_json
from molgap.graphs import smiles_to_pyg
from molgap.inference import load_model as _load_model, predict_graphs
from molgap.schnet import SchNetWrapper

OUT_DIR = RESULTS_DIR / "phase7" / "ood_validation"
N_TARGET = 300

HF_BASE = (
    "https://huggingface.co/datasets/molssiai-hub/pubchemqc-b3lyp/"
    "resolve/main/data/b3lyp_pm6/train/{file}"
)
HF_API_TREE = (
    "https://huggingface.co/api/datasets/molssiai-hub/pubchemqc-b3lyp/"
    "tree/main/data/b3lyp_pm6/train"
)

USER_AGENT = "curl/8"
CHUNK_BYTES = 20_000_000
ELEMENTS = {"C", "H", "O", "N", "S", "F", "Cl"}
MW_MIN, MW_MAX = 200, 1000

PARAMS_300K = {
    "hidden_channels": 192,
    "num_filters": 192,
    "num_interactions": 6,
    "num_gaussians": 50,
    "cutoff": 6.0,
    "dropout": 0.0,
}
MODEL_300K = MODELS_DIR / "gnn_schnet_3d_300k.pt"
GRAPHS_300K = RESULTS_DIR / "phase7" / "pyg_3d_graphs_etkdg_300k.pt"


def list_files():
    req = urllib.request.Request(HF_API_TREE, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return sorted(d["path"].split("/")[-1] for d in data if d.get("type") == "file")


def fetch_chunk(filename, start=0, size=CHUNK_BYTES):
    url = HF_BASE.format(file=filename)
    headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{start + size - 1}"}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception:
            time.sleep(2 ** attempt)
    return None


def formula_elements(formula):
    return set(re.findall(r'[A-Z][a-z]?', formula))


def parse_records(buf):
    records = []
    try:
        for obj in ijson.items(io.BytesIO(buf), "item"):
            cid = obj.get("cid")
            mw_raw = obj.get("pubchem-molecular-weight")
            formula = obj.get("pubchem-molecular-formula")
            smiles = obj.get("pubchem-isomeric-smiles")
            homo = obj.get("energy-alpha-homo")
            lumo = obj.get("energy-alpha-lumo")
            gap = obj.get("energy-alpha-gap")

            if not all(v is not None for v in [cid, mw_raw, formula, smiles, homo, lumo, gap]):
                continue

            mw = float(mw_raw)
            if not (MW_MIN <= mw <= MW_MAX):
                continue

            els = formula_elements(formula)
            if not els or not els.issubset(ELEMENTS):
                continue

            records.append({
                "cid": int(cid), "mw": mw, "formula": formula,
                "smiles": str(smiles),
                "homo": float(homo), "lumo": float(lumo), "gap": float(gap),
            })
    except (ijson.JSONError, Exception):
        pass
    return records


def load_training_cids():
    cids = set()
    for p in RAW_DIR.glob("phase*.csv"):
        df = pd.read_csv(p, usecols=["cid"])
        cids.update(df["cid"].tolist())
    print(f"  Training CIDs to exclude: {len(cids)}")
    return cids


def load_300k_model(device):
    """Load 300k model — trained on raw eV (no normalization)."""
    model = SchNetWrapper(**PARAMS_300K, use_charges=True).to(device)
    model.load_state_dict(
        torch.load(str(MODEL_300K), weights_only=False, map_location=device)
    )
    model.eval()
    return model


def predict_300k(model, pyg_list, device, batch_size=64):
    """Direct prediction — no denormalization needed."""
    loader = DataLoader(pyg_list, batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            charges = batch.charges if hasattr(batch, "charges") else None
            with torch.amp.autocast("cuda"):
                out = model(batch.z, batch.pos, batch.batch, charges=charges)
            preds.append(out.cpu().numpy())
    return np.vstack(preds)


def main():
    ensure_dirs(OUT_DIR)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Phase 7: OOD Validation ({N_TARGET} mol, MW 200-1000, CHONSFCl) ===\n")

    train_cids = load_training_cids()

    print(f"  Fetching file list from HuggingFace...")
    files = list_files()
    print(f"  Found {len(files)} files in b3lyp_pm6")

    np.random.shuffle(files)
    all_records = []
    for filename in files:
        if len(all_records) >= N_TARGET:
            break
        print(f"  Fetching {filename}...", end=" ", flush=True)
        buf = fetch_chunk(filename)
        if buf is None:
            print("FAILED")
            continue
        records = parse_records(buf)
        new = [r for r in records if r["cid"] not in train_cids]
        if len(new) > 10:
            idx = np.random.choice(len(new), 10, replace=False)
            new = [new[i] for i in idx]
        all_records.extend(new)
        print(f"kept {len(new)} (total: {len(all_records)})")

    if len(all_records) > N_TARGET:
        all_records = all_records[:N_TARGET]

    print(f"\n  Collected {len(all_records)} OOD molecules")
    ood_df = pd.DataFrame(all_records)
    ood_df.to_csv(OUT_DIR / "ood_molecules_300k.csv", index=False, encoding="utf-8")

    bins = [(200, 300), (300, 500), (500, 700), (700, 1000)]
    print(f"\n  MW distribution:")
    for lo, hi in bins:
        n = ((ood_df["mw"] >= lo) & (ood_df["mw"] < hi)).sum()
        print(f"    MW {lo}-{hi}: {n}")

    print(f"\n  Generating ETKDG conformers...")
    pyg_list, valid_idx = [], []
    for i, row in ood_df.iterrows():
        d = smiles_to_pyg(row["smiles"])
        if d is not None:
            pyg_list.append(d)
            valid_idx.append(i)
    valid_idx = np.array(valid_idx)
    print(f"  3D success: {len(pyg_list)}/{len(ood_df)}")

    # Load models
    print(f"\n  Loading P6 model (44.8k, normalized)...")
    p6_model, p6_mean, p6_std, _ = _load_model(
        MODEL_PHASE6, PARAMS_PHASE6, GRAPHS_PHASE6, device=device,
    )
    print(f"  Loading P7 SchNet 300k model (raw eV)...")
    p7_model = load_300k_model(device)

    # Predict
    print(f"  Predicting with P6...")
    p6_preds = predict_graphs(p6_model, pyg_list, p6_mean, p6_std, device)
    print(f"  Predicting with P7 300k...")
    p7_preds = predict_300k(p7_model, pyg_list, device)

    # Metrics
    ood_valid = ood_df.loc[valid_idx].reset_index(drop=True)
    y_true = ood_valid[TARGET_COLS].values
    m_p6 = regression_metrics(y_true, p6_preds)
    m_p7 = regression_metrics(y_true, p7_preds)

    for i, t in enumerate(TARGET_COLS):
        ood_valid[f"{t}_pred_p6"] = p6_preds[:, i]
        ood_valid[f"{t}_pred_p7"] = p7_preds[:, i]

    print(f"\n{'='*75}")
    print(f"  OOD Validation: P6 (44.8k) vs P7 SchNet 300k ({len(ood_valid)} molecules)")
    print(f"{'='*75}")
    print(f"  {'':5s}  {'--- Phase 6 ---':^25s}  {'--- 300k ---':^25s}")
    print(f"  {'':5s}  {'MAE':>7s} {'RMSE':>7s} {'R2':>7s}  {'MAE':>7s} {'RMSE':>7s} {'R2':>7s}")
    for t in TARGET_COLS:
        print(f"  {t:5s}  {m_p6[t]['mae']:7.4f} {m_p6[t]['rmse']:7.4f} {m_p6[t]['r2']:7.4f}  "
              f"{m_p7[t]['mae']:7.4f} {m_p7[t]['rmse']:7.4f} {m_p7[t]['r2']:7.4f}")
    print(f"  {'avg':5s}  {m_p6['average']['mae']:7.4f} {m_p6['average']['rmse']:7.4f} {m_p6['average']['r2']:7.4f}  "
          f"{m_p7['average']['mae']:7.4f} {m_p7['average']['rmse']:7.4f} {m_p7['average']['r2']:7.4f}")

    # Per MW bin
    print(f"\n  Per MW-bin MAE (avg of HOMO/LUMO/Gap):")
    print(f"  {'MW range':>10s}  {'n':>4s}  {'P6':>7s}  {'300k':>7s}  {'better':>7s}")
    for lo, hi in bins:
        mask = (ood_valid["mw"] >= lo) & (ood_valid["mw"] < hi)
        n = mask.sum()
        if n == 0:
            continue
        sub_true = ood_valid.loc[mask, TARGET_COLS].values
        sub_p6 = np.array([ood_valid.loc[mask, f"{t}_pred_p6"].values for t in TARGET_COLS]).T
        sub_p7 = np.array([ood_valid.loc[mask, f"{t}_pred_p7"].values for t in TARGET_COLS]).T
        mae_p6 = np.mean(np.abs(sub_true - sub_p6))
        mae_p7 = np.mean(np.abs(sub_true - sub_p7))
        better = "300k" if mae_p7 < mae_p6 else "P6"
        print(f"  {lo:>4d}-{hi:<4d}  {n:4d}  {mae_p6:7.4f}  {mae_p7:7.4f}  {better:>7s}")

    ood_valid.to_csv(OUT_DIR / "ood_comparison_300k.csv", index=False, encoding="utf-8")
    save_json({
        "n_target": N_TARGET,
        "n_predicted": len(ood_valid),
        "mw_range": [MW_MIN, MW_MAX],
        "elements": "CHONSFCl",
        "metrics_p6": m_p6,
        "metrics_p7_300k": m_p7,
    }, OUT_DIR / "ood_summary_300k.json")

    print(f"\n  Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
