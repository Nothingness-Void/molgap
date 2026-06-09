"""
Phase 5: Out-of-distribution validation.

Fetch 100 random molecules from PubChemQC that are NOT in the training set,
predict with the stacking model, compare with PubChemQC ground truth.

Strategy:
  - Use a DIFFERENT HF subset (e.g. chnopsfclnakmgca500) or different file
    offsets to get molecules outside our training data
  - No MW/element filter — grab anything to test true generalization
  - Predict with SchNet + LGBM stacking
  - Compare ML predictions vs PubChemQC computed values
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
import urllib.error
import warnings
from pathlib import Path

import ijson
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import (
    MODELS_DIR,
    RAW_DIR,
    RESULTS_DIR,
    TARGET_COLS,
    ensure_dirs,
    regression_metrics,
    save_json,
    compute_gasteiger_charges,
)
from molgap.schnet import SchNetWrapper

OUT_DIR = RESULTS_DIR / "phase5" / "ood_validation"
SEED = 42
N_TARGET = 100

# Use a different subset to guarantee out-of-distribution
HF_BASE = (
    "https://huggingface.co/datasets/molssiai-hub/pubchemqc-b3lyp/"
    "resolve/main/data/b3lyp_pm6_chnopsfclnakmgca500/train/{file}"
)
HF_API_TREE = (
    "https://huggingface.co/api/datasets/molssiai-hub/pubchemqc-b3lyp/"
    "tree/main/data/b3lyp_pm6_chnopsfclnakmgca500/train"
)

USER_AGENT = "curl/8"
CHUNK_BYTES = 20_000_000  # 20 MB per file


def list_files():
    req = urllib.request.Request(HF_API_TREE, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return sorted(d["path"].split("/")[-1] for d in data if d.get("type") == "file")


def fetch_chunk(filename, start=0, size=CHUNK_BYTES):
    url = HF_BASE.format(file=filename)
    headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{start + size - 1}"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def parse_records(buf):
    records = []
    try:
        for obj in ijson.items(io.BytesIO(buf), "item"):
            cid = obj.get("cid")
            mw = obj.get("pubchem-molecular-weight")
            formula = obj.get("pubchem-molecular-formula")
            smiles = obj.get("pubchem-isomeric-smiles")
            homo = obj.get("energy-alpha-homo")
            lumo = obj.get("energy-alpha-lumo")
            gap = obj.get("energy-alpha-gap")

            if all(v is not None for v in [cid, smiles, homo, lumo, gap]):
                records.append({
                    "cid": int(cid),
                    "mw": float(mw) if mw else None,
                    "formula": formula,
                    "smiles": str(smiles),
                    "homo": float(homo),
                    "lumo": float(lumo),
                    "gap": float(gap),
                })
    except (ijson.JSONError, Exception):
        pass
    return records


def load_training_cids():
    """Load CIDs from training data to exclude."""
    csv_path = RAW_DIR / "phase3_chonsfcl_mw200_1000_30k.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return set(df["cid"].tolist())
    return set()


def main():
    ensure_dirs(OUT_DIR)
    np.random.seed(SEED)

    print(f"=== Phase 5: Out-of-Distribution Validation ===", flush=True)
    print(f"  Target: {N_TARGET} molecules from PubChemQC (not in training)", flush=True)

    # Load training CIDs to exclude
    train_cids = load_training_cids()
    print(f"  Training CIDs to exclude: {len(train_cids)}", flush=True)

    # List available files
    print(f"  Fetching file list from HuggingFace...", flush=True)
    files = list_files()
    print(f"  Found {len(files)} files in chnopsfclnakmgca500 subset", flush=True)

    # Sample random files, always read from offset=0 for valid JSON parsing
    np.random.shuffle(files)

    all_records = []
    files_tried = 0

    for filename in files:
        if len(all_records) >= N_TARGET:
            break

        files_tried += 1
        print(f"  Fetching {filename} (offset=0, {CHUNK_BYTES//1e6:.0f}MB)...", flush=True)
        try:
            buf = fetch_chunk(filename, start=0, size=CHUNK_BYTES)
        except Exception as e:
            print(f"    Failed: {e}", flush=True)
            continue

        records = parse_records(buf)
        # Filter: not in training set, and randomly sample a few per file for diversity
        new_records = [r for r in records if r["cid"] not in train_cids]
        if len(new_records) > 20:
            idx = np.random.choice(len(new_records), 20, replace=False)
            new_records = [new_records[i] for i in idx]
        all_records.extend(new_records)
        print(f"    Got {len(records)} records, kept {len(new_records)} "
              f"(total: {len(all_records)})", flush=True)

    # Take exactly N_TARGET
    if len(all_records) > N_TARGET:
        all_records = all_records[:N_TARGET]

    print(f"\n  Collected {len(all_records)} out-of-distribution molecules", flush=True)

    if not all_records:
        print("  ERROR: no molecules fetched, exiting", flush=True)
        return

    ood_df = pd.DataFrame(all_records)
    ood_df.to_csv(OUT_DIR / "ood_molecules.csv", index=False, encoding="utf-8")

    # ── Generate 3D conformers (PM6 preferred) and predict with SchNet ──
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from molgap.utils import create_split_indices

    print(f"\n  Generating 3D conformers (ETKDG only — matches training)...", flush=True)

    def smiles_to_pyg(smi, use_charges=True):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol_h = AllChem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) != 0:
            if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) != 0:
                return None
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        n = mol_h.GetNumAtoms()
        if n == 0:
            return None
        conf = mol_h.GetConformer()
        z = torch.tensor([mol_h.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)], dtype=torch.long)
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
        data = Data(z=z, pos=pos)
        if use_charges:
            charges = compute_gasteiger_charges(mol_h)
            data.charges = torch.tensor(charges, dtype=torch.float32)
        return data

    pyg_list = []
    valid_idx = []
    for i, row in ood_df.iterrows():
        d = smiles_to_pyg(row["smiles"])
        if d is not None:
            pyg_list.append(d)
            valid_idx.append(i)
    valid_idx = np.array(valid_idx)
    has_charges = len(pyg_list) > 0 and hasattr(pyg_list[0], 'charges')
    print(f"  3D success: {len(pyg_list)}/{len(ood_df)} (ETKDG)", flush=True)

    # Load SchNet tuned model params from Optuna results
    optuna_params_path = RESULTS_DIR / "phase4" / "schnet_optuna" / "optuna_best_params.json"
    if optuna_params_path.exists():
        with open(optuna_params_path) as f:
            optuna_params = json.load(f)
        SCHNET_MODEL_PARAMS = {
            "hidden_channels": optuna_params["hidden_channels"],
            "num_filters": optuna_params["num_filters"],
            "num_interactions": optuna_params["num_interactions"],
            "num_gaussians": optuna_params["num_gaussians"],
            "cutoff": optuna_params["cutoff"],
            "dropout": optuna_params["dropout"],
        }
        print(f"  Loaded Optuna best params from {optuna_params_path}", flush=True)
    else:
        SCHNET_MODEL_PARAMS = {
            "hidden_channels": 192,
            "num_filters": 256,
            "num_interactions": 6,
            "num_gaussians": 100,
            "cutoff": 6.0,
            "dropout": 0.2,
        }
        print(f"  WARNING: Optuna params not found, using defaults", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    # Get y_mean/y_std from ETKDG training graphs (must match training)
    GRAPHS_PATH_ETKDG = RESULTS_DIR / "phase4" / "pyg_3d_graphs_etkdg.pt"
    GRAPHS_PATH_LEGACY = RESULTS_DIR / "phase4" / "pyg_3d_graphs.pt"
    graphs_path = GRAPHS_PATH_ETKDG if GRAPHS_PATH_ETKDG.exists() else GRAPHS_PATH_LEGACY
    data_list = torch.load(graphs_path, weights_only=False)
    train_idx_g, _, _ = create_split_indices(len(data_list), random_state=SEED)
    train_y = np.stack([data_list[i].y.squeeze(0).numpy() for i in train_idx_g])
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std[y_std < 1e-6] = 1.0
    del data_list

    model = SchNetWrapper(**SCHNET_MODEL_PARAMS, use_charges=has_charges).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "gnn_schnet_3d_tuned.pt",
                                     weights_only=True, map_location=device))
    model.eval()

    print(f"  Predicting with SchNet (charges={has_charges})...", flush=True)
    loader = DataLoader(pyg_list, batch_size=64)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast("cuda"):
                charges = getattr(batch, 'charges', None)
                out = model(batch.z, batch.pos, batch.batch, charges=charges)
            preds.append(out.cpu().numpy() * y_std + y_mean)
    schnet_preds = np.vstack(preds)

    # Build comparison
    ood_valid = ood_df.loc[valid_idx].reset_index(drop=True)
    for i, t in enumerate(TARGET_COLS):
        ood_valid[f"{t}_pred"] = schnet_preds[:, i]
        ood_valid[f"{t}_err"] = ood_valid[f"{t}_pred"] - ood_valid[t]
        ood_valid[f"{t}_abs_err"] = np.abs(ood_valid[f"{t}_err"])

    # Metrics
    y_true = ood_valid[TARGET_COLS].values
    y_pred = schnet_preds
    m = regression_metrics(y_true, y_pred)

    print(f"\n{'='*65}", flush=True)
    print(f"  OOD Validation Results ({len(ood_valid)} molecules)", flush=True)
    print(f"{'='*65}", flush=True)
    for t in TARGET_COLS:
        print(f"  {t:5s}: MAE={m[t]['mae']:.4f}  RMSE={m[t]['rmse']:.4f}  R2={m[t]['r2']:.4f}", flush=True)
    print(f"  avg  : MAE={m['average']['mae']:.4f}  RMSE={m['average']['rmse']:.4f}  R2={m['average']['r2']:.4f}", flush=True)

    # Stats
    print(f"\n  Element coverage:", flush=True)
    if "formula" in ood_valid.columns:
        import re
        all_elements = set()
        for f in ood_valid["formula"].dropna():
            all_elements.update(re.findall(r'[A-Z][a-z]?', str(f)))
        print(f"    Elements seen: {sorted(all_elements)}", flush=True)

    if "mw" in ood_valid.columns:
        print(f"  MW range: {ood_valid['mw'].min():.1f} - {ood_valid['mw'].max():.1f}", flush=True)

    print(f"\n  Per-target error distribution (eV):", flush=True)
    for t in TARGET_COLS:
        ae = ood_valid[f"{t}_abs_err"]
        print(f"    {t}: mean={ae.mean():.4f} median={ae.median():.4f} "
              f"max={ae.max():.4f} <0.2eV={100*(ae<0.2).mean():.1f}%", flush=True)

    ood_valid.to_csv(OUT_DIR / "ood_comparison.csv", index=False, encoding="utf-8")

    save_json({
        "n_fetched": len(ood_df),
        "n_predicted": len(ood_valid),
        "n_3d_failed": len(ood_df) - len(ood_valid),
        "model": "SchNet_3D_optuna",
        "metrics": m,
        "mw_range": [float(ood_valid["mw"].min()), float(ood_valid["mw"].max())]
        if "mw" in ood_valid.columns else None,
    }, OUT_DIR / "ood_summary.json")

    print(f"\n  Saved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
