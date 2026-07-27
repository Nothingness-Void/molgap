"""
Phase 7: Conformer ensemble prediction experiment.

Question: how much of the OOD error comes from ETKDG conformer randomness?
Method: for the 500-mol Phase 6 OOD set, generate K conformers per molecule,
predict each with the Phase 6 model, average over first k = 1/3/5/8 conformers,
and compare metrics. Also report per-molecule prediction std across conformers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from molgap.constants import (
    RESULTS_DIR, TARGET_COLS, MODEL_PHASE6, PARAMS_PHASE6, GRAPHS_PHASE6, SEED,
)
from molgap.utils import regression_metrics, save_json, ensure_dirs
from molgap.graphs import smiles_to_pyg_ensemble
from molgap.inference import load_model

K = 8
OUT_DIR = RESULTS_DIR / "phase7" / "conformer_ensemble"
OOD_CSV = RESULTS_DIR / "phase6" / "ood_validation" / "ood_molecules_p6.csv"


def main():
    ensure_dirs(OUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    ood = pd.read_csv(OOD_CSV)
    print(f"OOD molecules: {len(ood)}", flush=True)

    model, y_mean, y_std, device = load_model(
        MODEL_PHASE6, PARAMS_PHASE6, GRAPHS_PHASE6,
    )

    # generate conformers
    all_graphs, mol_ids = [], []
    n_confs = []
    print(f"Generating up to {K} conformers per molecule...", flush=True)
    for i, row in ood.iterrows():
        gs = smiles_to_pyg_ensemble(row["smiles"], k=K)
        all_graphs.extend(gs)
        mol_ids.extend([i] * len(gs))
        n_confs.append(len(gs))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ood)} molecules, {len(all_graphs)} conformers", flush=True)
    mol_ids = np.array(mol_ids)
    print(f"Total conformers: {len(all_graphs)}; molecules with 0 confs: {sum(c == 0 for c in n_confs)}", flush=True)

    # predict all conformers
    loader = DataLoader(all_graphs, batch_size=64)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.z, batch.pos, batch.batch, charges=batch.charges)
            preds.append(out.cpu().numpy() * y_std + y_mean)
    preds = np.vstack(preds)  # (n_conformers, 3)

    # aggregate per molecule
    valid_idx = sorted(set(mol_ids.tolist()))
    y_true = ood.loc[valid_idx, TARGET_COLS].values

    summary = {}
    for k_use in [1, 3, 5, K]:
        y_pred = np.stack([preds[mol_ids == i][:k_use].mean(axis=0) for i in valid_idx])
        m = regression_metrics(y_true, y_pred)
        summary[f"k={k_use}"] = m
        print(f"\n  k={k_use} conformers:", flush=True)
        for t in TARGET_COLS:
            print(f"    {t:5s}: MAE={m[t]['mae']:.4f}  RMSE={m[t]['rmse']:.4f}  R2={m[t]['r2']:.4f}", flush=True)
        print(f"    avg  : MAE={m['average']['mae']:.4f}  RMSE={m['average']['rmse']:.4f}  R2={m['average']['r2']:.4f}", flush=True)

    # per-molecule conformer spread
    stds = np.stack([preds[mol_ids == i].std(axis=0) for i in valid_idx])
    print(f"\n  Conformer prediction std (eV) across {K} conformers:", flush=True)
    for j, t in enumerate(TARGET_COLS):
        print(f"    {t:5s}: mean={stds[:, j].mean():.4f}  median={np.median(stds[:, j]):.4f}  max={stds[:, j].max():.4f}", flush=True)

    # save per-molecule results
    res = ood.loc[valid_idx].reset_index(drop=True).copy()
    mean_preds = np.stack([preds[mol_ids == i].mean(axis=0) for i in valid_idx])
    single_preds = np.stack([preds[mol_ids == i][0] for i in valid_idx])
    for j, t in enumerate(TARGET_COLS):
        res[f"{t}_pred_single"] = single_preds[:, j]
        res[f"{t}_pred_ens{K}"] = mean_preds[:, j]
        res[f"{t}_conf_std"] = stds[:, j]
    res.to_csv(OUT_DIR / "ensemble_comparison.csv", index=False, encoding="utf-8")

    save_json({
        "n_molecules": len(valid_idx),
        "k_max": K,
        "metrics_by_k": summary,
        "conf_std_mean": {t: float(stds[:, j].mean()) for j, t in enumerate(TARGET_COLS)},
    }, OUT_DIR / "ensemble_summary.json")
    print(f"\nSaved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
