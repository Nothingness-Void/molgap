"""
Phase 9 (P9.4, variant A): train a Δ model — LightGBM learning the residual
Δ = GW − model-B3LYP — on the frozen 192+192-d hybrid embeddings.

The decisive test: can a structure-aware model beat the constant-bias baseline?
We report, on a SCAFFOLD-split held-out test set, the GW-accuracy MAE of:
  - raw            (no correction: just B3LYP prediction)
  - const          (constant correction: B3LYP + mean Δ from train) — the bar to beat
  - Δ-model        (B3LYP + LightGBM-predicted Δ)
  - Y-randomized   (shuffle Δ labels, retrain) — must collapse back to ~const,
                   else the model is fitting noise.

Inputs (from compute_delta.py):
  results/phase9/delta_oe62.csv             gw_*, pred_*, delta_*, smiles
  results/phase9/delta_oe62_embeddings.npz  emb_2d, emb_3d, smiles
Outputs:
  results/phase9/delta_model_metrics.json
  results/phase9/delta_lgbm_{homo,lumo,gap}.txt  (LightGBM boosters)

Usage:
  .venv\\Scripts\\python.exe scripts/phase9/train_delta.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from molgap.constants import RESULTS_DIR
from molgap.utils import murcko_scaffold_smiles

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PHASE9 = RESULTS_DIR / "phase9"
CSV = PHASE9 / "delta_oe62.csv"
NPZ = PHASE9 / "delta_oe62_embeddings.npz"
TARGETS = ("homo", "lumo", "gap")
SEED = 42
TEST_FRAC = 0.2

LGB_PARAMS = dict(
    n_estimators=1500, learning_rate=0.02, num_leaves=31, max_depth=-1,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
    reg_lambda=2.0, min_child_samples=30, random_state=SEED, n_jobs=-1, verbose=-1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV)
    parser.add_argument("--npz", type=Path, default=NPZ)
    parser.add_argument("--out-dir", type=Path, default=PHASE9)
    parser.add_argument("--out-prefix", default="delta_model")
    parser.add_argument("--model-prefix", default="delta_lgbm")
    parser.add_argument("--predictions-out", type=Path, default=None)
    parser.add_argument(
        "--feature-mode",
        choices=["embedding", "embedding_desc", "embedding_desc_pred"],
        default="embedding",
    )
    return parser.parse_args()


def descriptor_features(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for smi in df["smiles"].astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([0.0] * 13)
            continue
        atoms = mol.GetAtoms()
        rows.append([
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.NumAromaticRings(mol),
            Lipinski.RingCount(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumHDonors(mol),
            Descriptors.FractionCSP3(mol),
            Crippen.MolLogP(mol),
            mol.GetNumHeavyAtoms(),
            sum(1 for a in atoms if a.GetSymbol() in {"N", "O", "S"}),
            sum(1 for a in atoms if a.GetSymbol() in {"F", "Cl"}),
            Chem.GetFormalCharge(mol),
        ])
    arr = np.asarray(rows, dtype=np.float32)
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd[sd < 1e-6] = 1.0
    return (np.nan_to_num(arr, nan=0.0) - mu) / sd


def build_features(df: pd.DataFrame, e2d: np.ndarray, e3d: np.ndarray, mode: str) -> np.ndarray:
    parts = [e2d, e3d]
    if mode in {"embedding_desc", "embedding_desc_pred"}:
        parts.append(descriptor_features(df))
    if mode == "embedding_desc_pred":
        pred = df[["pred_homo", "pred_lumo", "pred_gap"]].to_numpy(dtype=np.float32)
        mu = pred.mean(axis=0)
        sd = pred.std(axis=0)
        sd[sd < 1e-6] = 1.0
        parts.append((pred - mu) / sd)
    return np.hstack(parts).astype(np.float32)


def fit_lgbm(X_tr, y_tr):
    """LightGBM with an internal validation split for early stopping."""
    Xa, Xv, ya, yv = train_test_split(X_tr, y_tr, test_size=0.1, random_state=SEED)
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(Xa, ya, eval_set=[(Xv, yv)],
          callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    return m


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    npz = np.load(args.npz, allow_pickle=True)
    e2d, e3d = npz["emb_2d"], npz["emb_3d"]
    assert len(df) == len(e2d) == len(e3d), "row count mismatch csv vs npz"
    assert (npz["smiles"] == df["smiles"].to_numpy()).all(), "smiles order mismatch"

    X = build_features(df, e2d, e3d, args.feature_mode)
    smiles = df["smiles"].tolist()
    print(f"{len(df)} molecules, feature dim {X.shape[1]}")

    # ── Scaffold split (group by Murcko scaffold, no leakage) ──
    scaffolds = [murcko_scaffold_smiles(s) or "NONE" for s in smiles]
    n_scaf = len(set(scaffolds))
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    tr, te = next(gss.split(X, groups=scaffolds))
    print(f"{n_scaf} unique scaffolds -> train {len(tr)} / test {len(te)} "
          f"(scaffold-disjoint)\n")

    rng = np.random.RandomState(SEED)
    results = {"n": int(len(df)), "n_scaffolds": int(n_scaf),
               "feature_mode": args.feature_mode, "feature_dim": int(X.shape[1]),
               "n_train": int(len(tr)), "n_test": int(len(te))}
    pred_df = df.iloc[te].reset_index(drop=True).copy()

    print(f"  {'tgt':4s} {'raw':>7s} {'const':>7s} {'Δmodel':>7s} {'Yrand':>7s} "
          f"{'R²(Δ)':>7s}   verdict")
    print(f"  {'-'*60}")
    for t in TARGETS:
        y = df[f"delta_{t}"].to_numpy()
        pred_b3 = df[f"pred_{t}"].to_numpy()
        gw = df[f"gw_{t}"].to_numpy()
        gw_te = gw[te]

        # Δ model
        model = fit_lgbm(X[tr], y[tr])
        dpred = model.predict(X[te])
        mae_model = mean_absolute_error(gw_te, pred_b3[te] + dpred)
        r2_model = r2_score(gw_te, pred_b3[te] + dpred)

        # baselines
        mae_raw = mean_absolute_error(gw_te, pred_b3[te])
        mae_const = mean_absolute_error(gw_te, pred_b3[te] + y[tr].mean())

        # Y-randomization
        y_sh = y[tr].copy(); rng.shuffle(y_sh)
        m_yr = fit_lgbm(X[tr], y_sh)
        mae_yrand = mean_absolute_error(gw_te, pred_b3[te] + m_yr.predict(X[te]))

        beats_const = mae_model < mae_const
        signal_real = mae_yrand > mae_model * 1.05
        verdict = ("learns structure" if beats_const else "no gain over const")
        verdict += "; signal real" if signal_real else "; WEAK signal"

        results[t] = {
            "mae_raw": float(mae_raw), "mae_const": float(mae_const),
            "mae_delta_model": float(mae_model), "mae_yrand": float(mae_yrand),
            "r2_delta_model": float(r2_model),
            "beats_const": bool(beats_const), "signal_real": bool(signal_real),
        }
        print(f"  {t:4s} {mae_raw:7.3f} {mae_const:7.3f} {mae_model:7.3f} "
              f"{mae_yrand:7.3f} {r2_model:7.3f}   {verdict}")
        pred_df[f"gw_pred_raw_{t}"] = pred_b3[te]
        pred_df[f"gw_pred_const_{t}"] = pred_b3[te] + y[tr].mean()
        pred_df[f"gw_pred_lgbm_delta_{t}"] = pred_b3[te] + dpred

        # Write via Python (pathlib handles non-ASCII paths; LightGBM's C
        # save_model does not — it mangles the "文档" path and fails).
        (args.out_dir / f"{args.model_prefix}_{t}.txt").write_text(
            model.booster_.model_to_string(), encoding="utf-8")

    print("\n  raw=no correction · const=B3LYP+mean(Δ) · Δmodel=B3LYP+LightGBM(Δ)")
    print("  Want: Δmodel < const (learns structure) AND Yrand ≈ const (signal real)")

    metrics_out = args.out_dir / f"{args.out_prefix}_metrics.json"
    metrics_out.write_text(json.dumps(results, indent=2))
    if args.predictions_out is not None:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(args.predictions_out, index=False, encoding="utf-8")
    print(f"\nSaved {metrics_out} + {args.model_prefix}_*.txt to {args.out_dir}")
    if args.predictions_out is not None:
        print(f"Saved predictions: {args.predictions_out}")


if __name__ == "__main__":
    main()
