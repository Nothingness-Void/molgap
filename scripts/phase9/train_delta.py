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

import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from molgap.constants import RESULTS_DIR
from molgap.utils import murcko_scaffold_smiles

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


def fit_lgbm(X_tr, y_tr):
    """LightGBM with an internal validation split for early stopping."""
    Xa, Xv, ya, yv = train_test_split(X_tr, y_tr, test_size=0.1, random_state=SEED)
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(Xa, ya, eval_set=[(Xv, yv)],
          callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    return m


def main():
    df = pd.read_csv(CSV)
    npz = np.load(NPZ, allow_pickle=True)
    e2d, e3d = npz["emb_2d"], npz["emb_3d"]
    assert len(df) == len(e2d) == len(e3d), "row count mismatch csv vs npz"
    assert (npz["smiles"] == df["smiles"].to_numpy()).all(), "smiles order mismatch"

    X = np.hstack([e2d, e3d]).astype(np.float32)  # [n, 384]
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
               "n_train": int(len(tr)), "n_test": int(len(te))}

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

        # Write via Python (pathlib handles non-ASCII paths; LightGBM's C
        # save_model does not — it mangles the "文档" path and fails).
        (PHASE9 / f"delta_lgbm_{t}.txt").write_text(
            model.booster_.model_to_string(), encoding="utf-8")

    print("\n  raw=no correction · const=B3LYP+mean(Δ) · Δmodel=B3LYP+LightGBM(Δ)")
    print("  Want: Δmodel < const (learns structure) AND Yrand ≈ const (signal real)")

    (PHASE9 / "delta_model_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved delta_model_metrics.json + delta_lgbm_*.txt to {PHASE9}")


if __name__ == "__main__":
    main()
