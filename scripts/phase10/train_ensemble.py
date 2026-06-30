"""
Phase 10 (M1, tier 1+1.5): UQ via a LightGBM Δ-model ensemble + calibration.

This is the uncertainty-quantification entry point for the property database. It
reuses *exactly* the Phase 9 Δ-learning setup (frozen 192+192-d hybrid embeddings,
OE62 in-distribution GW molecules, scaffold split) and turns the single LightGBM
into an N-member ensemble. The spread across members is the predictive uncertainty
(σ) attached to every GW prediction.

Pipeline per target (homo/lumo/gap):
  1. scaffold split → train / test            (same SEED/TEST_FRAC as train_delta)
  2. scaffold split the train → fit / calib    (calibration set, no leakage)
  3. train N members: each on a bootstrap of fit + its own seed → diverse models
  4. μ = mean(member Δ), σ = std(member Δ); GW prediction = B3LYP + μ
  5. CALIBRATE: a raw ensemble σ is not a trustworthy confidence. Fit a single
     scale s on the calib set so that s·σ matches real error magnitude
     (sigma-scaling recalibration, Levi 2022 / Laves 2020).
  6. report on test: MAE (must track single-model), ENCE before/after calibration,
     1σ/2σ coverage, reliability curve.

Why calibration is non-negotiable: an uncalibrated σ is fake confidence. If the
model says ±0.4 eV, ~68% of molecules must actually fall within ±0.4 eV, else the
flag misleads everything downstream (OOD gating, "send-to-GW" active learning).

Outputs (results/phase10/):
  uq_ensemble_metrics.json              per-target MAE, ENCE before/after, coverage, scale s
  reliability_{homo,lumo,gap}.png       observed vs expected coverage curve
  ensemble_lgbm/{target}_m{k}.txt       N saved boosters per target (reused by inference)
  ensemble_calibration.json             {target: {scale, sigma_mean}} for inference

Usage:
  .venv\\Scripts\\python.exe scripts/phase10/train_ensemble.py
  .venv\\Scripts\\python.exe scripts/phase10/train_ensemble.py --members 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows consoles default to GBK and choke on σ/² in the progress table.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # headless: write PNG, never open a window
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski
from scipy.special import erfinv
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score

from molgap.constants import RESULTS_DIR
from molgap.utils import murcko_scaffold_smiles

PHASE9 = RESULTS_DIR / "phase9"
PHASE10 = RESULTS_DIR / "phase10"
CSV = PHASE9 / "delta_oe62.csv"
NPZ = PHASE9 / "delta_oe62_embeddings.npz"
TARGETS = ("homo", "lumo", "gap")
SEED = 42
TEST_FRAC = 0.2
CALIB_FRAC = 0.15  # carved out of train (scaffold-disjoint) for σ recalibration

# Same base config as Phase 9 train_delta; subsample/colsample give member
# diversity, and we add a per-member bootstrap on top for honest epistemic spread.
LGB_PARAMS = dict(
    n_estimators=1500, learning_rate=0.02, num_leaves=31, max_depth=-1,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
    reg_lambda=2.0, min_child_samples=30, n_jobs=-1, verbose=-1,
)


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


def fit_member(X_tr, y_tr, seed):
    """One ensemble member: bootstrap the rows, fit with an internal early-stop
    split. Bootstrap + distinct seed = the diversity that makes σ meaningful."""
    rng = np.random.RandomState(seed)
    n = len(X_tr)
    boot = rng.randint(0, n, size=n)  # bootstrap resample
    oob = np.setdiff1d(np.arange(n), boot)
    if len(oob) < 20:  # guarantee a non-trivial early-stopping set
        oob = rng.choice(n, size=max(20, n // 10), replace=False)
    params = dict(LGB_PARAMS, random_state=seed)
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr[boot], y_tr[boot], eval_set=[(X_tr[oob], y_tr[oob])],
          callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    return m


def ensemble_predict(members, X):
    """Stack member predictions → (mean, std) over the ensemble axis."""
    P = np.stack([m.predict(X) for m in members], axis=0)  # [n_members, n]
    return P.mean(axis=0), P.std(axis=0)


def ence(errors, sigmas, n_bins=10):
    """Expected Normalized Calibration Error: bin by predicted σ, compare the
    root-mean-variance (RMV) to the RMSE in each bin. 0 = perfectly calibrated.
    Standard regression-UQ metric (Levi et al. 2022)."""
    abs_err = np.abs(errors)
    order = np.argsort(sigmas)
    sigmas, abs_err = sigmas[order], abs_err[order]
    bins = np.array_split(np.arange(len(sigmas)), n_bins)
    total, w = 0.0, 0
    for b in bins:
        if len(b) == 0:
            continue
        rmv = np.sqrt(np.mean(sigmas[b] ** 2))
        rmse = np.sqrt(np.mean(abs_err[b] ** 2))
        if rmv > 1e-9:
            total += len(b) * abs(rmv - rmse) / rmv
            w += len(b)
    return float(total / max(w, 1))


def coverage(errors, sigmas, p):
    """Observed fraction of points whose true value lies in the central p
    interval implied by a Gaussian σ (expected = p if calibrated)."""
    z = np.sqrt(2.0) * erfinv(p)  # half-width in σ units for central mass p
    return float(np.mean(np.abs(errors) <= z * sigmas))


def reliability_curve(errors, sigmas, target, path):
    """Observed vs expected coverage across nominal levels → PNG."""
    levels = np.linspace(0.05, 0.95, 19)
    obs = [coverage(errors, sigmas, p) for p in levels]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(levels, obs, "o-", ms=4, label="observed")
    ax.set_xlabel("expected coverage")
    ax.set_ylabel("observed coverage")
    ax.set_title(f"Reliability — {target.upper()}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=10, help="ensemble size")
    ap.add_argument("--csv", type=str, default=str(CSV))
    ap.add_argument("--npz", type=str, default=str(NPZ))
    ap.add_argument("--out-dir", type=str, default=str(PHASE10))
    ap.add_argument(
        "--feature-mode",
        choices=["embedding", "embedding_desc", "embedding_desc_pred"],
        default="embedding",
    )
    args = ap.parse_args()
    n_members = args.members
    out_dir = Path(args.out_dir)
    ensemble_dir = out_dir / "ensemble_lgbm"

    df = pd.read_csv(args.csv)
    npz = np.load(args.npz, allow_pickle=True)
    e2d, e3d = npz["emb_2d"], npz["emb_3d"]
    assert len(df) == len(e2d) == len(e3d), "row count mismatch csv vs npz"
    assert (npz["smiles"] == df["smiles"].to_numpy()).all(), "smiles order mismatch"

    X = build_features(df, e2d, e3d, args.feature_mode)
    smiles = df["smiles"].tolist()
    scaffolds = np.array([murcko_scaffold_smiles(s) or "NONE" for s in smiles])
    print(f"{len(df)} molecules, feature dim {X.shape[1]}, "
          f"{len(set(scaffolds))} scaffolds, {n_members} members\n")

    # ── Outer scaffold split: train / test (matches train_delta SEED/TEST_FRAC) ──
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    tr, te = next(gss.split(X, groups=scaffolds))

    # ── Inner scaffold split: fit / calib (no scaffold leaks into calibration) ──
    gss_c = GroupShuffleSplit(n_splits=1, test_size=CALIB_FRAC, random_state=SEED)
    fit_rel, cal_rel = next(gss_c.split(X[tr], groups=scaffolds[tr]))
    fit_idx, cal_idx = tr[fit_rel], tr[cal_rel]
    print(f"fit {len(fit_idx)} / calib {len(cal_idx)} / test {len(te)} "
          f"(all scaffold-disjoint)\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble_dir.mkdir(exist_ok=True)
    results = {"n": int(len(df)), "n_members": n_members,
               "feature_mode": args.feature_mode, "feature_dim": int(X.shape[1]),
               "n_fit": int(len(fit_idx)), "n_calib": int(len(cal_idx)),
               "n_test": int(len(te))}
    calib_cfg = {}

    print(f"  {'tgt':4s} {'MAE':>7s} {'R²':>6s} {'scale':>6s} "
          f"{'ENCE↓pre':>9s} {'ENCE↓post':>10s} {'σ̄':>6s} {'cov1σ':>6s} {'cov2σ':>6s}")
    print(f"  {'-'*78}")

    for t in TARGETS:
        y = df[f"delta_{t}"].to_numpy()          # Δ = GW − B3LYP (the regression target)
        pred_b3 = df[f"pred_{t}"].to_numpy()     # fixed B3LYP offset
        gw = df[f"gw_{t}"].to_numpy()

        # Train N diverse members on the fit set.
        members = []
        for k in range(n_members):
            members.append(fit_member(X[fit_idx], y[fit_idx], seed=SEED + k))
            print(f"    [{t}] member {k + 1}/{n_members} trained", flush=True)

        # Calibration set: raw σ, then fit the scale so s·σ matches real error.
        mu_cal, sig_cal = ensemble_predict(members, X[cal_idx])
        err_cal = gw[cal_idx] - (pred_b3[cal_idx] + mu_cal)
        sig_cal = np.clip(sig_cal, 1e-6, None)
        scale = float(np.sqrt(np.mean((err_cal / sig_cal) ** 2)))  # sigma-scaling

        # Test set: final, held-out evaluation.
        mu_te, sig_te = ensemble_predict(members, X[te])
        sig_te = np.clip(sig_te, 1e-6, None)
        gw_pred = pred_b3[te] + mu_te
        err_te = gw[te] - gw_pred

        mae = mean_absolute_error(gw[te], gw_pred)
        r2 = r2_score(gw[te], gw_pred)
        ence_pre = ence(err_te, sig_te)
        ence_post = ence(err_te, sig_te * scale)
        cov1 = coverage(err_te, sig_te * scale, 0.6827)
        cov2 = coverage(err_te, sig_te * scale, 0.9545)
        sig_mean = float((sig_te * scale).mean())

        reliability_curve(err_te, sig_te * scale, t,
                          out_dir / f"reliability_{t}.png")
        for k, m in enumerate(members):
            (ensemble_dir / f"{t}_m{k}.txt").write_text(
                m.booster_.model_to_string(), encoding="utf-8")

        results[t] = {
            "mae": float(mae), "r2": float(r2), "scale": scale,
            "ence_pre": ence_pre, "ence_post": ence_post,
            "sigma_mean": sig_mean,
            "coverage_1sigma": cov1, "coverage_2sigma": cov2,
        }
        calib_cfg[t] = {"scale": scale, "sigma_mean_raw": float(sig_te.mean())}

        print(f"  {t:4s} {mae:7.3f} {r2:6.3f} {scale:6.2f} "
              f"{ence_pre:9.3f} {ence_post:10.3f} {sig_mean:6.3f} "
              f"{cov1:6.3f} {cov2:6.3f}", flush=True)

    (out_dir / "uq_ensemble_metrics.json").write_text(json.dumps(results, indent=2))
    (out_dir / "ensemble_calibration.json").write_text(json.dumps(calib_cfg, indent=2))
    print("\n  MAE should track Phase 9 single-model (UQ must not cost accuracy).")
    print("  cov1σ→0.68 / cov2σ→0.95 after calibration = trustworthy σ.")
    print(f"\nSaved uq_ensemble_metrics.json + reliability_*.png + boosters to {out_dir}")


if __name__ == "__main__":
    main()
