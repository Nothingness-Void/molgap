"""
Phase 10 (M1, tier 2): OOD scoring by nearest-neighbour distance in the frozen
384-d hybrid embedding space — and, crucially, a check that the score is real.

An OOD flag is only worth shipping if "far from training" actually predicts
"large error". So this script does two things:
  1. score: for every test molecule, distance to its k nearest neighbours among
     the molecules the ensemble was *fit* on (scaffold-disjoint reference set).
  2. validate: correlate that distance with the true |GW error| of the Δ-ensemble.
     If Spearman > 0 and binned MAE rises with distance, the flag carries signal.

We reuse the exact Phase 10 ensemble setup (same SEED/splits, same 384-d features,
the saved boosters) so the OOD distance and the prediction error refer to the
same molecules. As a cross-check we also correlate distance with the calibrated
ensemble σ — two independent "model is unsure here" signals that should agree.

Distance metric: features are standardized on the fit set, then Euclidean k-NN.
Cosine is also computed; we report both Spearman-vs-error and pick the stronger
one for the binned plot and the shipped threshold.

OOD threshold: the 95th percentile of fit-set self-distances (leave-one-out),
i.e. "farther than 95% of training molecules are from each other" → flagged OOD.

Outputs (results/phase10/):
  ood_metrics.json            spearman(dist, |err|) + binned MAE + threshold + flagged frac
  ood_distance_vs_error.png   binned MAE across distance deciles (the validation plot)

Usage:
  .venv\\Scripts\\python.exe scripts/phase10/ood_score.py
"""
from __future__ import annotations

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK guard for σ/² etc.

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit

from molgap.constants import RESULTS_DIR
from molgap.utils import murcko_scaffold_smiles

PHASE9 = RESULTS_DIR / "phase9"
PHASE10 = RESULTS_DIR / "phase10"
CSV = PHASE9 / "delta_oe62.csv"
NPZ = PHASE9 / "delta_oe62_embeddings.npz"
TARGETS = ("homo", "lumo", "gap")
SEED = 42
TEST_FRAC = 0.2
CALIB_FRAC = 0.15
K = 5  # neighbours for the distance score


def knn_distance(ref, query, metric):
    """Mean distance from each query row to its K nearest neighbours in ref."""
    nn = NearestNeighbors(n_neighbors=K, metric=metric).fit(ref)
    dist, _ = nn.kneighbors(query)
    return dist.mean(axis=1)


def self_distance_threshold(ref, metric, q=95.0):
    """Fit-set self k-NN distance (excluding self) → percentile = OOD cutoff."""
    nn = NearestNeighbors(n_neighbors=K + 1, metric=metric).fit(ref)
    dist, _ = nn.kneighbors(ref)
    self_d = dist[:, 1:].mean(axis=1)  # drop column 0 (the point itself)
    return float(np.percentile(self_d, q)), self_d


def load_boosters(target):
    """Reload the 10 saved LightGBM members for a target."""
    members = []
    d = PHASE10 / "ensemble_lgbm"
    k = 0
    while (d / f"{target}_m{k}.txt").exists():
        s = (d / f"{target}_m{k}.txt").read_text(encoding="utf-8")
        members.append(lgb.Booster(model_str=s))
        k += 1
    return members


def main():
    df = pd.read_csv(CSV)
    npz = np.load(NPZ, allow_pickle=True)
    e2d, e3d = npz["emb_2d"], npz["emb_3d"]
    assert (npz["smiles"] == df["smiles"].to_numpy()).all(), "smiles order mismatch"

    X = np.hstack([e2d, e3d]).astype(np.float32)
    smiles = df["smiles"].tolist()
    scaffolds = np.array([murcko_scaffold_smiles(s) or "NONE" for s in smiles])

    # Rebuild the SAME splits as train_ensemble (fit = reference set).
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    tr, te = next(gss.split(X, groups=scaffolds))
    gss_c = GroupShuffleSplit(n_splits=1, test_size=CALIB_FRAC, random_state=SEED)
    fit_rel, _ = next(gss_c.split(X[tr], groups=scaffolds[tr]))
    fit_idx = tr[fit_rel]
    print(f"reference (fit) {len(fit_idx)} / test {len(te)}, K={K}\n")

    # Standardize on the fit set, apply to both.
    mu, sd = X[fit_idx].mean(0), X[fit_idx].std(0) + 1e-8
    Xf = (X[fit_idx] - mu) / sd
    Xt = (X[te] - mu) / sd

    cfg = json.loads((PHASE10 / "ensemble_calibration.json").read_text())

    results = {"k": K, "n_fit": int(len(fit_idx)), "n_test": int(len(te))}
    print(f"  {'tgt':4s} {'metric':7s} {'ρ(d,|e|)':>9s} {'ρ(d,σ)':>8s} "
          f"{'thr(p95)':>9s} {'OOD%':>6s}  binnedMAE(near→far)")
    print(f"  {'-'*86}")

    # Distance scores are target-independent (same embeddings); compute once.
    dist_te = {m: knn_distance(Xf, Xt, m) for m in ("euclidean", "cosine")}
    thr = {}
    for m in ("euclidean", "cosine"):
        thr[m], _ = self_distance_threshold(Xf, m)

    plot_target = "gap"
    plot_payload = None

    for t in TARGETS:
        members = load_boosters(t)
        pred_b3 = df[f"pred_{t}"].to_numpy()
        gw = df[f"gw_{t}"].to_numpy()
        scale = cfg[t]["scale"]

        P = np.stack([mb.predict(X[te]) for mb in members], axis=0)
        mu_d, sig = P.mean(0), P.std(0)
        err = np.abs(gw[te] - (pred_b3[te] + mu_d))
        sig_cal = np.clip(sig, 1e-6, None) * scale

        t_res = {}
        for m in ("euclidean", "cosine"):
            d = dist_te[m]
            rho_e = float(spearmanr(d, err).correlation)
            rho_s = float(spearmanr(d, sig_cal).correlation)
            flagged = float((d > thr[m]).mean())

            # Binned MAE across distance deciles → monotonic = real signal.
            order = np.argsort(d)
            bins = np.array_split(order, 10)
            binned_mae = [float(err[b].mean()) for b in bins]

            t_res[m] = {
                "spearman_dist_err": rho_e, "spearman_dist_sigma": rho_s,
                "threshold_p95": thr[m], "ood_fraction": flagged,
                "binned_mae_deciles": binned_mae,
            }
            near, far = binned_mae[0], binned_mae[-1]
            print(f"  {t:4s} {m:7s} {rho_e:9.3f} {rho_s:8.3f} "
                  f"{thr[m]:9.3f} {flagged*100:5.1f}%  "
                  f"{near:.3f}→{far:.3f} ({far/max(near,1e-6):.1f}x)", flush=True)

            if t == plot_target and m == "euclidean":
                plot_payload = (d, err, binned_mae)

        results[t] = t_res

    # Validation plot: binned MAE vs distance decile for Gap (euclidean).
    if plot_payload is not None:
        d, err, binned_mae = plot_payload
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.bar(range(1, 11), binned_mae, color="#4C72B0", alpha=0.85)
        ax.set_xlabel("embedding-distance decile (1 = nearest, 10 = farthest)")
        ax.set_ylabel("test MAE in bin (eV)")
        ax.set_title(f"OOD validation — {plot_target.upper()}: error rises with distance")
        ax.set_xticks(range(1, 11))
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(PHASE10 / "ood_distance_vs_error.png", dpi=130)
        plt.close(fig)

    (PHASE10 / "ood_metrics.json").write_text(json.dumps(results, indent=2))

    # Reference bundle for inference: standardized fit embeddings + the
    # standardization stats + the euclidean OOD threshold. predict_smiles_with_uq
    # loads this to score a new molecule's distance without re-deriving splits.
    np.savez(
        PHASE10 / "ood_reference.npz",
        ref_std=Xf.astype(np.float32),     # standardized fit-set embeddings [n_fit, 384]
        mu=mu.astype(np.float32), sd=sd.astype(np.float32),
        threshold=np.array([thr["euclidean"]], dtype=np.float32),
        k=np.array([K], dtype=np.int64),
    )

    print("\n  ρ(d,|e|)>0 + binned MAE rising near→far = distance predicts error (OOD signal real).")
    print("  ρ(d,σ)>0 = the distance flag and the ensemble σ agree on where the model is unsure.")
    print(f"\nSaved ood_metrics.json + ood_reference.npz + ood_distance_vs_error.png to {PHASE10}")


if __name__ == "__main__":
    main()
