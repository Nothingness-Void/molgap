"""
Phase 7: bootstrap 95% confidence intervals for the experimental comparison.

Zero retraining — resamples the already-saved per-molecule predictions in
results/phase7/full_comparison/experimental_3models.csv to put error bars on
MAE / R2 for each model (GPS 2D / SchNet 3D / Hybrid), raw and bias-corrected.

Answers "is the 0.12 MAE real or sampling noise?" without touching the GPU.

Usage:
  .venv\\Scripts\\python.exe scripts/phase7/bootstrap_ci.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from molgap.constants import RESULTS_DIR, TARGET_COLS

CSV = RESULTS_DIR / "phase7" / "full_comparison" / "experimental_3models.csv"
OUT = RESULTS_DIR / "phase7" / "full_comparison" / "bootstrap_ci.json"

N_BOOT = 2000
SEED = 42
MODELS = {"2d": "GPS 2D", "3d": "SchNet 3D", "hybrid": "Hybrid"}


def boot_ci(y_true, y_pred, n_boot=N_BOOT, seed=SEED):
    """Return point estimate + 95% CI for MAE and R2 via bootstrap resampling."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    maes, r2s = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)  # resample with replacement
        yt, yp = y_true[idx], y_pred[idx]
        maes.append(mean_absolute_error(yt, yp))
        r2s.append(r2_score(yt, yp))
    maes, r2s = np.array(maes), np.array(r2s)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mae_lo": float(np.percentile(maes, 2.5)),
        "mae_hi": float(np.percentile(maes, 97.5)),
        "r2": float(r2_score(y_true, y_pred)),
        "r2_lo": float(np.percentile(r2s, 2.5)),
        "r2_hi": float(np.percentile(r2s, 97.5)),
    }


def main():
    df = pd.read_csv(CSV)
    n = len(df)
    print(f"Loaded {n} experimental molecules from {CSV.name}\n")

    results = {"n": n, "n_boot": N_BOOT}
    for corrected in (False, True):
        tag = "bias_corrected" if corrected else "raw"
        results[tag] = {}
        print(f"{'='*70}\n  {tag.upper()}  (95% CI over {N_BOOT} bootstraps)\n{'='*70}")
        for mk, mname in MODELS.items():
            results[tag][mk] = {}
            maes_avg = []
            print(f"\n  {mname}")
            for t in TARGET_COLS:
                yt = df[t].to_numpy(dtype=float)
                yp = df[f"{t}_{mk}"].to_numpy(dtype=float)
                if corrected:  # remove each model's own mean bias on this target
                    yp = yp - (yp - yt).mean()
                ci = boot_ci(yt, yp)
                results[tag][mk][t] = ci
                maes_avg.append(ci["mae"])
                print(f"    {t:5s} MAE {ci['mae']:.3f} "
                      f"[{ci['mae_lo']:.3f}, {ci['mae_hi']:.3f}]   "
                      f"R2 {ci['r2']:+.3f} [{ci['r2_lo']:+.3f}, {ci['r2_hi']:+.3f}]")
            print(f"    avg MAE {np.mean(maes_avg):.3f} eV")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
