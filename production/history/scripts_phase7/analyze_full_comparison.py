"""Deeper analysis of results/phase7/full_comparison/experimental_3models.csv:
by-source breakdown, per-model bias coefficients, worst molecules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from molgap.constants import RESULTS_DIR

CSV = RESULTS_DIR / "phase7" / "full_comparison" / "experimental_3models.csv"
MODELS = ["2d", "3d", "hybrid"]
TARGETS = ["homo", "lumo", "gap"]


def main():
    df = pd.read_csv(CSV)
    print(f"Total: {len(df)} molecules")
    print(f"Sources: {df['source'].value_counts().to_dict()}\n")

    # ── Per-model systematic bias (pred - measured), whole set ──
    print("=" * 70)
    print("  Systematic bias  mean(pred - measured) eV  [= correction offset]")
    print("=" * 70)
    print(f"  {'':8s}  {'HOMO':>16s}  {'LUMO':>16s}  {'Gap':>16s}")
    for m in MODELS:
        row = f"  {m:8s}  "
        for t in TARGETS:
            err = df[f"{t}_{m}"] - df[t]
            row += f"{err.mean():+6.3f}±{err.std():4.2f}    "
        print(row)

    # ── By-source corrected MAE (bias removed per source) ──
    print("\n" + "=" * 70)
    print("  By-source bias-corrected MAE (eV)")
    print("=" * 70)
    for src in ["OLED", "HOPV15"]:
        sub = df[df["source"] == src]
        print(f"\n  [{src}] n={len(sub)}")
        print(f"  {'':8s}  {'HOMO':>7s}  {'LUMO':>7s}  {'Gap':>7s}  {'avg':>7s}")
        for m in MODELS:
            maes = []
            for t in TARGETS:
                err = sub[f"{t}_{m}"] - sub[t]
                corrected = err - err.mean()
                maes.append(np.abs(corrected).mean())
            print(f"  {m:8s}  {maes[0]:7.4f}  {maes[1]:7.4f}  {maes[2]:7.4f}  {np.mean(maes):7.4f}")

    # ── Worst molecules by hybrid Gap abs error ──
    print("\n" + "=" * 70)
    print("  Worst 8 molecules (hybrid Gap |error|, raw)")
    print("=" * 70)
    df["gap_abserr_hy"] = (df["gap_hybrid"] - df["gap"]).abs()
    worst = df.nlargest(8, "gap_abserr_hy")
    print(f"  {'name':16s} {'src':7s} {'Gap_meas':>8s} {'Gap_2d':>7s} {'Gap_3d':>7s} {'Gap_hy':>7s}")
    for _, r in worst.iterrows():
        print(f"  {r['name']:16s} {r['source']:7s} {r['gap']:8.2f} "
              f"{r['gap_2d']:7.2f} {r['gap_3d']:7.2f} {r['gap_hybrid']:7.2f}")

    # ── Gap: which model wins per molecule ──
    print("\n" + "=" * 70)
    print("  Per-molecule Gap winner (smallest |error|)")
    print("=" * 70)
    errs = {m: (df[f"gap_{m}"] - df["gap"]).abs() for m in MODELS}
    edf = pd.DataFrame(errs)
    winner = edf.idxmin(axis=1)
    print(f"  win counts: {winner.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
