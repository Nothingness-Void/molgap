"""
Parse Gaussian .out files and compare with ML predictions.

Usage:
  python scripts/phase5/parse_gaussian_outputs.py --gout-dir <path_to_gaussian_outputs>

Extracts HOMO/LUMO/gap from Gaussian output, merges with ML predictions,
computes deviation.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from molgap.utils import RESULTS_DIR, ensure_dirs, save_json

OUT_DIR = RESULTS_DIR / "phase5" / "gaussian_validation"
PREDICTIONS_CSV = OUT_DIR / "predictions.csv"
HARTREE_TO_EV = 27.211386245988


def parse_gaussian_output(filepath: Path) -> dict | None:
    """Extract HOMO, LUMO, gap from a Gaussian .out/.log file."""
    text = filepath.read_text(encoding="utf-8", errors="replace")

    if "Normal termination" not in text:
        print(f"  WARNING: {filepath.name} did not terminate normally")
        return None

    # Find orbital energies from the last "Population analysis" block
    # Look for "Alpha  occ." and "Alpha virt." eigenvalues
    occ_energies = []
    first_virt = None
    in_last_occ_block = False

    for line in text.splitlines():
        if "Alpha  occ. eigenvalues" in line:
            vals = re.findall(r"[-]?\d+\.\d+", line)
            occ_energies = [float(v) for v in vals]
            in_last_occ_block = True
            first_virt = None
        elif "Alpha virt. eigenvalues" in line:
            if in_last_occ_block and first_virt is None:
                vals = re.findall(r"[-]?\d+\.\d+", line)
                if vals:
                    first_virt = float(vals[0])
                    in_last_occ_block = False

    if not occ_energies or first_virt is None:
        print(f"  WARNING: {filepath.name} no orbital energies found")
        return None

    homo_hartree = occ_energies[-1]
    lumo_hartree = first_virt
    homo_ev = homo_hartree * HARTREE_TO_EV
    lumo_ev = lumo_hartree * HARTREE_TO_EV
    gap_ev = lumo_ev - homo_ev

    return {
        "homo_gaussian": homo_ev,
        "lumo_gaussian": lumo_ev,
        "gap_gaussian": gap_ev,
        "homo_hartree": homo_hartree,
        "lumo_hartree": lumo_hartree,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gout-dir", type=Path, required=True,
                        help="Directory containing Gaussian .out files")
    args = parser.parse_args()

    ensure_dirs(OUT_DIR)

    pred_df = pd.read_csv(PREDICTIONS_CSV)
    print(f"  Loaded {len(pred_df)} ML predictions", flush=True)

    out_files = list(args.gout_dir.glob("*.out")) + list(args.gout_dir.glob("*.log"))
    print(f"  Found {len(out_files)} Gaussian output files", flush=True)

    gaussian_results = {}
    for f in sorted(out_files):
        name = f.stem
        result = parse_gaussian_output(f)
        if result:
            gaussian_results[name] = result
            print(f"  {name}: HOMO={result['homo_gaussian']:.4f} "
                  f"LUMO={result['lumo_gaussian']:.4f} "
                  f"Gap={result['gap_gaussian']:.4f} eV", flush=True)
        else:
            print(f"  {name}: FAILED to parse", flush=True)

    # Merge with predictions
    rows = []
    for _, row in pred_df.iterrows():
        name = row["name"]
        safe_name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")

        r = row.to_dict()
        if safe_name in gaussian_results:
            g = gaussian_results[safe_name]
            r.update(g)
            r["homo_diff"] = r["homo_pred"] - g["homo_gaussian"]
            r["lumo_diff"] = r["lumo_pred"] - g["lumo_gaussian"]
            r["gap_diff"] = r["gap_pred"] - g["gap_gaussian"]
            r["homo_abs_err"] = abs(r["homo_diff"])
            r["lumo_abs_err"] = abs(r["lumo_diff"])
            r["gap_abs_err"] = abs(r["gap_diff"])
        rows.append(r)

    result_df = pd.DataFrame(rows)

    matched = result_df.dropna(subset=["homo_gaussian"])
    if not matched.empty:
        print(f"\n{'='*70}", flush=True)
        print(f"  ML vs Gaussian Comparison ({len(matched)} molecules)", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  {'Name':<18s} {'HOMO_ML':>8s} {'HOMO_G':>8s} {'Δ':>7s}  "
              f"{'LUMO_ML':>8s} {'LUMO_G':>8s} {'Δ':>7s}  "
              f"{'Gap_ML':>7s} {'Gap_G':>7s} {'Δ':>7s}", flush=True)
        print(f"  {'-'*95}", flush=True)

        for _, r in matched.iterrows():
            print(f"  {r['name']:<18s} "
                  f"{r['homo_pred']:8.4f} {r['homo_gaussian']:8.4f} {r['homo_diff']:+7.4f}  "
                  f"{r['lumo_pred']:8.4f} {r['lumo_gaussian']:8.4f} {r['lumo_diff']:+7.4f}  "
                  f"{r['gap_pred']:7.4f} {r['gap_gaussian']:7.4f} {r['gap_diff']:+7.4f}",
                  flush=True)

        print(f"\n  Mean Absolute Error:", flush=True)
        print(f"    HOMO: {matched['homo_abs_err'].mean():.4f} eV", flush=True)
        print(f"    LUMO: {matched['lumo_abs_err'].mean():.4f} eV", flush=True)
        print(f"    Gap:  {matched['gap_abs_err'].mean():.4f} eV", flush=True)

        summary = {
            "n_compared": len(matched),
            "homo_mae": float(matched["homo_abs_err"].mean()),
            "lumo_mae": float(matched["lumo_abs_err"].mean()),
            "gap_mae": float(matched["gap_abs_err"].mean()),
            "avg_mae": float((matched["homo_abs_err"].mean() +
                              matched["lumo_abs_err"].mean() +
                              matched["gap_abs_err"].mean()) / 3),
        }
        save_json(summary, OUT_DIR / "gaussian_comparison_summary.json")

    comparison_path = OUT_DIR / "ml_vs_gaussian.csv"
    result_df.to_csv(comparison_path, index=False, encoding="utf-8")
    print(f"\n  Saved: {comparison_path}", flush=True)


if __name__ == "__main__":
    main()
