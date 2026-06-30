"""
Phase 9 (P9.2): compute Δ = GW − model-predicted-B3LYP on the in-distribution
OE62 GW molecules, and characterize the residual.

This is the moment of truth for Δ-learning: how far is B3LYP from GW, and does the
gap have learnable structure? We also dump the 192-d embeddings here so the Δ model
(P9.4) trains on exactly these features without recomputing.

Outputs (results/phase9/):
  delta_oe62.csv             per-molecule: smiles, GW, pred-B3LYP, Δ
  delta_oe62_embeddings.npz  emb_2d / emb_3d / smiles (Δ-model features)
  delta_oe62_summary.json    Δ distribution + residual-compression stats

Usage:
  .venv\\Scripts\\python.exe scripts/phase9/compute_delta.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from molgap.constants import RESULTS_DIR
from molgap.inference import load_hybrid, predict_smiles_batch_hybrid

# Reuse the in-distribution screen from the probe script (same dir).
from probe_oe62_indist import (
    gw_homo_lumo, molecule_elements, ALLOWED_ELEMENTS, MW_MIN, MW_MAX,
)

OE62 = "data/raw/oe62_df_5k.json"
OUTDIR = RESULTS_DIR / "phase9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oe62", type=Path, default=Path(OE62))
    parser.add_argument("--hybrid-key", default="phase7_hybrid")
    parser.add_argument("--out-dir", type=Path, default=OUTDIR)
    parser.add_argument("--out-prefix", default="delta_oe62")
    return parser.parse_args()


def pct(a, q):
    return float(np.percentile(a, q))


def main():
    args = parse_args()
    df = pd.read_json(args.oe62, orient="split")
    print(f"Loaded {len(df)} OE62 GW rows\n")

    # ── Collect in-distribution molecules with GW HOMO/LUMO ──
    rows = []
    for _, row in df.iterrows():
        hl = gw_homo_lumo(row)
        if hl is None:
            continue
        gw_h, gw_l = hl
        smi = row.get("canonical_smiles")
        if not isinstance(smi, str) or not smi:
            continue
        els, mw = molecule_elements(smi)
        if els is None or (els - ALLOWED_ELEMENTS) or not (MW_MIN <= mw <= MW_MAX):
            continue
        rows.append({"smiles": smi, "gw_homo": gw_h, "gw_lumo": gw_l,
                     "gw_gap": gw_l - gw_h})
    cand = pd.DataFrame(rows)
    print(f"In-distribution candidates: {len(cand)}")

    # ── Predict B3LYP with the hybrid (and grab embeddings) ──
    print("Loading hybrid, predicting B3LYP (ETKDG + 2D/3D + fusion)...")
    models = load_hybrid(key=args.hybrid_key)
    vi, preds, e2d, e3d = predict_smiles_batch_hybrid(
        cand["smiles"].tolist(), models=models, return_embeddings=True,
    )
    print(f"ETKDG+predict valid: {len(vi)}/{len(cand)}")

    cv = cand.iloc[vi].reset_index(drop=True)
    cv["pred_homo"], cv["pred_lumo"], cv["pred_gap"] = preds[:, 0], preds[:, 1], preds[:, 2]
    for t in ("homo", "lumo", "gap"):
        cv[f"delta_{t}"] = cv[f"gw_{t}"] - cv[f"pred_{t}"]

    # ── Characterize the residual ──
    summary = {"n": int(len(cv)), "hybrid_key": args.hybrid_key}
    print(f"\n{'='*72}\n  Δ = GW − model-B3LYP  ({len(cv)} molecules)\n{'='*72}")
    print(f"  {'':5s} {'Δ mean':>8s} {'Δ std':>8s} {'Δ p5..p95':>16s}   "
          f"{'std(GW)':>8s} {'std(Δ)':>8s} {'compress':>8s}")
    for t in ("homo", "lumo", "gap"):
        d = cv[f"delta_{t}"].to_numpy()
        gw = cv[f"gw_{t}"].to_numpy()
        std_gw, std_d = float(gw.std()), float(d.std())
        compress = std_gw / std_d if std_d > 1e-9 else float("nan")
        summary[t] = {
            "delta_mean": float(d.mean()), "delta_std": std_d,
            "delta_p5": pct(d, 5), "delta_p50": pct(d, 50), "delta_p95": pct(d, 95),
            "abs_delta_mean": float(np.abs(d).mean()),
            "std_gw": std_gw, "residual_compression": compress,
        }
        print(f"  {t:5s} {d.mean():+8.3f} {std_d:8.3f} "
              f"{pct(d,5):+7.2f}..{pct(d,95):+6.2f}   "
              f"{std_gw:8.3f} {std_d:8.3f} {compress:7.1f}x")

    print("\n  Reading: 'compress' = std(absolute GW) / std(Δ). >1 means the residual")
    print("  is tighter than the absolute target → learning Δ is easier than GW.")

    # ── Save ──
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.out_dir / f"{args.out_prefix}.csv"
    npz_out = args.out_dir / f"{args.out_prefix}_embeddings.npz"
    summary_out = args.out_dir / f"{args.out_prefix}_summary.json"
    cv.to_csv(csv_out, index=False)
    np.savez(npz_out,
             emb_2d=e2d, emb_3d=e3d, smiles=cv["smiles"].to_numpy())
    summary_out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {csv_out}")
    print(f"Saved {npz_out}")
    print(f"Saved {summary_out}")


if __name__ == "__main__":
    main()
