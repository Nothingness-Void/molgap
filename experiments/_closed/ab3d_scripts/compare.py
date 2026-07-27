"""A/B step 5: aggregate the three arms into a comparison table.

Reads encoder_<name>.json (resources + standalone 3D metrics) and fusion_<name>.json
(deployment-style fusion metrics) for every arm present, and writes a JSON + a
markdown table.

Decision read:
  · standalone 3D Gap MAE/R²  → architecture capability (primary, leak-free).
  · sec/epoch + peak mem      → "运行压力 / 运行时间" on RTX 5060.
  · fusion Gap MAE/R²         → end-to-end 2D+3D confirmation.

Outputs (results/ab3d/):
  comparison.json
  comparison.md

Usage:
  .venv\\Scripts\\python.exe scripts/ab3d/compare.py
"""
from __future__ import annotations

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import json

from molgap.constants import AB_ENCODERS, RESULTS_DIR

OUT = RESULTS_DIR / "ab3d"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main():
    rows = []
    for name in AB_ENCODERS:
        enc = _load(OUT / f"encoder_{name}.json")
        fus = _load(OUT / f"fusion_{name}.json")
        if enc is None:
            print(f"  (skip {name}: no encoder_{name}.json yet)")
            continue
        rows.append({
            "encoder": name,
            "n_params": enc["n_params"],
            "use_charges": enc["use_charges"],
            "sec_per_epoch": enc["sec_per_epoch"],
            "train_seconds": enc["train_seconds"],
            "peak_train_mem_mb": enc["peak_train_mem_mb"],
            "epochs_ran": enc["epochs_ran"],
            "solo_gap_mae": enc["test"]["Gap"]["mae"],
            "solo_gap_r2": enc["test"]["Gap"]["r2"],
            "solo_avg_mae": enc["test"]["average"]["mae"],
            "fusion_gap_mae": fus["test"]["Gap"]["mae"] if fus else None,
            "fusion_gap_r2": fus["test"]["Gap"]["r2"] if fus else None,
        })

    (OUT / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def f(v, fmt):
        return fmt.format(v) if v is not None else "—"

    lines = [
        "# 3D encoder A/B (10k subset, scaffold split, fixed budget)",
        "",
        "| encoder | params | charges | s/epoch | total s | peak mem (MB) | epochs | **solo Gap MAE** | solo Gap R² | solo avg MAE | fusion Gap MAE | fusion Gap R² |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['encoder']} | {r['n_params']:,} | {r['use_charges']} | "
            f"{f(r['sec_per_epoch'], '{:.2f}')} | {f(r['train_seconds'], '{:.0f}')} | "
            f"{f(r['peak_train_mem_mb'], '{:.0f}')} | {r['epochs_ran']} | "
            f"**{f(r['solo_gap_mae'], '{:.4f}')}** | {f(r['solo_gap_r2'], '{:.4f}')} | "
            f"{f(r['solo_avg_mae'], '{:.4f}')} | {f(r['fusion_gap_mae'], '{:.4f}')} | "
            f"{f(r['fusion_gap_r2'], '{:.4f}')} |"
        )
    lines += [
        "",
        "- **solo** = 3D encoder trained end-to-end with its own head (leak-free, scaffold split) — primary discriminator.",
        "- **fusion** = shared 2D GPS + this 3D encoder via gated FusionHead.",
        "- charges: SchNet uses Gasteiger charges (deployed form); ViSNet/TensorNet use Z+geometry (native).",
    ]
    (OUT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] Saved {OUT/'comparison.json'} and {OUT/'comparison.md'}")


if __name__ == "__main__":
    main()
