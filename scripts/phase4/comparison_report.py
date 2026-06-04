"""
Phase 4 Step 5: Final comparison report across all models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import RESULTS_DIR, TARGET_COLS, ensure_dirs, save_json

OUT_DIR = RESULTS_DIR / "phase4"
PHASE3_OPT = RESULTS_DIR / "phase3" / "optimize"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    ensure_dirs(OUT_DIR)
    print("=== Phase 4 Step 5: Final Comparison Report ===\n")

    rows = []

    # Phase 3 baseline
    rows.append({"model": "Phase3_baseline(no_select)", "source": "phase3",
                 "average_mae": 0.1706, "average_r2": 0.8755})

    # Phase 3 optimized models
    p3_comp = PHASE3_OPT / "model_comparison.csv"
    if p3_comp.exists():
        df3 = pd.read_csv(p3_comp)
        for _, r in df3.iterrows():
            if r["model"] == "Phase3_baseline(no_select)":
                continue
            row = {"model": f"P3_{r['model']}", "source": "phase3"}
            for col in df3.columns:
                if col != "model":
                    row[col] = r[col]
            rows.append(row)

    # Phase 4 ensemble
    ens_path = OUT_DIR / "ensemble_summary.json"
    if ens_path.exists():
        ens = load_json(ens_path)
        rows.append({"model": "P4_Blend_weighted", "source": "phase4_ensemble",
                     "average_mae": ens["blend_mae"], "average_r2": ens["blend_r2"]})
        rows.append({"model": "P4_Stack_ridge", "source": "phase4_ensemble",
                     "average_mae": ens["stack_mae"], "average_r2": ens["stack_r2"]})

    ens_csv = OUT_DIR / "ensemble_comparison.csv"
    if ens_csv.exists():
        df_ens = pd.read_csv(ens_csv)
        for _, r in df_ens.iterrows():
            if r["model"] in ["Blend_weighted", "Blend_average", "Stack_ridge"]:
                continue
            existing = [x["model"] for x in rows]
            name = f"P4_ens_{r['model']}"
            if name not in existing:
                row = {"model": name, "source": "phase4_ensemble"}
                for col in df_ens.columns:
                    if col != "model":
                        row[col] = r.get(col)
                rows.append(row)

    # Phase 4 per-target
    pt_path = OUT_DIR / "per_target_summary.json"
    if pt_path.exists():
        pt = load_json(pt_path)
        m = pt["metrics"]
        row = {"model": "P4_PerTarget_tuned", "source": "phase4_per_target",
               "average_mae": m["average"]["mae"], "average_r2": m["average"]["r2"]}
        for t in TARGET_COLS:
            row[f"{t}_mae"] = m[t]["mae"]
            row[f"{t}_r2"] = m[t]["r2"]
        rows.append(row)

    # Phase 4 SchNet 3D
    schnet_path = OUT_DIR / "schnet_metrics.json"
    if schnet_path.exists():
        sch = load_json(schnet_path)
        m = sch["metrics"]
        row = {"model": "P4_GNN_SchNet_3D", "source": "phase4_gnn_3d",
               "average_mae": m["average"]["mae"], "average_r2": m["average"]["r2"]}
        for t in TARGET_COLS:
            row[f"{t}_mae"] = m[t]["mae"]
            row[f"{t}_r2"] = m[t]["r2"]
        rows.append(row)

    # Phase 4 GNN
    gnn_path = OUT_DIR / "gnn_metrics.json"
    if gnn_path.exists():
        gnn = load_json(gnn_path)
        m = gnn["metrics"]
        row = {"model": "P4_GNN_AttentiveFP", "source": "phase4_gnn",
               "average_mae": m["average"]["mae"], "average_r2": m["average"]["r2"]}
        for t in TARGET_COLS:
            row[f"{t}_mae"] = m[t]["mae"]
            row[f"{t}_r2"] = m[t]["r2"]
        rows.append(row)

    # Build comparison table
    df = pd.DataFrame(rows)
    df = df.sort_values("average_r2", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_DIR / "model_comparison_final.csv", index=False)

    print("  MODEL COMPARISON (sorted by avg R2)")
    print("  " + "="*65)
    for _, r in df.iterrows():
        mae = r.get("average_mae", float("nan"))
        r2 = r.get("average_r2", float("nan"))
        print(f"  {r['model']:35s}  MAE={mae:.4f}  R2={r2:.4f}")

    best = df.iloc[0]
    print(f"\n  BEST: {best['model']}  MAE={best['average_mae']:.4f}  R2={best['average_r2']:.4f}")

    baseline_r2 = 0.8853
    improvement = best["average_r2"] - baseline_r2
    print(f"  vs Phase3 best: R2 {'+'if improvement>=0 else ''}{improvement:.4f}")

    save_json({
        "best_model": best["model"],
        "best_mae": float(best["average_mae"]),
        "best_r2": float(best["average_r2"]),
        "improvement_r2": float(improvement),
        "n_models_compared": len(df),
    }, OUT_DIR / "phase4_summary.json")

    print(f"\n  Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
