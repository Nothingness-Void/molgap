"""
17_build_master_experiment_table.py — build the master experiment log.

Collects ALL experiment results across phases into one CSV:
  results/master_experiment_log.csv

Two sources of data:
  1. Hard-coded collectors for historical experiments (Phase 1.1-1.5, Phase 2.1-2.5)
  2. Auto-scan: any JSON file in results/experiments/ with the standard schema
     is automatically included. This is the preferred way to add new experiments.

Standard experiment JSON schema (place in results/experiments/):
  {
    "phase": "3",
    "sub_stage": "3.1",
    "experiment": "production_chonsfcl_50k",
    "model": "lightgbm_tuned",
    "data_desc": "50k CHONSFCl",
    "elements": "C,Cl,F,H,N,O,S",
    "mw_range": "200-500",
    "n_data": 50000,
    "split": "random_test",
    "metrics": {
      "homo":    {"mae": 0.13, "rmse": 0.18, "r2": 0.89},
      "lumo":    {"mae": 0.14, "rmse": 0.20, "r2": 0.94},
      "gap":     {"mae": 0.18, "rmse": 0.27, "r2": 0.90},
      "average": {"mae": 0.15, "rmse": 0.22, "r2": 0.91}
    }
  }

Regenerate the table at any time:
  python scripts/experiments/17_build_master_experiment_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
AUTO_DIR = RESULTS / "experiments"
OUT = RESULTS / "master_experiment_log.csv"

COL_ORDER = [
    "phase", "sub_stage", "experiment", "model", "data_desc",
    "elements", "mw_range", "n_data", "split",
    "homo_mae", "homo_rmse", "homo_r2",
    "lumo_mae", "lumo_rmse", "lumo_r2",
    "gap_mae", "gap_rmse", "gap_r2",
    "average_mae", "average_rmse", "average_r2",
]


def make_row(phase, sub, experiment, model, data_desc, elements, mw_range,
             n_data, split, metrics):
    r = {
        "phase": phase,
        "sub_stage": sub,
        "experiment": experiment,
        "model": model,
        "data_desc": data_desc,
        "elements": elements,
        "mw_range": mw_range,
        "n_data": n_data,
        "split": split,
    }
    for target in ["homo", "lumo", "gap", "average"]:
        if target in metrics:
            for m in ["mae", "rmse", "r2"]:
                r[f"{target}_{m}"] = metrics[target].get(m)
    return r


# ── Historical collectors (Phase 1 & 2) ────────────────────

def collect_phase1_1(rows):
    """1.1 Baseline models — 10k CHON MW200-300."""
    csv_path = RESULTS / "phase1" / "baseline" / "model_comparison_baseline.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    for _, r_ in df[df["split"] == "test"].iterrows():
        m = r_["model"]
        metrics = {t: {k: r_[f"{t}_{k}"] for k in ["mae", "rmse", "r2"]}
                   for t in ["homo", "lumo", "gap", "average"]}
        rows.append(make_row("1", "1.1", f"baseline_{m}", m,
                             "10k CHON", "C,H,N,O", "200-300", 10000,
                             "random_test", metrics))


def collect_phase1_2(rows):
    """1.2 Optuna tuning — 10k (hardcoded, file was overwritten by 30k run)."""
    rows.append(make_row("1", "1.2", "optuna_tuned_lgbm_10k", "lightgbm_tuned",
                         "10k CHON", "C,H,N,O", "200-300", 10000, "random_test",
                         {"homo": {"mae": 0.1355, "rmse": 0.2010, "r2": 0.882},
                          "lumo": {"mae": 0.1413, "rmse": 0.2126, "r2": 0.946},
                          "gap":  {"mae": 0.1799, "rmse": 0.2906, "r2": 0.908},
                          "average": {"mae": 0.1522, "rmse": 0.2347, "r2": 0.912}}))
    rows.append(make_row("1", "1.2", "optuna_tuned_lgbm_10k", "lightgbm_tuned",
                         "10k CHON", "C,H,N,O", "200-300", 10000, "scaffold_test",
                         {"homo": {"mae": 0.1637, "rmse": 0.2267, "r2": 0.851},
                          "lumo": {"mae": 0.1766, "rmse": 0.2532, "r2": 0.917},
                          "gap":  {"mae": 0.2294, "rmse": 0.3139, "r2": 0.876},
                          "average": {"mae": 0.1899, "rmse": 0.2646, "r2": 0.881}}))


def collect_phase1_3(rows):
    """1.3 Embedding experiments — 10k CHON."""
    csv_path = RESULTS / "phase1" / "embeddings" / "embedding_model_comparison.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    for _, r_ in df.iterrows():
        exp = r_["experiment"]
        metrics = {t: {k: r_[f"{t}_{k}"] for k in ["mae", "rmse", "r2"]}
                   for t in ["homo", "lumo", "gap", "average"]}
        rows.append(make_row("1", "1.3", f"emb_{exp}", exp,
                             "10k CHON", "C,H,N,O", "200-300", 10000,
                             "random_test", metrics))


def collect_phase1_4(rows):
    """1.4 Advanced models — 30k CHON."""
    csv_path = RESULTS / "phase1" / "advanced" / "advanced_model_comparison.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    for _, r_ in df.iterrows():
        m = r_["model"]
        metrics = {t: {k: r_[f"{t}_{k}"] for k in ["mae", "rmse", "r2"]}
                   for t in ["homo", "lumo", "gap", "average"]}
        rows.append(make_row("1", "1.4", f"adv_{m}", m,
                             "30k CHON", "C,H,N,O", "200-300", 30000,
                             "random_test", metrics))


def collect_phase1_5(rows):
    """1.5 Data scaling — 30k tuned LightGBM."""
    json_path = RESULTS / "phase1" / "tuning" / "tuning_result_summary.json"
    if not json_path.exists():
        return
    d = json.load(open(json_path, encoding="utf-8"))
    rows.append(make_row("1", "1.5", "data_scaling_30k_tuned", "lightgbm_tuned",
                         "30k CHON", "C,H,N,O", "200-300", 30000,
                         "random_test", d["random_test"]))
    rows.append(make_row("1", "1.5", "data_scaling_30k_tuned", "lightgbm_tuned",
                         "30k CHON", "C,H,N,O", "200-300", 30000,
                         "scaffold_test", d["scaffold_test"]))


def collect_phase2(rows):
    """2.x Generalization steps."""
    gen_dir = RESULTS / "phase2" / "generalization"
    if not gen_dir.exists():
        return
    step_map = {
        "step0_chon_mw200_300":     ("2.1", "C,H,N,O",        "200-300"),
        "step1_chon_mw200_500":     ("2.2", "C,H,N,O",        "200-500"),
        "step2_chons_mw200_500":    ("2.3", "C,H,N,O,S",      "200-500"),
        "step3_chonsf_mw200_500":   ("2.4", "C,F,H,N,O,S",    "200-500"),
        "step4_chonsfcl_mw200_500": ("2.5", "C,Cl,F,H,N,O,S", "200-500"),
    }
    for step_file in sorted(gen_dir.glob("step*_metrics.json")):
        d = json.load(open(step_file, encoding="utf-8"))
        step_name = d["step"]
        if step_name in step_map:
            sub, elems, mwr = step_map[step_name]
            rows.append(make_row("2", sub, f"gen_{step_name}", "lightgbm_tuned",
                                 f"10k {elems}", elems, mwr,
                                 d["n_molecules"], "random_test", d["metrics"]))


# ── Auto-scan collector ─────────────────────────────────────

def collect_auto(rows):
    """Scan results/experiments/*.json for new experiments in standard schema."""
    if not AUTO_DIR.exists():
        return
    seen = {r["experiment"] for r in rows}
    for json_file in sorted(AUTO_DIR.glob("*.json")):
        if json_file.name.startswith("_"):
            continue
        try:
            d = json.load(open(json_file, encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"  WARNING: skipping invalid JSON: {json_file}")
            continue

        if "metrics" not in d or "experiment" not in d:
            print(f"  WARNING: skipping {json_file.name} (missing required fields)")
            continue

        exp_name = d["experiment"]
        if exp_name in seen:
            continue
        seen.add(exp_name)

        rows.append(make_row(
            phase=d.get("phase", "?"),
            sub=d.get("sub_stage", "?"),
            experiment=exp_name,
            model=d.get("model", "unknown"),
            data_desc=d.get("data_desc", ""),
            elements=d.get("elements", ""),
            mw_range=d.get("mw_range", ""),
            n_data=d.get("n_data", 0),
            split=d.get("split", "random_test"),
            metrics=d["metrics"],
        ))
        print(f"  [auto] added {json_file.name}: {exp_name}")


# ── Main ────────────────────────────────────────────────────

def build():
    rows = []

    # Historical collectors
    collect_phase1_1(rows)
    collect_phase1_2(rows)
    collect_phase1_3(rows)
    collect_phase1_4(rows)
    collect_phase1_5(rows)
    collect_phase2(rows)

    # Auto-scan for new experiments
    collect_auto(rows)

    # Build DataFrame
    master = pd.DataFrame(rows)
    for c in COL_ORDER:
        if c not in master.columns:
            master[c] = None
    master = master[COL_ORDER]
    master.to_csv(OUT, index=False, encoding="utf-8")

    print(f"\nMaster experiment log: {len(master)} rows")
    print(f"Saved to: {OUT}\n")

    print("=" * 100)
    print("MASTER EXPERIMENT LOG")
    print("=" * 100)
    cols = ["phase", "sub_stage", "experiment", "data_desc", "split",
            "average_mae", "average_r2"]
    print(master[cols].to_string(index=False))
    print()

    # Phase summary
    print("=" * 100)
    print("BEST PER SUB-STAGE")
    print("=" * 100)
    random_only = master[master["split"] == "random_test"].copy()
    if not random_only.empty:
        best = random_only.loc[random_only.groupby("sub_stage")["average_mae"].idxmin()]
        print(best[["phase", "sub_stage", "experiment", "data_desc",
                     "average_mae", "average_r2"]].to_string(index=False))


if __name__ == "__main__":
    build()
