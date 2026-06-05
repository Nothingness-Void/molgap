"""
Build project-progress visualizations for the current MolGap repository state.

Outputs:
  results/overview/phase_summary.csv
  results/overview/phase2_generalization_curve.png
  results/overview/hard_task_progress.png
  results/overview/model_family_snapshot.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molgap.utils import RESULTS_DIR, ensure_dirs


OUT_DIR = RESULTS_DIR / "overview"
MASTER_LOG = RESULTS_DIR / "master_experiment_log.csv"
PHASE2_SUMMARY = RESULTS_DIR / "phase2" / "generalization" / "generalization_summary.csv"
PHASE4_FINAL = RESULTS_DIR / "phase4" / "model_comparison_final.csv"


def save_phase_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "phase": "P1",
            "focus": "CHON optimization",
            "data_scope": "30k CHON, MW 200-300",
            "best_experiment": "data_scaling_30k_tuned",
            "best_model": "LightGBM tuned",
            "average_mae": 0.1497746636107264,
            "average_r2": 0.920516972523088,
            "note": "Best easy-chemistry traditional result",
        },
        {
            "phase": "P2",
            "focus": "Generalization study",
            "data_scope": "10k per step, expanded chemistry",
            "best_experiment": "gen_step0_chon_mw200_300",
            "best_model": "LightGBM tuned",
            "average_mae": 0.16173711018931436,
            "average_r2": 0.9012428549959606,
            "note": "R2 declines smoothly to 0.8736 by step4",
        },
        {
            "phase": "P3",
            "focus": "Scale-up + optimization",
            "data_scope": "30k CHONSFCl, MW 200-500",
            "best_experiment": "phase3_tuned_lgbm",
            "best_model": "LightGBM tuned",
            "average_mae": 0.1596478487680887,
            "average_r2": 0.8853385942334663,
            "note": "Best traditional model on the hard task",
        },
        {
            "phase": "P4",
            "focus": "Ensemble + GNN",
            "data_scope": "30k CHONSFCl, MW 200-500",
            "best_experiment": "phase4_schnet_3d",
            "best_model": "SchNet 3D",
            "average_mae": 0.1491643190383911,
            "average_r2": 0.894249419371287,
            "note": "Best overall model so far",
        },
        {
            "phase": "P5",
            "focus": "Commercial prediction",
            "data_scope": "Application stage",
            "best_experiment": "template_smoke_test",
            "best_model": "pending final production model",
            "average_mae": None,
            "average_r2": None,
            "note": "Deferred until model/report side is stable",
        },
    ]

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "phase_summary.csv", index=False, encoding="utf-8")
    return summary


def plot_phase2_generalization(phase2: pd.DataFrame) -> None:
    labels = [
        "CHON\n200-300",
        "CHON\n200-500",
        "CHONS\n200-500",
        "CHONSF\n200-500",
        "CHONSFCl\n200-500",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)

    axes[0].plot(labels, phase2["average_r2"], marker="o", linewidth=2.2, color="#1768ac")
    axes[0].set_title("Phase 2 Generalization: Average R²")
    axes[0].set_ylabel("Average R²")
    axes[0].set_ylim(0.84, 0.91)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(labels, phase2["average_mae"], marker="o", linewidth=2.2, color="#c44900")
    axes[1].set_title("Phase 2 Generalization: Average MAE")
    axes[1].set_ylabel("Average MAE (eV)")
    axes[1].set_ylim(0.158, 0.178)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Phase 2: Chemistry Expansion Does Not Cause a Cliff-Edge Failure", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase2_generalization_curve.png", bbox_inches="tight")
    plt.close(fig)


def plot_hard_task_progress() -> None:
    df = pd.DataFrame(
        [
            {"milestone": "P2 Step4\n10k tuned", "average_r2": 0.873570442072875, "average_mae": 0.17543173681374458},
            {"milestone": "P3 Baseline\n30k no select", "average_r2": 0.8755400476869474, "average_mae": 0.17062998444516383},
            {"milestone": "P3 Tuned\n30k LGBM", "average_r2": 0.8853385942334663, "average_mae": 0.1596478487680887},
            {"milestone": "P4 Ridge Stack", "average_r2": 0.8912024164336897, "average_mae": 0.15433068588734075},
            {"milestone": "P4 SchNet 3D", "average_r2": 0.894249419371287, "average_mae": 0.1491643190383911},
        ]
    )

    fig, ax1 = plt.subplots(figsize=(11.5, 5.2), dpi=220)
    ax2 = ax1.twinx()

    bars = ax1.bar(df["milestone"], df["average_r2"], color="#1768ac", alpha=0.9, width=0.62)
    ax2.plot(df["milestone"], df["average_mae"], color="#c44900", marker="o", linewidth=2.4)

    ax1.axhline(0.9, color="#555555", linestyle="--", linewidth=1.0, label="R² = 0.9 target")
    ax1.set_ylabel("Average R²")
    ax2.set_ylabel("Average MAE (eV)")
    ax1.set_ylim(0.865, 0.901)
    ax2.set_ylim(0.145, 0.178)
    ax1.set_title("Hard-Task Progress: CHONSFCl, MW 200-500")

    for bar, value in zip(bars, df["average_r2"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.0006, f"{value:.3f}",
                 ha="center", va="bottom", fontsize=9)
    for x, value in enumerate(df["average_mae"]):
        ax2.text(x, value - 0.0012, f"{value:.3f}", color="#c44900",
                 ha="center", va="top", fontsize=9)

    ax1.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hard_task_progress.png", bbox_inches="tight")
    plt.close(fig)


def plot_model_family_snapshot(phase4_final: pd.DataFrame) -> None:
    selected = phase4_final[
        phase4_final["model"].isin(
            [
                "P4_GNN_SchNet_3D",
                "P4_Stack_ridge",
                "P4_PerTarget_tuned",
                "P3_Tuned_LGBM",
                "P3_Tuned_XGB",
                "P4_GNN_AttentiveFP",
                "P3_HistGBT",
            ]
        )
    ].copy()

    selected["label"] = [
        "SchNet 3D",
        "Ridge stack",
        "Per-target LGBM",
        "Tuned LGBM",
        "Tuned XGB",
        "AttentiveFP",
        "HistGBT",
    ]
    selected = selected.sort_values("average_r2", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=220)
    bars = ax.barh(selected["label"], selected["average_r2"], color="#1768ac", alpha=0.9)
    ax.set_xlim(0.872, 0.898)
    ax.set_xlabel("Average R²")
    ax.set_title("Hard-Task Model Snapshot: Best Traditional, Ensemble, and GNN Models")
    ax.grid(True, axis="x", alpha=0.25)

    for bar, r2, mae in zip(bars, selected["average_r2"], selected["average_mae"]):
        ax.text(r2 + 0.00035, bar.get_y() + bar.get_height() / 2,
                f"R²={r2:.3f} | MAE={mae:.3f}",
                va="center", ha="left", fontsize=8.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_family_snapshot.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ensure_dirs(OUT_DIR)
    plt.style.use("seaborn-v0_8-whitegrid")

    master = pd.read_csv(MASTER_LOG)
    phase2 = pd.read_csv(PHASE2_SUMMARY)
    phase4_final = pd.read_csv(PHASE4_FINAL)

    save_phase_summary(master)
    plot_phase2_generalization(phase2)
    plot_hard_task_progress()
    plot_model_family_snapshot(phase4_final)

    print("Wrote:")
    print(f"  {OUT_DIR / 'phase_summary.csv'}")
    print(f"  {OUT_DIR / 'phase2_generalization_curve.png'}")
    print(f"  {OUT_DIR / 'hard_task_progress.png'}")
    print(f"  {OUT_DIR / 'model_family_snapshot.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
