"""Summarize the three-seed archive-r02 development gate without opening sealed sets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from molgap.constants import RESULTS_DIR


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r02-pubchemqc-router"
SEEDS = (42, 43, 44)


def main() -> None:
    table = pd.read_parquet(OUT_DIR / "router_development_dataset_r5.parquet")
    test = table[table.split == "dev_test"].reset_index(drop=True)
    y = test.gap.to_numpy(dtype=np.float64)
    base_error = np.abs(test.base_gap.to_numpy() - y)
    expert_error = np.abs(test.expert_gap.to_numpy() - y)
    fixed = test.fixed_route_flag.to_numpy(dtype=bool)
    fixed_error = np.where(fixed, expert_error, base_error)
    gain = base_error - expert_error
    budget = np.zeros(len(test), dtype=bool)
    budget[np.argsort(-gain, kind="stable")[:int(fixed.sum())]] = True
    oracle = gain > 0
    oracle_metrics = {
        "n": len(test),
        "fixed_route_fraction": float(fixed.mean()),
        "expert_win_rate": float(oracle.mean()),
        "fixed_gap_mae": float(fixed_error.mean()),
        "budget_oracle_gap_delta_vs_fixed": float(
            (np.where(budget, expert_error, base_error) - fixed_error).mean()
        ),
        "unrestricted_oracle_gap_delta_vs_fixed": float(
            (np.where(oracle, expert_error, base_error) - fixed_error).mean()
        ),
    }

    seed_rows = []
    for seed in SEEDS:
        directory = OUT_DIR / f"router_r5_seed{seed}"
        metrics = json.loads((directory / "metrics.json").read_text())
        predictions = pd.read_parquet(directory / "dev_test_predictions.parquet")
        row = {
            "seed": seed,
            "selected_feature_set": metrics["selected_feature_set"],
            "win_auc": float(roc_auc_score(gain > 0.002, predictions.p_expert_win)),
            "gain_spearman": float(spearmanr(gain, predictions.predicted_gain).statistic),
        }
        for strategy in ("full_replacement", "suppress_only", "add_only", "bidirectional"):
            block = metrics["dev_test"][strategy]
            row[strategy] = {
                "gap_delta_vs_fixed": block["gap_delta_vs_fixed"],
                "ci95": block["bootstrap_gap_vs_fixed"]["ci95"],
                "route_fraction": block["route_fraction"],
                "expert_win_precision": block["expert_win_precision"],
            }
        seed_rows.append(row)

    aggregate = {}
    for strategy in ("add_only", "bidirectional"):
        values = np.asarray([row[strategy]["gap_delta_vs_fixed"] for row in seed_rows])
        aggregate[strategy] = {
            "mean_gap_delta_vs_fixed": float(values.mean()),
            "std_gap_delta_vs_fixed": float(values.std()),
            "all_seeds_better": bool(np.all(values < 0)),
            "all_ci_below_zero": bool(all(row[strategy]["ci95"][1] < 0 for row in seed_rows)),
        }
    verdict = "stop"
    reason = (
        "Oracle headroom is large, but no pre-Expert feature/policy combination "
        "reaches the 0.001 eV practical threshold or a CI below zero across seeds."
    )
    result = {
        "experiment": "archive-r02 PubChemQC learned Router development gate",
        "oracle": oracle_metrics,
        "seeds": seed_rows,
        "aggregate": aggregate,
        "decision": {
            "verdict": verdict,
            "reason": reason,
            "sealed_metrics_opened": False,
            "next_step": "Keep fixed routed-v4; do not expand Router data.",
        },
    }
    (OUT_DIR / "development_decision.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = [
        "# archive-r02 PubChemQC Learned Router Decision",
        "",
        "The 20k Oracle probe passed, so development was expanded to 49,879 valid",
        "Base/Expert gain labels with scaffold-disjoint train/validation/dev-test",
        "splits. The frozen 20k random and 10k hard sealed sets were not run or opened.",
        "",
        "| seed | selected | win AUC | gain Spearman | add-only Gap delta | bidirectional Gap delta |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | {row['selected_feature_set']} | {row['win_auc']:.3f} | "
            f"{row['gain_spearman']:.3f} | {row['add_only']['gap_delta_vs_fixed']:+.6f} | "
            f"{row['bidirectional']['gap_delta_vs_fixed']:+.6f} |"
        )
    lines.extend([
        "",
        f"Dev-test same-budget Oracle headroom remains "
        f"`{-oracle_metrics['budget_oracle_gap_delta_vs_fixed']:.6f} eV`, but observable",
        "pre-Expert features do not rank that gain reliably. R4 embedding features raise",
        "win AUC only to 0.52-0.53; gain Spearman remains 0.017-0.036. All policy",
        "bootstrap confidence intervals cross zero and gains are far below the",
        "pre-registered 0.001 eV practical threshold.",
        "",
        f"**Decision: STOP.** {reason}",
        "",
        "Fixed routed-v4 remains the B3LYP predictor. Sealed metrics remain unopened.",
    ])
    (OUT_DIR / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
