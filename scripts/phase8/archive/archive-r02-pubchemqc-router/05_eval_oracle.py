"""Evaluate the independent PubChemQC archive-r02 Oracle Go/No-Go gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from molgap.constants import RESULTS_DIR
from molgap.router import DEFAULT_TARGET_WEIGHTS, oracle_router_analysis


OUT_DIR = RESULTS_DIR / "phase8" / "archive" / "archive-r02-pubchemqc-router"
TARGETS = ("homo", "lumo", "gap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=OUT_DIR / "oracle_probe_predictions.parquet")
    parser.add_argument("--metrics-out", type=Path, default=OUT_DIR / "oracle_probe_metrics.json")
    parser.add_argument("--labels-out", type=Path, default=OUT_DIR / "oracle_probe_gain_labels.parquet")
    parser.add_argument("--decision-out", type=Path, default=OUT_DIR / "oracle_probe_decision.md")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(
    frame: pd.DataFrame, args: argparse.Namespace, weights: np.ndarray
) -> tuple[dict, dict]:
    y = frame[list(TARGETS)].to_numpy(dtype=np.float64)
    base = frame[[f"base_{target}" for target in TARGETS]].to_numpy(dtype=np.float64)
    expert = frame[[f"expert_{target}" for target in TARGETS]].to_numpy(dtype=np.float64)
    fixed = base[:, 2] < 4.0
    return oracle_router_analysis(
        y, base, expert, fixed, target_names=TARGETS, weights=weights,
        win_deltas=(0.0, 0.002, 0.005), n_bootstrap=args.bootstrap, seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.predictions)
    frame = frame[frame.prediction_success].copy().reset_index(drop=True)
    random_frame = frame[frame.sampling_source == "random"].copy().reset_index(drop=True)
    gap_weights = np.array([0.0, 0.0, 1.0])
    random_metrics, _ = evaluate(random_frame, args, gap_weights)
    all_metrics, _ = evaluate(frame, args, gap_weights)
    random_weighted_metrics, _ = evaluate(random_frame, args, DEFAULT_TARGET_WEIGHTS)
    all_weighted_metrics, _ = evaluate(frame, args, DEFAULT_TARGET_WEIGHTS)

    base_error = np.abs(frame[[f"base_{t}" for t in TARGETS]].to_numpy() - frame[list(TARGETS)].to_numpy())
    expert_error = np.abs(frame[[f"expert_{t}" for t in TARGETS]].to_numpy() - frame[list(TARGETS)].to_numpy())
    frame["gain_gap"] = base_error[:, 2] - expert_error[:, 2]
    frame["gain_weighted"] = (base_error - expert_error) @ DEFAULT_TARGET_WEIGHTS
    frame["expert_win_gap"] = frame.gain_gap > 0
    frame["expert_meaningful_win_0.002"] = frame.gain_gap > 0.002
    frame["expert_meaningful_win_0.005"] = frame.gain_gap > 0.005
    frame["downside_gap"] = np.maximum(-frame.gain_gap, 0)
    frame.to_parquet(args.labels_out, index=False)

    budget_delta = random_metrics["bootstrap"]["budget_oracle_minus_fixed_gap"]
    headroom = -float(budget_delta["delta"])
    ci = [-float(budget_delta["ci95"][1]), -float(budget_delta["ci95"][0])]
    if headroom > 0.0015 and ci[0] > 0:
        verdict = "go"
        next_step = "Expand the independent pool and build Router train/validation/sealed splits."
    elif headroom >= 0.0005:
        verdict = "research_only"
        next_step = "Do not expand yet; inspect gain concentration and consider a second 20k probe."
    else:
        verdict = "stop"
        next_step = "Keep fixed routed-v4 and stop archive-r02 data expansion."
    metrics = {
        "experiment": "archive-r02 independent PubChemQC Oracle probe",
        "primary_gate_subset": "10k unbiased random",
        "decision_criterion": "budget-matched Oracle extra Gap improvement >0.0015 eV with CI above zero",
        "random_gap_objective": random_metrics,
        "full_probe_gap_objective": all_metrics,
        "random_weighted_objective": random_weighted_metrics,
        "full_probe_weighted_objective": all_weighted_metrics,
        "decision": {
            "verdict": verdict,
            "budget_oracle_extra_gap_improvement_eV": headroom,
            "ci95_eV": ci,
            "next_step": next_step,
        },
    }
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fixed = random_metrics["methods"]["fixed"]["gap"]["mae"]
    budget = random_metrics["methods"]["budget_oracle"]["gap"]["mae"]
    oracle = random_metrics["methods"]["oracle"]["gap"]["mae"]
    route = random_metrics["routes"]["fixed"]["delta_0"]
    text = f"""# archive-r02 PubChemQC Oracle Probe Decision

Primary gate: 10k label-blind random molecules from the independent 100k pool.
The other 10k descriptor-diverse molecules are diagnostic only.

| subset | n | fixed route | fixed precision | fixed recall | fixed Gap MAE | budget Oracle Gap MAE | unrestricted Oracle Gap MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | {len(random_frame)} | {route['route_fraction']:.1%} | {route['precision']:.1%} | {route['recall']:.1%} | {fixed:.6f} | {budget:.6f} | {oracle:.6f} |

Budget-matched Oracle adds **{headroom:.6f} eV** Gap improvement over fixed v4
(paired-bootstrap 95% CI `{ci[0]:.6f}` to `{ci[1]:.6f}`).

**Decision: {verdict.upper()}.** {next_step}
"""
    args.decision_out.write_text(text, encoding="utf-8")
    print(json.dumps(metrics["decision"], indent=2))


if __name__ == "__main__":
    main()
