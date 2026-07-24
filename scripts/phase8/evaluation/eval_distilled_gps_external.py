"""Evaluate a distilled GPS student without opening any sealed set."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from molgap.multi2d import (
    TARGETS,
    DualGPSExpertPaths,
    add_mean_ensembles,
    delta_block,
    load_dual_gps_experts,
    load_gps_predictors,
    metric_block,
    predict_dual_gps_experts,
    predict_gps_models,
)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def predict_table(table, experts, candidates, device, batch_size):
    kept_experts, expert_predictions = predict_dual_gps_experts(
        table.smiles, experts, device, graph_batch_size=batch_size
    )
    kept_candidates, candidate_predictions = predict_gps_models(
        table.smiles, candidates, device, graph_batch_size=batch_size
    )
    if not np.array_equal(kept_experts, kept_candidates):
        raise RuntimeError("Teacher and candidate retained different rows")
    predictions = add_mean_ensembles(
        expert_predictions, {"teacher": ["control_a", "repair_v2"]}
    )
    predictions.update(candidate_predictions)
    return table.iloc[kept_experts].reset_index(drop=True), predictions


def evaluate(y_true, predictions, targets, draws, seed, candidate_name):
    metrics = {
        name: metric_block(y_true, prediction, targets)
        for name, prediction in predictions.items()
    }
    deltas = {
        baseline: delta_block(
            y_true,
            predictions[baseline],
            predictions[candidate_name],
            target_names=targets,
            n_bootstrap=draws,
            seed=seed,
        )
        for baseline in ("control_a", "teacher")
    }
    return {"metrics": metrics, "student_delta_vs": deltas}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-csv", type=Path, required=True)
    parser.add_argument("--pcqm-csv", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--candidate-name", default="student_w30")
    parser.add_argument("--control-gps7", type=Path, required=True)
    parser.add_argument("--control-gps9", type=Path, required=True)
    parser.add_argument("--control-head", type=Path, required=True)
    parser.add_argument("--repair-gps7", type=Path, required=True)
    parser.add_argument("--repair-gps9", type=Path, required=True)
    parser.add_argument("--repair-head", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--graph-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/DCU device is required")
    device = torch.device("cuda")
    experts = load_dual_gps_experts(
        [
            DualGPSExpertPaths(
                "control_a",
                args.control_gps7,
                args.control_gps9,
                args.control_head,
            ),
            DualGPSExpertPaths(
                "repair_v2",
                args.repair_gps7,
                args.repair_gps9,
                args.repair_head,
            ),
        ],
        device,
    )
    candidates = load_gps_predictors({args.candidate_name: args.student}, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress = args.out_dir / "progress.json"
    atomic_json({"status": "runtime_ready", "sealed_mounted": False}, progress)

    common = pd.read_csv(args.common_csv)
    common, predictions = predict_table(
        common, experts, candidates, device, args.graph_batch_size
    )
    y_common = common.loc[:, TARGETS].to_numpy(np.float64)
    common_result = {"n": len(common), "scopes": {}}
    for scope in ("all", "ood1000", "p8_targeted_hard"):
        mask = (
            np.ones(len(common), dtype=bool)
            if scope == "all"
            else common.eval_set.eq(scope).to_numpy()
        )
        common_result["scopes"][scope] = {
            "n": int(mask.sum()),
            **evaluate(
                y_common[mask],
                {name: value[mask] for name, value in predictions.items()},
                TARGETS,
                args.bootstrap_draws,
                args.seed,
                args.candidate_name,
            ),
        }
    common_output = common.loc[
        :, [column for column in ("eval_set", "cid", "smiles", *TARGETS) if column in common]
    ].copy()
    for name, prediction in predictions.items():
        for index, target in enumerate(TARGETS):
            common_output[f"{name}_{target}"] = prediction[:, index]
    atomic_json(common_result, args.out_dir / "common_metrics.json")
    atomic_csv(common_output, args.out_dir / "common_predictions.csv")
    atomic_json({"status": "common_complete", "n_common": len(common)}, progress)

    pcqm = pd.read_csv(args.pcqm_csv)
    pcqm, predictions = predict_table(
        pcqm, experts, candidates, device, args.graph_batch_size
    )
    y_pcqm = pcqm.gap_true.to_numpy(np.float64)[:, None]
    gap_predictions = {name: value[:, 2:3] for name, value in predictions.items()}
    pcqm_result = {
        "n": len(pcqm),
        **evaluate(
            y_pcqm,
            gap_predictions,
            ("gap",),
            args.bootstrap_draws,
            args.seed + 100,
            args.candidate_name,
        ),
    }
    pcqm_output = pcqm.loc[
        :, [column for column in ("pcqm_idx", "idx", "smiles", "gap_true") if column in pcqm]
    ].copy()
    for name, prediction in gap_predictions.items():
        pcqm_output[f"{name}_gap"] = prediction[:, 0]
    atomic_json(pcqm_result, args.out_dir / "pcqm_metrics.json")
    atomic_csv(pcqm_output, args.out_dir / "pcqm_predictions.csv")

    candidate_vs_teacher = {
        "common_all_average": common_result["scopes"]["all"]["student_delta_vs"]["teacher"]["average"]["delta"],
        "common_all_gap": common_result["scopes"]["all"]["student_delta_vs"]["teacher"]["gap"]["delta"],
        "ood_average": common_result["scopes"]["ood1000"]["student_delta_vs"]["teacher"]["average"]["delta"],
        "ood_gap": common_result["scopes"]["ood1000"]["student_delta_vs"]["teacher"]["gap"]["delta"],
        "p8_hard_average": common_result["scopes"]["p8_targeted_hard"]["student_delta_vs"]["teacher"]["average"]["delta"],
        "p8_hard_gap": common_result["scopes"]["p8_targeted_hard"]["student_delta_vs"]["teacher"]["gap"]["delta"],
        "pcqm_gap": pcqm_result["student_delta_vs"]["teacher"]["gap"]["delta"],
    }
    limits = {
        "common_all_average": 0.001,
        "common_all_gap": 0.001,
        "ood_average": 0.002,
        "ood_gap": 0.002,
        "p8_hard_average": 0.002,
        "p8_hard_gap": 0.002,
        "pcqm_gap": 0.002,
    }
    gate = {
        "pass": all(candidate_vs_teacher[name] <= limit for name, limit in limits.items()),
        "candidate_name": args.candidate_name,
        "candidate_minus_teacher_mae_eV": candidate_vs_teacher,
        "student_minus_teacher_mae_eV": candidate_vs_teacher,
        "maximum_allowed_regression_eV": limits,
        "sealed_opened": False,
    }
    atomic_json(gate, args.out_dir / "gate.json")
    atomic_json(
        {
            "status": "complete",
            "n_common": len(common),
            "n_pcqm": len(pcqm),
            "gate_pass": gate["pass"],
            "sealed_opened": False,
        },
        progress,
    )
    print(json.dumps({"gate": gate, "common": common_result, "pcqm": pcqm_result}, indent=2))


if __name__ == "__main__":
    main()
