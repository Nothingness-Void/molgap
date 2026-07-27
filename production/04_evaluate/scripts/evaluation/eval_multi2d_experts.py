"""Evaluate static ensembles of frozen dual-GPS experts on fixed datasets."""
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
    metric_block,
    predict_dual_gps_experts,
    targetwise_oracle,
)


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def parse_expert(values: list[str]) -> DualGPSExpertPaths:
    name, gps7, gps9, head = values
    return DualGPSExpertPaths(name, Path(gps7), Path(gps9), Path(head))


def parse_ensemble(value: str) -> tuple[str, list[str]]:
    name, separator, members = value.partition("=")
    parsed = [member.strip() for member in members.split(",") if member.strip()]
    if not separator or not name or len(parsed) < 2:
        raise argparse.ArgumentTypeError(
            "Ensembles must use NAME=EXPERT_A,EXPERT_B[,EXPERT_C]"
        )
    return name, parsed


def validate_baseline_name(
    baseline_name: str,
    expert_names: set[str],
    ensemble_names: set[str],
) -> None:
    prediction_names = expert_names | ensemble_names
    if baseline_name not in prediction_names:
        raise ValueError(
            f"Unknown baseline {baseline_name!r}; expected one of {sorted(prediction_names)}"
        )


def evaluate(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    baseline_name: str,
    target_names: tuple[str, ...],
    draws: int,
    seed: int,
) -> dict:
    oracle, selected = targetwise_oracle(y_true, predictions)
    metrics = {
        name: metric_block(y_true, prediction, target_names)
        for name, prediction in predictions.items()
    }
    metrics["targetwise_oracle"] = metric_block(y_true, oracle, target_names)
    deltas = {
        name: delta_block(
            y_true,
            predictions[baseline_name],
            prediction,
            target_names=target_names,
            n_bootstrap=draws,
            seed=seed,
        )
        for name, prediction in predictions.items()
        if name != baseline_name
    }
    oracle_names = list(predictions)
    usage = {
        target: {
            oracle_names[index]: int(np.count_nonzero(selected[:, target_index] == index))
            for index in range(len(oracle_names))
        }
        for target_index, target in enumerate(target_names)
    }
    return {"metrics": metrics, "delta_vs_baseline": deltas, "oracle_usage": usage}


def predict_table(table, experts, ensembles, device, graph_batch_size):
    kept, expert_predictions = predict_dual_gps_experts(
        table.smiles,
        experts,
        device,
        graph_batch_size=graph_batch_size,
    )
    table = table.iloc[kept].reset_index(drop=True)
    predictions = add_mean_ensembles(expert_predictions, ensembles)
    return table, predictions


def write_predictions(table, predictions, path, target_names=TARGETS):
    output = table.copy()
    for name, prediction in predictions.items():
        for index, target in enumerate(target_names):
            output[f"{name}_{target}"] = prediction[:, index]
    atomic_csv(output, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-csv", type=Path, required=True)
    parser.add_argument("--pcqm-csv", type=Path, required=True)
    parser.add_argument("--sealed-csv", type=Path, required=True)
    parser.add_argument(
        "--expert", nargs=4, action="append", metavar=("NAME", "GPS7", "GPS9", "HEAD"), required=True
    )
    parser.add_argument("--ensemble", type=parse_ensemble, action="append", default=[])
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--graph-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/DCU device is required")
    specs = [parse_expert(values) for values in args.expert]
    ensembles = dict(args.ensemble)
    validate_baseline_name(args.baseline_name, {spec.name for spec in specs}, set(ensembles))
    device = torch.device("cuda")
    experts = load_dual_gps_experts(specs, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress = args.out_dir / "progress.json"
    atomic_json({"status": "runtime_ready"}, progress)

    common = pd.read_csv(args.common_csv)
    common, predictions = predict_table(
        common, experts, ensembles, device, args.graph_batch_size
    )
    y_true = common.loc[:, TARGETS].to_numpy(dtype=np.float64)
    common_result = {"n_valid": int(len(common)), "scopes": {}}
    for scope in ("all", "ood1000", "p8_targeted_hard"):
        mask = (
            np.ones(len(common), dtype=bool)
            if scope == "all"
            else common.eval_set.eq(scope).to_numpy()
        )
        common_result["scopes"][scope] = {
            "n": int(mask.sum()),
            **evaluate(
                y_true[mask],
                {name: value[mask] for name, value in predictions.items()},
                args.baseline_name,
                TARGETS,
                args.bootstrap_draws,
                args.seed,
            ),
        }
    atomic_json(common_result, args.out_dir / "common_metrics.json")
    write_predictions(
        common.loc[:, ["eval_set", "cid", "smiles", *TARGETS]],
        predictions,
        args.out_dir / "common_predictions.csv",
    )
    atomic_json({"status": "common_complete", "n_common": len(common)}, progress)

    sealed = pd.read_csv(args.sealed_csv)
    sealed, predictions = predict_table(
        sealed, experts, ensembles, device, args.graph_batch_size
    )
    sealed_true = sealed.loc[:, TARGETS].to_numpy(dtype=np.float64)
    sealed_result = {
        "n_valid": int(len(sealed)),
        **evaluate(
            sealed_true,
            predictions,
            args.baseline_name,
            TARGETS,
            args.bootstrap_draws,
            args.seed,
        ),
    }
    atomic_json(sealed_result, args.out_dir / "sealed_metrics.json")
    sealed_columns = [
        column for column in ("bucket", "cid", "smiles", *TARGETS) if column in sealed
    ]
    write_predictions(
        sealed.loc[:, sealed_columns], predictions, args.out_dir / "sealed_predictions.csv"
    )
    atomic_json(
        {"status": "sealed_complete", "n_common": len(common), "n_sealed": len(sealed)},
        progress,
    )

    pcqm = pd.read_csv(args.pcqm_csv)
    pcqm, predictions = predict_table(
        pcqm, experts, ensembles, device, args.graph_batch_size
    )
    gap_true = pcqm.gap_true.to_numpy(dtype=np.float64)[:, None]
    gap_predictions = {name: value[:, 2:3] for name, value in predictions.items()}
    pcqm_result = {
        "n_valid": int(len(pcqm)),
        **evaluate(
            gap_true,
            gap_predictions,
            args.baseline_name,
            ("gap",),
            args.bootstrap_draws,
            args.seed,
        ),
    }
    atomic_json(pcqm_result, args.out_dir / "pcqm_metrics.json")
    pcqm_columns = [
        column for column in ("pcqm_idx", "idx", "smiles", "gap_true") if column in pcqm
    ]
    write_predictions(
        pcqm.loc[:, pcqm_columns],
        gap_predictions,
        args.out_dir / "pcqm_predictions.csv",
        target_names=("gap",),
    )
    atomic_json(
        {
            "status": "complete",
            "n_common": len(common),
            "n_sealed": len(sealed),
            "n_pcqm": len(pcqm),
        },
        progress,
    )
    print(json.dumps({"common": common_result, "sealed": sealed_result, "pcqm": pcqm_result}, indent=2))


if __name__ == "__main__":
    main()
