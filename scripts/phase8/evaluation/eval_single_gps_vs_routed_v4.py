"""Evaluate one GPS checkpoint against fixed routed-v4 prediction artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader

from molgap.gps import GPSWrapper
from molgap.graphs import smiles_to_2d_pyg


TARGETS = ("homo", "lumo", "gap")


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(value: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    value.to_csv(temporary, index=False)
    os.replace(temporary, path)


def metric_block(truth: np.ndarray, prediction: np.ndarray) -> dict:
    result = {}
    for index, target in enumerate(TARGETS):
        result[target] = {
            "mae_eV": float(np.abs(prediction[:, index] - truth[:, index]).mean()),
            "r2": float(r2_score(truth[:, index], prediction[:, index])),
        }
    result["average"] = {
        "mae_eV": float(np.abs(prediction - truth).mean()),
        "r2": float(np.mean([result[target]["r2"] for target in TARGETS])),
    }
    return result


def paired_delta(
    truth: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict:
    result = {}
    for index, target in enumerate((*TARGETS, "average")):
        if target == "average":
            delta = np.abs(candidate - truth).mean(axis=1) - np.abs(
                baseline - truth
            ).mean(axis=1)
        else:
            delta = np.abs(candidate[:, index] - truth[:, index]) - np.abs(
                baseline[:, index] - truth[:, index]
            )
        standard_error = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        result[target] = {
            "mae_delta_eV": float(delta.mean()),
            "normal_ci95_eV": [
                float(delta.mean() - 1.96 * standard_error),
                float(delta.mean() + 1.96 * standard_error),
            ],
            "candidate_win_rate": float((delta < 0).mean()),
        }
    return result


def predict(
    model: GPSWrapper,
    table: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    graphs, kept = [], []
    for position, smiles in enumerate(table.smiles):
        graph = smiles_to_2d_pyg(smiles)
        if graph is not None:
            graphs.append(graph)
            kept.append(position)
    selected = table.iloc[kept].reset_index(drop=True)
    predictions = []
    with torch.inference_mode():
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                predictions.append(
                    model(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch,
                    ).float().cpu()
                )
    return selected, torch.cat(predictions).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-baseline-csv", type=Path, required=True)
    parser.add_argument("--pcqm-baseline-csv", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--hidden-channels", type=int, default=192)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/DCU device is required")
    device = torch.device("cuda")
    model = GPSWrapper(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_heads=4,
        dropout=0.05,
    ).to(device)
    model.load_state_dict(
        torch.load(args.candidate, map_location=device, weights_only=True)
    )
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress = args.out_dir / "progress.json"
    atomic_json({"status": "runtime_ready"}, progress)

    common = pd.read_csv(args.common_baseline_csv)
    common, candidate = predict(model, common, device, args.batch_size)
    truth = common.loc[:, TARGETS].to_numpy(np.float64)
    baseline = common.loc[
        :, [f"routed_v4_{target}" for target in TARGETS]
    ].to_numpy(np.float64)
    common_metrics = {"n_valid": int(len(common)), "scopes": {}}
    for scope in ("all", "ood1000", "p8_targeted_hard"):
        mask = (
            np.ones(len(common), dtype=bool)
            if scope == "all"
            else common.eval_set.eq(scope).to_numpy()
        )
        common_metrics["scopes"][scope] = {
            "n": int(mask.sum()),
            "routed_v4_500k": metric_block(truth[mask], baseline[mask]),
            args.candidate_name: metric_block(truth[mask], candidate[mask]),
            "candidate_minus_routed_v4_500k": paired_delta(
                truth[mask],
                baseline[mask],
                candidate[mask],
            ),
        }
    common_output = common.loc[
        :, ["eval_set", "cid", "smiles", *TARGETS]
    ].copy()
    for index, target in enumerate(TARGETS):
        common_output[f"routed_v4_500k_{target}"] = baseline[:, index]
        common_output[f"{args.candidate_name}_{target}"] = candidate[:, index]
    atomic_json(common_metrics, args.out_dir / "common_metrics.json")
    atomic_csv(common_output, args.out_dir / "common_predictions.csv")
    atomic_json({"status": "common_complete", "n_common": len(common)}, progress)

    pcqm = pd.read_csv(args.pcqm_baseline_csv)
    pcqm, candidate = predict(model, pcqm, device, args.batch_size)
    gap_truth = pcqm.gap.to_numpy(np.float64)
    baseline_gap = pcqm.routed_v4_gap.to_numpy(np.float64)
    candidate_gap = candidate[:, 2]
    gap_delta = np.abs(candidate_gap - gap_truth) - np.abs(
        baseline_gap - gap_truth
    )
    standard_error = float(gap_delta.std(ddof=1) / np.sqrt(len(gap_delta)))
    pcqm_metrics = {
        "n_valid": int(len(pcqm)),
        "routed_v4_500k_gap_mae_eV": float(
            np.abs(baseline_gap - gap_truth).mean()
        ),
        f"{args.candidate_name}_gap_mae_eV": float(
            np.abs(candidate_gap - gap_truth).mean()
        ),
        "candidate_minus_routed_v4_500k_gap": {
            "mae_delta_eV": float(gap_delta.mean()),
            "normal_ci95_eV": [
                float(gap_delta.mean() - 1.96 * standard_error),
                float(gap_delta.mean() + 1.96 * standard_error),
            ],
            "candidate_win_rate": float((gap_delta < 0).mean()),
        },
    }
    pcqm_output = pcqm.loc[:, ["cid", "smiles", "gap"]].copy()
    pcqm_output["routed_v4_500k_gap"] = baseline_gap
    pcqm_output[f"{args.candidate_name}_gap"] = candidate_gap
    atomic_json(pcqm_metrics, args.out_dir / "pcqm_metrics.json")
    atomic_csv(pcqm_output, args.out_dir / "pcqm_predictions.csv")
    atomic_json(
        {"status": "complete", "n_common": len(common), "n_pcqm": len(pcqm)},
        progress,
    )
    print(
        json.dumps({"common": common_metrics, "pcqm": pcqm_metrics}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
