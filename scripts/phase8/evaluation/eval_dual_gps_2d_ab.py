"""Paired external evaluation for two frozen dual-GPS 2D models."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

from molgap.fusion import DualGPSFusionHead
from molgap.gps import GPSWrapper
from molgap.graphs import smiles_to_2d_pyg


TARGETS = ("homo", "lumo", "gap")


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


def load_model(model: torch.nn.Module, path: Path, device: torch.device) -> torch.nn.Module:
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def metric_block(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    result = {}
    for index, target in enumerate(TARGETS):
        result[target] = {
            "mae_eV": float(np.abs(prediction[:, index] - y_true[:, index]).mean()),
            "r2": float(r2_score(y_true[:, index], prediction[:, index])),
        }
    result["average"] = {
        "mae_eV": float(np.abs(prediction - y_true).mean()),
        "r2": float(np.mean([result[target]["r2"] for target in TARGETS])),
    }
    return result


def bootstrap_delta(
    y_true: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed: int,
    draws: int,
) -> dict:
    rng = np.random.default_rng(seed)
    result = {}
    for index, target in enumerate((*TARGETS, "average")):
        if target == "average":
            delta = np.abs(candidate - y_true).mean(axis=1) - np.abs(baseline - y_true).mean(axis=1)
        else:
            delta = np.abs(candidate[:, index] - y_true[:, index]) - np.abs(baseline[:, index] - y_true[:, index])
        means = np.empty(draws, dtype=np.float64)
        for draw in range(draws):
            sample = rng.integers(0, len(delta), len(delta))
            means[draw] = delta[sample].mean()
        result[target] = {
            "mae_delta_eV": float(delta.mean()),
            "ci95_eV": [float(value) for value in np.quantile(means, [0.025, 0.975])],
            "p_candidate_better": float((means < 0).mean()),
        }
    return result


def predict(
    table: pd.DataFrame,
    models: tuple[tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module], ...],
    device: torch.device,
    graph_batch_size: int,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    graphs, kept = [], []
    for position, smiles in enumerate(table.smiles):
        graph = smiles_to_2d_pyg(smiles)
        if graph is not None:
            graphs.append(graph)
            kept.append(position)
    table = table.iloc[kept].reset_index(drop=True)
    if not graphs:
        raise RuntimeError("No valid 2D graphs were constructed")

    encoded7 = [[] for _ in models]
    encoded9 = [[] for _ in models]
    with torch.inference_mode():
        for batch in GeometricDataLoader(graphs, batch_size=graph_batch_size, shuffle=False):
            batch = batch.to(device)
            for index, (gps7, gps9, _) in enumerate(models):
                encoded7[index].append(gps7.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch).float().cpu())
                encoded9[index].append(gps9.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch).float().cpu())

    predictions = []
    with torch.inference_mode():
        for index, (_, _, head) in enumerate(models):
            h7, h9 = torch.cat(encoded7[index]), torch.cat(encoded9[index])
            chunks = []
            for batch7, batch9 in TorchDataLoader(TensorDataset(h7, h9), batch_size=4096, shuffle=False):
                chunks.append(head(batch7.to(device), batch9.to(device)).float().cpu())
            predictions.append(torch.cat(chunks).numpy())
    return table, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-csv", type=Path, required=True)
    parser.add_argument("--pcqm-csv", type=Path, required=True)
    parser.add_argument("--sealed-csv", type=Path)
    parser.add_argument("--baseline-gps7", type=Path, required=True)
    parser.add_argument("--baseline-gps9", type=Path, required=True)
    parser.add_argument("--baseline-head", type=Path, required=True)
    parser.add_argument("--candidate-gps7", type=Path, required=True)
    parser.add_argument("--candidate-gps9", type=Path, required=True)
    parser.add_argument("--candidate-head", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
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
    gps_args = dict(hidden_channels=192, num_heads=4, dropout=0.05)
    models = (
        (
            load_model(GPSWrapper(num_layers=7, **gps_args), args.baseline_gps7, device),
            load_model(GPSWrapper(num_layers=9, **gps_args), args.baseline_gps9, device),
            load_model(DualGPSFusionHead(hidden=192), args.baseline_head, device),
        ),
        (
            load_model(GPSWrapper(num_layers=7, **gps_args), args.candidate_gps7, device),
            load_model(GPSWrapper(num_layers=9, **gps_args), args.candidate_gps9, device),
            load_model(DualGPSFusionHead(hidden=192), args.candidate_head, device),
        ),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "progress.json"
    atomic_json({"status": "runtime_ready"}, progress_path)

    common = pd.read_csv(args.common_csv)
    common, (baseline, candidate) = predict(common, models, device, args.graph_batch_size)
    y_true = common.loc[:, TARGETS].to_numpy(dtype=np.float64)
    metrics = {"n_valid": int(len(common)), "scopes": {}}
    for scope in ("all", "ood1000", "p8_targeted_hard"):
        mask = np.ones(len(common), dtype=bool) if scope == "all" else common.eval_set.eq(scope).to_numpy()
        metrics["scopes"][scope] = {
            "n": int(mask.sum()),
            args.baseline_name: metric_block(y_true[mask], baseline[mask]),
            args.candidate_name: metric_block(y_true[mask], candidate[mask]),
            "candidate_minus_baseline": bootstrap_delta(
                y_true[mask], baseline[mask], candidate[mask], args.seed, args.bootstrap_draws
            ),
        }
    output = common.loc[:, ["eval_set", "cid", "smiles", *TARGETS]].copy()
    for index, target in enumerate(TARGETS):
        output[f"{args.baseline_name}_{target}"] = baseline[:, index]
        output[f"{args.candidate_name}_{target}"] = candidate[:, index]
    atomic_json(metrics, args.out_dir / "common_metrics.json")
    atomic_csv(output, args.out_dir / "common_predictions.csv")
    atomic_json({"status": "common_complete", "n_common": int(len(common))}, progress_path)

    sealed_n = 0
    if args.sealed_csv is not None:
        sealed = pd.read_csv(args.sealed_csv)
        sealed, (baseline, candidate) = predict(sealed, models, device, args.graph_batch_size)
        sealed_true = sealed.loc[:, TARGETS].to_numpy(dtype=np.float64)
        sealed_metrics = {
            "n_valid": int(len(sealed)),
            args.baseline_name: metric_block(sealed_true, baseline),
            args.candidate_name: metric_block(sealed_true, candidate),
            "candidate_minus_baseline": bootstrap_delta(
                sealed_true, baseline, candidate, args.seed, args.bootstrap_draws
            ),
        }
        sealed_output = sealed.loc[:, [column for column in ("bucket", "cid", "smiles", *TARGETS) if column in sealed]].copy()
        for index, target in enumerate(TARGETS):
            sealed_output[f"{args.baseline_name}_{target}"] = baseline[:, index]
            sealed_output[f"{args.candidate_name}_{target}"] = candidate[:, index]
        atomic_json(sealed_metrics, args.out_dir / "sealed_metrics.json")
        atomic_csv(sealed_output, args.out_dir / "sealed_predictions.csv")
        sealed_n = len(sealed)
        atomic_json(
            {"status": "sealed_complete", "n_common": int(len(common)), "n_sealed": sealed_n},
            progress_path,
        )

    pcqm = pd.read_csv(args.pcqm_csv)
    pcqm, (baseline, candidate) = predict(pcqm, models, device, args.graph_batch_size)
    gap_true = pcqm.gap_true.to_numpy(dtype=np.float64)
    y_gap = np.column_stack((gap_true, gap_true, gap_true))
    pcqm_metrics = {
        "n_valid": int(len(pcqm)),
        args.baseline_name: {"gap": metric_block(y_gap, baseline)["gap"]},
        args.candidate_name: {"gap": metric_block(y_gap, candidate)["gap"]},
        "candidate_minus_baseline": {
            "gap": bootstrap_delta(y_gap, baseline, candidate, args.seed, args.bootstrap_draws)["gap"]
        },
    }
    pcqm_output = pcqm.loc[:, [column for column in ("pcqm_idx", "idx", "smiles", "gap_true") if column in pcqm]].copy()
    pcqm_output[f"{args.baseline_name}_gap"] = baseline[:, 2]
    pcqm_output[f"{args.candidate_name}_gap"] = candidate[:, 2]
    atomic_json(pcqm_metrics, args.out_dir / "pcqm_metrics.json")
    atomic_csv(pcqm_output, args.out_dir / "pcqm_predictions.csv")
    atomic_json(
        {"status": "complete", "n_common": int(len(common)), "n_sealed": sealed_n, "n_pcqm": int(len(pcqm))},
        progress_path,
    )
    print(json.dumps({"common": metrics, "pcqm": pcqm_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
