"""Analyze accepted matched-500K predictions without running model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from molgap.training_reproducibility import atomic_json, sha256_file


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_917


def _load(path: Path) -> dict[str, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ("prediction", "target", "source_idx")
    if any(key not in payload for key in required):
        raise ValueError(f"Incomplete prediction payload: {path}")
    result = {
        key: payload[key].detach().cpu().numpy().reshape(-1)
        for key in required
    }
    if len(result["prediction"]) != 50_000:
        raise ValueError(f"Unexpected row count: {path}")
    if not all(np.isfinite(result[key]).all() for key in ("prediction", "target")):
        raise ValueError(f"Non-finite prediction payload: {path}")
    result["sha256"] = sha256_file(path)
    return result


def _bootstrap(values: np.ndarray) -> dict[str, object]:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    chunk = 100
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        stop = min(start + chunk, BOOTSTRAP_REPLICATES)
        indices = generator.integers(
            0, values.size, size=(stop - start, values.size)
        )
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "mean_eV": float(values.mean()),
        "bootstrap_95pct_eV": [
            float(value) for value in np.quantile(means, [0.025, 0.975])
        ],
        "probability_mean_ge_0p003": float((means >= 0.003).mean()),
    }


def _weighted_median_weight(
    first: np.ndarray, second: np.ndarray, target: np.ndarray
) -> float:
    delta = first - second
    mask = delta != 0
    ratios = (target[mask] - second[mask]) / delta[mask]
    weights = np.abs(delta[mask])
    order = np.argsort(ratios)
    ratios, weights = ratios[order], weights[order]
    index = int(np.searchsorted(np.cumsum(weights), weights.sum() / 2, side="left"))
    return float(np.clip(ratios[min(index, len(ratios) - 1)], 0.0, 1.0))


def _crossfit_pair(
    first: np.ndarray,
    second: np.ndarray,
    target: np.ndarray,
    source_idx: np.ndarray,
) -> dict[str, object]:
    prediction = np.empty_like(target, dtype=np.float64)
    weights = []
    folds = source_idx % 5
    for fold in range(5):
        train = folds != fold
        held_out = ~train
        weight = _weighted_median_weight(first[train], second[train], target[train])
        weights.append(weight)
        prediction[held_out] = (
            weight * first[held_out] + (1.0 - weight) * second[held_out]
        )
    return {
        "fold_weights_first": weights,
        "mean_weight_first": float(np.mean(weights)),
        "oof_mae_eV": float(np.abs(prediction - target).mean()),
    }


def _pair_report(
    first_name: str,
    second_name: str,
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    source_idx: np.ndarray,
) -> dict[str, object]:
    first, second = predictions[first_name], predictions[second_name]
    first_error = np.abs(first - target)
    second_error = np.abs(second - target)
    fixed = 0.5 * (first + second)
    return {
        "first": first_name,
        "second": second_name,
        "signed_residual_correlation": float(
            np.corrcoef(first - target, second - target)[0, 1]
        ),
        "absolute_residual_correlation": float(
            np.corrcoef(first_error, second_error)[0, 1]
        ),
        "first_win_rate": float((first_error < second_error).mean()),
        "fixed_50_50_mae_eV": float(np.abs(fixed - target).mean()),
        "oracle_mae_eV": float(np.minimum(first_error, second_error).mean()),
        "crossfit": _crossfit_pair(first, second, target, source_idx),
    }


def _crossfit_hierarchical(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    target: np.ndarray,
    source_idx: np.ndarray,
) -> dict[str, object]:
    prediction = np.empty_like(target, dtype=np.float64)
    first_weights, base_weights = [], []
    folds = source_idx % 5
    for fold in range(5):
        train = folds != fold
        held_out = ~train
        first_weight = _weighted_median_weight(
            first[train], second[train], target[train]
        )
        base_train = first_weight * first[train] + (1.0 - first_weight) * second[train]
        base_held_out = (
            first_weight * first[held_out]
            + (1.0 - first_weight) * second[held_out]
        )
        base_weight = _weighted_median_weight(
            base_train, third[train], target[train]
        )
        prediction[held_out] = (
            base_weight * base_held_out + (1.0 - base_weight) * third[held_out]
        )
        first_weights.append(first_weight)
        base_weights.append(base_weight)
    return {
        "fold_weights_first_within_base": first_weights,
        "fold_weights_base_vs_third": base_weights,
        "mean_weight_first_within_base": float(np.mean(first_weights)),
        "mean_weight_base_vs_third": float(np.mean(base_weights)),
        "oof_mae_eV": float(np.abs(prediction - target).mean()),
    }


def analyze(paths: dict[str, Path]) -> dict[str, object]:
    payloads = {name: _load(path) for name, path in paths.items()}
    reference = payloads["edge_state"]
    for name, payload in payloads.items():
        if not np.array_equal(payload["source_idx"], reference["source_idx"]):
            raise ValueError(f"source_idx mismatch: {name}")
        if not np.array_equal(payload["target"], reference["target"]):
            raise ValueError(f"target mismatch: {name}")

    target = reference["target"].astype(np.float64)
    source_idx = reference["source_idx"].astype(np.int64)
    predictions = {
        name: payload["prediction"].astype(np.float64)
        for name, payload in payloads.items()
    }
    errors = {
        name: np.abs(prediction - target)
        for name, prediction in predictions.items()
    }
    mae = {name: float(error.mean()) for name, error in errors.items()}

    comparisons = {}
    for reference_name, candidate_name in (
        ("edge_state", "edge_local_only"),
        ("edge_state", "edge_sparse_global_369"),
        ("edge_sparse_global_369", "edge_local_only"),
        ("edge_local_only", "k1"),
        ("edge_sparse_global_369", "k1"),
        ("edge_local_only", "gptrans_t"),
        ("gptrans_t", "k1"),
    ):
        key = f"{candidate_name}_over_{reference_name}"
        comparisons[key] = _bootstrap(
            errors[reference_name] - errors[candidate_name]
        )

    pairs = {}
    for first, second in (
        ("k1", "gptrans_t"),
        ("k1", "edge_local_only"),
        ("gptrans_t", "edge_local_only"),
        ("k1", "edge_sparse_global_369"),
        ("gptrans_t", "edge_sparse_global_369"),
        ("edge_local_only", "edge_sparse_global_369"),
    ):
        pairs[f"{first}__{second}"] = _pair_report(
            first, second, predictions, target, source_idx
        )

    hierarchical = {}
    for third in ("edge_local_only", "edge_sparse_global_369"):
        hierarchical[f"k1__gptrans_t__{third}"] = _crossfit_hierarchical(
            predictions["k1"],
            predictions["gptrans_t"],
            predictions[third],
            target,
            source_idx,
        )

    return {
        "format": "molgap-pcqm-500k-v4-local-ablation-analysis-v1",
        "benchmark_id": "pcqm-fixed500k-dev50k-matched60-v4",
        "rows": int(len(target)),
        "source_idx_and_target_alignment": True,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "prediction_sha256": {
            name: payload["sha256"] for name, payload in payloads.items()
        },
        "mae_eV": mae,
        "paired_improvements": comparisons,
        "pair_residual_and_blend": pairs,
        "hierarchical_crossfit": hierarchical,
        "causal_decomposition_eV": {
            "remove_every_layer_dense_global": mae["edge_state"]
            - mae["edge_local_only"],
            "add_sparse_dense_global_369": mae["edge_local_only"]
            - mae["edge_sparse_global_369"],
            "replace_no_global_with_k1_slot_369": mae["edge_local_only"]
            - mae["k1"],
            "replace_sparse_dense_global_with_k1_slot_369": mae[
                "edge_sparse_global_369"
            ]
            - mae["k1"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-state", type=Path, required=True)
    parser.add_argument("--k1", type=Path, required=True)
    parser.add_argument("--gptrans-t", type=Path, required=True)
    parser.add_argument("--edge-local-only", type=Path, required=True)
    parser.add_argument("--edge-sparse-global-369", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        {
            "edge_state": args.edge_state,
            "k1": args.k1,
            "gptrans_t": args.gptrans_t,
            "edge_local_only": args.edge_local_only,
            "edge_sparse_global_369": args.edge_sparse_global_369,
        }
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
