"""Evaluate frozen prediction-combination rules across PCQM development scales."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


MATRIX_SHA256 = "c52e944def35d8d9ccb41011c9cf820182702c1a9006189b6fc488397169edbb"
K1_500K_SHA256 = "e43728a30a74a19e9e969cef3e94f3c0f9718a1c606bfa5306bd9b3c87ca8d31"
PAIR_500K_SHA256 = "35fdaa76f10e166ccb123b6005929a420f18b2678a286023d4aa62e48ec9ca01"
K1_NAME = "neural_atom_k1_v4"
PAIR_NAME = "neural_atom_k1_pair_token"
ALPHA_GRID = np.linspace(0.0, 1.0, 41)
BIN_WEIGHTS = np.array([0.0, 0.5, 1.0])
FOLD_COUNT = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes((json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    os.replace(temporary, path)


def _arrays(source, target, base, expert, expected_start: int):
    source = np.asarray(source, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    base = np.asarray(base, dtype=np.float64).reshape(-1)
    expert = np.asarray(expert, dtype=np.float64).reshape(-1)
    if any(len(part) != 50_000 for part in (source, target, base, expert)):
        raise ValueError("Each prediction role must contain exactly 50,000 rows")
    if not np.array_equal(source, np.arange(expected_start, expected_start + 50_000)):
        raise ValueError("Prediction rows do not match the frozen source range/order")
    if not all(np.isfinite(part).all() for part in (target, base, expert)):
        raise ValueError("Prediction or target contains non-finite values")
    return source, target, base, expert


def _load_100k(path: Path):
    if _sha256(path) != MATRIX_SHA256:
        raise ValueError("100K aligned prediction matrix SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    ids = payload["model_ids"]
    if ids.count(K1_NAME) != 1 or ids.count(PAIR_NAME) != 1:
        raise ValueError("Expected exactly one K1 and PairToken prediction column")
    predictions = payload["predictions_eV"]
    return _arrays(
        payload["source_idx"].numpy(),
        payload["target_eV"].numpy(),
        predictions[:, ids.index(K1_NAME)].numpy(),
        predictions[:, ids.index(PAIR_NAME)].numpy(),
        100_000,
    )


def _load_500k(k1_path: Path, pair_path: Path):
    if _sha256(k1_path) != K1_500K_SHA256:
        raise ValueError("500K K1 prediction payload SHA-256 mismatch")
    if _sha256(pair_path) != PAIR_500K_SHA256:
        raise ValueError("500K PairToken prediction payload SHA-256 mismatch")
    k1 = torch.load(k1_path, map_location="cpu", weights_only=True)
    pair = torch.load(pair_path, map_location="cpu", weights_only=True)
    for payload in (k1, pair):
        if not {"prediction", "target", "source_idx"}.issubset(payload):
            raise ValueError("500K prediction payload lacks required fields")
    if not torch.equal(k1["source_idx"], pair["source_idx"]):
        raise ValueError("500K source-index vectors differ")
    if not torch.equal(k1["target"], pair["target"]):
        raise ValueError("500K target tensors differ")
    return _arrays(
        k1["source_idx"].view(-1).numpy(),
        k1["target"].view(-1).numpy(),
        k1["prediction"].view(-1).numpy(),
        pair["prediction"].view(-1).numpy(),
        500_000,
    )


def _mae(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.abs(target - prediction).mean())


def _fuse(base: np.ndarray, expert: np.ndarray, weight) -> np.ndarray:
    return weight * base + (1.0 - weight) * expert


def fit_global_weight(target: np.ndarray, base: np.ndarray, expert: np.ndarray) -> float:
    errors = np.abs(target[:, None] - _fuse(base[:, None], expert[:, None], ALPHA_GRID))
    scores = errors.mean(axis=0)
    winners = np.flatnonzero(scores <= scores.min() + 1e-12)
    return float(ALPHA_GRID[winners[np.argmin(np.abs(ALPHA_GRID[winners] - 0.5))]])


def fit_disagreement_bins(
    target: np.ndarray, base: np.ndarray, expert: np.ndarray, global_weight: float
) -> dict:
    disagreement = np.abs(base - expert)
    thresholds = np.quantile(disagreement, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(disagreement, thresholds, right=True)
    weights: list[float] = []
    counts: list[int] = []
    diagnostics: list[dict] = []
    for bin_id in range(5):
        mask = bins == bin_id
        counts.append(int(mask.sum()))
        weight = global_weight
        selected = None
        advantage = None
        if mask.sum() >= 5_000:
            errors = np.abs(
                target[mask, None]
                - _fuse(base[mask, None], expert[mask, None], BIN_WEIGHTS)
            ).mean(axis=0)
            winners = np.flatnonzero(errors <= errors.min() + 1e-12)
            selected = float(BIN_WEIGHTS[winners[np.argmin(np.abs(BIN_WEIGHTS[winners] - 0.5))]])
            global_error = _mae(target[mask], _fuse(base[mask], expert[mask], global_weight))
            advantage = global_error - float(errors.min())
            if advantage >= 0.001:
                weight = selected
        weights.append(weight)
        diagnostics.append(
            {
                "bin": bin_id,
                "best_discrete_weight": selected,
                "fitting_gain_over_global_eV": advantage,
                "rule_changed": weight != global_weight,
            }
        )
    return {
        "thresholds_eV": [float(value) for value in thresholds],
        "weights_on_k1": weights,
        "fitting_rows_per_bin": counts,
        "fitting_diagnostics": diagnostics,
    }


def apply_disagreement_bins(base: np.ndarray, expert: np.ndarray, rule: dict):
    bins = np.digitize(np.abs(base - expert), rule["thresholds_eV"], right=True)
    weights = np.asarray(rule["weights_on_k1"])[bins]
    return _fuse(base, expert, weights), np.bincount(bins, minlength=5).tolist()


def _bootstrap_gain(gain: np.ndarray, *, seed: int = 42, repetitions: int = 5000) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for offset in range(0, repetitions, 100):
        count = min(100, repetitions - offset)
        indices = rng.integers(0, len(gain), size=(count, len(gain)))
        means[offset : offset + count] = gain[indices].mean(axis=1)
    return {
        "n_replicates": repetitions,
        "seed": seed,
        "gain_ci99_eV": [float(value) for value in np.quantile(means, [0.005, 0.995])],
    }


def _crossfit(source: np.ndarray, target: np.ndarray, base: np.ndarray, expert: np.ndarray):
    global_prediction = np.empty_like(base)
    bin_prediction = np.empty_like(base)
    global_weights: list[float] = []
    for fold in range(FOLD_COUNT):
        training = source % FOLD_COUNT != fold
        held_out = ~training
        weight = fit_global_weight(target[training], base[training], expert[training])
        rule = fit_disagreement_bins(target[training], base[training], expert[training], weight)
        global_weights.append(weight)
        global_prediction[held_out] = _fuse(base[held_out], expert[held_out], weight)
        bin_prediction[held_out], _ = apply_disagreement_bins(base[held_out], expert[held_out], rule)
    return {
        "source_fold_rule": "source_idx % 5",
        "global_weight_by_fold": global_weights,
        "equal_mae_eV": _mae(target, _fuse(base, expert, 0.5)),
        "global_weight_oof_mae_eV": _mae(target, global_prediction),
        "disagreement_bin_oof_mae_eV": _mae(target, bin_prediction),
    }


def _evaluate(
    source: np.ndarray,
    target: np.ndarray,
    benchmark: np.ndarray,
    prediction: np.ndarray,
    simpler_prediction: np.ndarray | None,
    *,
    seed: int,
) -> dict:
    gain = np.abs(benchmark - target) - np.abs(prediction - target)
    fold_gains = [float(gain[source % FOLD_COUNT == fold].mean()) for fold in range(FOLD_COUNT)]
    result = {
        "mae_eV": _mae(target, prediction),
        "gain_vs_stronger_single_eV": float(gain.mean()),
        "fold_gain_eV": fold_gains,
        "paired_row_bootstrap": _bootstrap_gain(gain, seed=seed),
    }
    result["gain_vs_simpler_rule_eV"] = (
        None
        if simpler_prediction is None
        else _mae(target, simpler_prediction) - result["mae_eV"]
    )
    result["exploratory_gate_pass"] = bool(
        result["gain_vs_stronger_single_eV"] >= 0.001
        and result["paired_row_bootstrap"]["gain_ci99_eV"][0] > 0
        and min(fold_gains) > 0
        and (
            simpler_prediction is None or result["gain_vs_simpler_rule_eV"] >= 0.001
        )
    )
    return result


def run_transfer(matrix_path: Path, k1_path: Path, pair_path: Path, output: Path) -> dict:
    started = time.perf_counter()
    source100, target100, base100, pair100 = _load_100k(matrix_path)
    source500, target500, base500, pair500 = _load_500k(k1_path, pair_path)

    crossfit = _crossfit(source100, target100, base100, pair100)
    global_weight = fit_global_weight(target100, base100, pair100)
    bin_rule = fit_disagreement_bins(target100, base100, pair100, global_weight)
    equal500 = _fuse(base500, pair500, 0.5)
    global500 = _fuse(base500, pair500, global_weight)
    bins500, bin_counts500 = apply_disagreement_bins(base500, pair500, bin_rule)
    k1_mae = _mae(target500, base500)
    pair_mae = _mae(target500, pair500)
    benchmark = base500 if k1_mae <= pair_mae else pair500

    rounds = {
        "R1_fixed_equal": _evaluate(source500, target500, benchmark, equal500, None, seed=101),
        "R2_global_weight": _evaluate(source500, target500, benchmark, global500, equal500, seed=102),
        "R3_disagreement_bins": _evaluate(source500, target500, benchmark, bins500, global500, seed=103),
    }
    rounds["R2_global_weight"]["k1_weight_fitted_on_100k"] = global_weight
    rounds["R3_disagreement_bins"]["100k_fitted_rule"] = bin_rule
    rounds["R3_disagreement_bins"]["500k_rows_per_bin"] = bin_counts500
    result = {
        "format": "molgap-pcqm-specialist-combination-transfer-v1",
        "complete": True,
        "training_executed": False,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "roles": {
            "fitting": "official-train-derived-development-100k-50k",
            "transfer_evaluation": "official-train-derived-development-500k-50k",
            "transfer_role_already_used_for_checkpoint_selection": True,
        },
        "identity": {
            "100k_source_idx_min_max": [int(source100.min()), int(source100.max())],
            "500k_source_idx_min_max": [int(source500.min()), int(source500.max())],
            "rows_at_each_scale": 50_000,
            "100k_matrix_sha256": _sha256(matrix_path),
            "500k_k1_payload_sha256": _sha256(k1_path),
            "500k_pairtoken_payload_sha256": _sha256(pair_path),
        },
        "100k": {
            "k1_mae_eV": _mae(target100, base100),
            "pairtoken_mae_eV": _mae(target100, pair100),
            "crossfit": crossfit,
        },
        "500k": {
            "k1_mae_eV": k1_mae,
            "pairtoken_mae_eV": pair_mae,
            "stronger_single_model": "K1" if k1_mae <= pair_mae else "PairToken",
            "rounds": rounds,
        },
        "native_cost": {
            "cpu_wall_seconds": float(time.perf_counter() - started),
            "gpu_device_hours": 0.0,
        },
    }
    _atomic_json(output, result)
    return result
