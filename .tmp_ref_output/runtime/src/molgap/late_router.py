"""Utilities for conservative late blending between frozen predictors."""
from __future__ import annotations

import hashlib

import numpy as np


TARGET_NAMES = ("HOMO", "LUMO", "Gap")


def metric_block(target: np.ndarray, prediction: np.ndarray) -> dict:
    result = {}
    for column, name in enumerate(TARGET_NAMES):
        error = np.asarray(target[:, column] - prediction[:, column])
        result[name] = {"mae_eV": float(np.abs(error).mean())}
    result["average"] = {
        "mae_eV": float(np.mean([result[name]["mae_eV"] for name in TARGET_NAMES]))
    }
    return result


def build_router_features(
    base: np.ndarray,
    expert: np.ndarray,
    gps7: np.ndarray,
    gps9: np.ndarray,
    geometry: np.ndarray,
) -> np.ndarray:
    """Build compact disagreement and representation-distance features."""
    arrays = [np.asarray(value, dtype=np.float32) for value in (gps7, gps9, geometry)]
    norms = [np.linalg.norm(value, axis=1, keepdims=True) for value in arrays]
    distances = [
        np.linalg.norm(arrays[0] - arrays[1], axis=1, keepdims=True),
        np.linalg.norm(arrays[0] - arrays[2], axis=1, keepdims=True),
        np.linalg.norm(arrays[1] - arrays[2], axis=1, keepdims=True),
    ]
    cosines = []
    for left, right, left_norm, right_norm in (
        (arrays[0], arrays[1], norms[0], norms[1]),
        (arrays[0], arrays[2], norms[0], norms[2]),
        (arrays[1], arrays[2], norms[1], norms[2]),
    ):
        denominator = np.maximum(left_norm * right_norm, 1e-8)
        cosines.append(np.sum(left * right, axis=1, keepdims=True) / denominator)
    disagreement = np.asarray(expert, dtype=np.float32) - np.asarray(base, dtype=np.float32)
    features = np.concatenate(
        [base, expert, disagreement, np.abs(disagreement), *norms, *distances, *cosines],
        axis=1,
    )
    if not np.isfinite(features).all():
        raise ValueError("Router features contain non-finite values")
    return features.astype(np.float32, copy=False)


def optimal_alpha(target: np.ndarray, base: np.ndarray, expert: np.ndarray) -> np.ndarray:
    delta = np.asarray(expert) - np.asarray(base)
    alpha = np.divide(
        np.asarray(target) - np.asarray(base),
        delta,
        out=np.zeros_like(delta, dtype=np.float32),
        where=np.abs(delta) > 1e-8,
    )
    return np.clip(alpha, 0.0, 1.0).astype(np.float32, copy=False)


def blend(base: np.ndarray, expert: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.asarray(base) + np.asarray(alpha) * (np.asarray(expert) - np.asarray(base))


def scaffold_partition(scaffolds: list[str], modulus: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Assign complete scaffold groups to deterministic fit/selection partitions."""
    buckets = np.fromiter(
        (int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % modulus for value in scaffolds),
        dtype=np.int64,
        count=len(scaffolds),
    )
    fit = np.flatnonzero(buckets != 0)
    selection = np.flatnonzero(buckets == 0)
    if min(len(fit), len(selection)) == 0:
        raise ValueError("Scaffold partition produced an empty subset")
    return fit, selection


def grid_alpha(target: np.ndarray, base: np.ndarray, expert: np.ndarray, step: float = 0.02) -> np.ndarray:
    values = np.arange(0.0, 1.0 + step / 2.0, step, dtype=np.float32)
    result = np.zeros((1, target.shape[1]), dtype=np.float32)
    for column in range(target.shape[1]):
        losses = [
            np.abs(target[:, column] - blend(base[:, column], expert[:, column], value)).mean()
            for value in values
        ]
        result[0, column] = values[int(np.argmin(losses))]
    return result


def binned_alpha(
    target: np.ndarray,
    base: np.ndarray,
    expert: np.ndarray,
    edges: tuple[float, ...] = (-np.inf, 3.0, 4.0, 5.0, 6.0, np.inf),
) -> np.ndarray:
    bins = np.digitize(base[:, 2], edges[1:-1], right=False)
    result = np.zeros((len(edges) - 1, target.shape[1]), dtype=np.float32)
    global_alpha = grid_alpha(target, base, expert)[0]
    for index in range(len(result)):
        mask = bins == index
        result[index] = grid_alpha(target[mask], base[mask], expert[mask])[0] if mask.sum() >= 100 else global_alpha
    return result


def apply_binned_alpha(base: np.ndarray, expert: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    bins = np.digitize(base[:, 2], (3.0, 4.0, 5.0, 6.0), right=False)
    return blend(base, expert, alpha[bins])
