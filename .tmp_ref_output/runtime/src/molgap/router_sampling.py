"""Leakage-safe sampling helpers for Router development datasets."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from .utils import scaffold_split_key


def compute_scaffold_keys(smiles: Sequence[str], workers: int = 8) -> np.ndarray:
    """Compute split keys in parallel while preserving input order."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(scaffold_split_key, smiles, chunksize=500))
    return np.asarray(values, dtype=object)


def select_descriptor_diverse(
    frame: pd.DataFrame,
    candidate_indices: np.ndarray,
    *,
    features: Sequence[str],
    n_select: int,
    n_clusters: int,
    seed: int,
) -> tuple[np.ndarray, dict[int, float]]:
    """Sample approximately equal counts per descriptor cluster."""
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if n_select > len(candidate_indices):
        raise ValueError(f"Cannot select {n_select} from {len(candidate_indices)} rows")
    matrix = frame.loc[candidate_indices, list(features)].to_numpy(dtype=np.float64)
    medians = np.nanmedian(matrix, axis=0)
    matrix = np.where(np.isfinite(matrix), matrix, medians)
    scaled = StandardScaler().fit_transform(matrix)
    n_clusters = min(n_clusters, n_select, len(candidate_indices))
    labels = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        n_init=3,
        random_state=seed,
    ).fit_predict(scaled)

    rng = np.random.default_rng(seed)
    base_quota, extra = divmod(n_select, n_clusters)
    selected: list[int] = []
    probabilities: dict[int, float] = {}
    for rank, cluster in enumerate(rng.permutation(n_clusters)):
        members = candidate_indices[labels == cluster]
        quota = min(len(members), base_quota + int(rank < extra))
        if not quota:
            continue
        choices = rng.choice(members, size=quota, replace=False)
        selected.extend(int(value) for value in choices)
        probabilities.update({int(value): quota / len(members) for value in choices})

    if len(selected) < n_select:
        selected_set = set(selected)
        available = np.asarray(
            [value for value in candidate_indices if int(value) not in selected_set],
            dtype=np.int64,
        )
        missing = n_select - len(selected)
        fill = rng.choice(available, size=missing, replace=False)
        selected.extend(int(value) for value in fill)
        probabilities.update({int(value): missing / len(available) for value in fill})
    return np.asarray(selected[:n_select], dtype=np.int64), probabilities
