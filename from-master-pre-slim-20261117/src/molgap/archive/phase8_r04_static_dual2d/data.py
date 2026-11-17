"""Dataset selection and scaffold splitting for dual-2D static candidate pilots."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from molgap.router_sampling import compute_scaffold_keys, select_descriptor_diverse


PILOT_COUNTS = {
    "random": 18_000,
    "descriptor_diverse": 6_000,
    "narrow_conjugated": 2_400,
    "flexible": 1_800,
    "large_heteroatom": 1_800,
}


def _sample(frame: pd.DataFrame, mask, n: int, rng, excluded: set[int]) -> np.ndarray:
    eligible = frame.index[np.asarray(mask) & ~frame.source_idx.isin(excluded)].to_numpy()
    if len(eligible) < n:
        raise ValueError(f"Requested {n:,} rows from only {len(eligible):,} eligible")
    return rng.choice(eligible, n, replace=False)


def select_balanced_pilot(
    expansion: pd.DataFrame,
    topup: pd.DataFrame,
    *,
    counts: dict[str, int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Select mutually exclusive dual-2D static candidate source slices from the cached 500k pool."""
    counts = counts or PILOT_COUNTS
    rng = np.random.default_rng(seed)
    expansion = expansion.copy().reset_index(drop=True)
    expansion["source_idx"] = np.arange(len(expansion), dtype=np.int64)
    lookup = expansion[["canonical_smiles", "source_idx"]].drop_duplicates("canonical_smiles")
    enriched = (
        topup.drop_duplicates("canonical_smiles")
        .merge(lookup, on="canonical_smiles", how="inner", validate="one_to_one")
        .reset_index(drop=True)
    )
    selected: dict[str, np.ndarray] = {}
    used: set[int] = set()

    narrow_mask = (enriched.gap < 3.5) & (
        (enriched.aromatic_rings >= 3) | (enriched.aromatic_atom_fraction >= 0.80)
    )
    selected["narrow_conjugated"] = _sample(
        enriched, narrow_mask, counts["narrow_conjugated"], rng, used
    )
    used.update(enriched.loc[selected["narrow_conjugated"], "source_idx"].astype(int))

    flexible_mask = enriched.rotatable_bonds >= 8
    selected["flexible"] = _sample(
        enriched, flexible_mask, counts["flexible"], rng, used
    )
    used.update(enriched.loc[selected["flexible"], "source_idx"].astype(int))

    large_mask = (enriched.mw >= 500) | (
        (enriched.heavy_atoms >= 35) & ((enriched.has_s > 0) | (enriched.has_cl > 0))
    )
    selected["large_heteroatom"] = _sample(
        enriched, large_mask, counts["large_heteroatom"], rng, used
    )
    used.update(enriched.loc[selected["large_heteroatom"], "source_idx"].astype(int))

    diverse_candidates = enriched.index[~enriched.source_idx.isin(used)].to_numpy()
    diverse_features = [
        "mw", "heavy_atoms", "ring_count", "aromatic_rings",
        "aromatic_atom_fraction", "rotatable_bonds", "conjugated_bonds",
        "has_s", "has_cl", "has_f", "gap",
    ]
    diverse, _ = select_descriptor_diverse(
        enriched,
        diverse_candidates,
        features=diverse_features,
        n_select=counts["descriptor_diverse"],
        n_clusters=256,
        seed=seed,
    )
    selected["descriptor_diverse"] = diverse
    used.update(enriched.loc[diverse, "source_idx"].astype(int))

    random_candidates = expansion.index[~expansion.source_idx.isin(used)].to_numpy()
    random_rows = rng.choice(random_candidates, counts["random"], replace=False)

    frames = []
    for source, indices in selected.items():
        source_indices = enriched.loc[indices, "source_idx"].to_numpy(dtype=np.int64)
        part = expansion.iloc[source_indices].copy()
        part["sampling_source"] = source
        frames.append(part)
    random_part = expansion.iloc[random_rows].copy()
    random_part["sampling_source"] = "random"
    frames.append(random_part)
    result = pd.concat(frames, ignore_index=True)
    if result.source_idx.duplicated().any() or len(result) != sum(counts.values()):
        raise AssertionError("Pilot selection is not unique or has the wrong size")
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def scaffold_split_balanced(
    frame: pd.DataFrame,
    *,
    fractions: Sequence[float] = (0.90, 0.05, 0.05),
    names: Sequence[str] = ("train", "validation", "internal_test"),
    workers: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Greedily allocate whole scaffolds while balancing sampling sources."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to one")
    result = frame.copy()
    result["scaffold"] = compute_scaffold_keys(
        result.canonical_smiles.tolist(), workers=workers
    )
    sources = sorted(result.sampling_source.unique())
    counts = pd.crosstab(result.scaffold, result.sampling_source).reindex(
        columns=sources, fill_value=0
    )
    sizes = counts.sum(axis=1)
    rng = np.random.default_rng(seed)
    order = sorted(counts.index, key=lambda key: (-sizes[key], rng.random()))
    target = np.outer(np.asarray(fractions), counts.sum(axis=0).to_numpy(dtype=float))
    assigned = np.zeros_like(target)
    allocation: dict[str, str] = {}
    for scaffold in order:
        vector = counts.loc[scaffold].to_numpy(dtype=float)
        scores = []
        for split_index in range(len(names)):
            candidate = assigned.copy()
            candidate[split_index] += vector
            normalized = (candidate - target) / np.maximum(target, 1.0)
            overfill = np.maximum(candidate - target, 0.0) / np.maximum(target, 1.0)
            scores.append(float(np.square(normalized).sum() + 4.0 * np.square(overfill).sum()))
        choice = int(np.argmin(scores))
        assigned[choice] += vector
        allocation[scaffold] = names[choice]
    result["split"] = result.scaffold.map(allocation)
    return result
