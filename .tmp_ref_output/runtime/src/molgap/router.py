"""Reusable statistics and cheap molecular features for learned routing."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .utils import safe_mol


DEFAULT_TARGET_WEIGHTS = np.array([0.25, 0.25, 0.50], dtype=np.float64)


class EmbeddingRouterFeatures:
    """PCA and prototype-distance features fitted without router-label leakage."""

    def __init__(
        self,
        n_components: int = 16,
        n_clusters: int = 64,
        max_reference_samples: int = 100_000,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.max_reference_samples = max_reference_samples
        self.random_state = random_state

    def fit(
        self,
        h2_router_train: np.ndarray,
        h3_router_train: np.ndarray,
        h2_reference: np.ndarray,
        h3_reference: np.ndarray,
    ) -> "EmbeddingRouterFeatures":
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA

        h2_router_train = np.asarray(h2_router_train, dtype=np.float32)
        h3_router_train = np.asarray(h3_router_train, dtype=np.float32)
        h2_reference = np.asarray(h2_reference, dtype=np.float32)
        h3_reference = np.asarray(h3_reference, dtype=np.float32)
        self.pca_2d = PCA(
            n_components=self.n_components,
            svd_solver="randomized",
            random_state=self.random_state,
        ).fit(h2_router_train)
        self.pca_3d = PCA(
            n_components=self.n_components,
            svd_solver="randomized",
            random_state=self.random_state,
        ).fit(h3_router_train)

        rng = np.random.default_rng(self.random_state)
        n_reference = min(len(h2_reference), self.max_reference_samples)
        reference_idx = rng.choice(len(h2_reference), size=n_reference, replace=False)
        reference_2d = self.pca_2d.transform(h2_reference[reference_idx])
        reference_3d = self.pca_3d.transform(h3_reference[reference_idx])
        self.kmeans_2d = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=4096,
            n_init=3,
            random_state=self.random_state,
        ).fit(reference_2d)
        self.kmeans_3d = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=4096,
            n_init=3,
            random_state=self.random_state,
        ).fit(reference_3d)

        distance_2d = self.kmeans_2d.transform(reference_2d).min(axis=1)
        distance_3d = self.kmeans_3d.transform(reference_3d).min(axis=1)
        self.distance_p95_2d = float(np.quantile(distance_2d, 0.95))
        self.distance_p95_3d = float(np.quantile(distance_3d, 0.95))
        self.n_reference_fitted = int(n_reference)
        return self

    def transform(self, h2: np.ndarray, h3: np.ndarray) -> dict[str, np.ndarray]:
        h2 = np.asarray(h2, dtype=np.float32)
        h3 = np.asarray(h3, dtype=np.float32)
        z2 = self.pca_2d.transform(h2)
        z3 = self.pca_3d.transform(h3)
        distances_2d = self.kmeans_2d.transform(z2)
        distances_3d = self.kmeans_3d.transform(z3)
        nearest_2d = distances_2d.min(axis=1)
        nearest_3d = distances_3d.min(axis=1)
        result = {
            **{f"gps_pca_{i + 1:02d}": z2[:, i] for i in range(z2.shape[1])},
            **{f"schnet_pca_{i + 1:02d}": z3[:, i] for i in range(z3.shape[1])},
            "gps_embedding_norm": np.linalg.norm(h2, axis=1),
            "schnet_embedding_norm": np.linalg.norm(h3, axis=1),
            "gps_prototype_min_distance": nearest_2d,
            "gps_prototype_5mean_distance": np.partition(distances_2d, 4, axis=1)[:, :5].mean(axis=1),
            "gps_prototype_distance_ratio": nearest_2d / max(self.distance_p95_2d, 1e-12),
            "gps_prototype_over_p95": nearest_2d > self.distance_p95_2d,
            "schnet_prototype_min_distance": nearest_3d,
            "schnet_prototype_5mean_distance": np.partition(distances_3d, 4, axis=1)[:, :5].mean(axis=1),
            "schnet_prototype_distance_ratio": nearest_3d / max(self.distance_p95_3d, 1e-12),
            "schnet_prototype_over_p95": nearest_3d > self.distance_p95_3d,
        }
        return result

    def manifest(self) -> dict[str, object]:
        return {
            "n_components": int(self.n_components),
            "n_clusters": int(self.n_clusters),
            "n_reference_fitted": int(self.n_reference_fitted),
            "distance_p95_2d": float(self.distance_p95_2d),
            "distance_p95_3d": float(self.distance_p95_3d),
            "random_state": int(self.random_state),
        }


def per_molecule_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Return weighted absolute error for every molecule."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if weights is None:
        weight_array = np.full(y_true.shape[1], 1.0 / y_true.shape[1])
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != (y_true.shape[1],):
            raise ValueError(
                f"Expected {y_true.shape[1]} weights, got shape {weight_array.shape}"
            )
        if np.any(weight_array < 0) or not np.isfinite(weight_array).all():
            raise ValueError("Weights must be finite and non-negative")
        total = float(weight_array.sum())
        if total <= 0:
            raise ValueError("At least one target weight must be positive")
        weight_array = weight_array / total
    return np.abs(y_pred - y_true) @ weight_array


def paired_bootstrap_mean(
    delta: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    max_chunk_cells: int = 2_000_000,
) -> dict[str, object]:
    """Bootstrap a paired per-molecule delta without allocating all draws at once."""
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    if len(delta) == 0:
        raise ValueError("Cannot bootstrap an empty delta")
    if n_bootstrap <= 0:
        return {
            "delta": float(delta.mean()),
            "ci95": None,
            "probability_better": None,
            "n_bootstrap": 0,
            "seed": int(seed),
        }

    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    chunk_size = max(1, min(n_bootstrap, max_chunk_cells // len(delta)))
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        draws[start:stop] = delta[indices].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "delta": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
        "probability_better": float((draws < 0).mean()),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def _prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str],
    weights: Sequence[float],
) -> dict[str, object]:
    errors = np.abs(y_pred - y_true)
    result: dict[str, object] = {
        name: {"mae": float(errors[:, i].mean())}
        for i, name in enumerate(target_names)
    }
    result["average"] = {"mae": float(errors.mean())}
    result["weighted"] = {"mae": float(per_molecule_loss(y_true, y_pred, weights).mean())}
    return result


def _route_diagnostics(mask: np.ndarray, gain: np.ndarray, win_delta: float) -> dict[str, object]:
    wins = gain > win_delta
    true_positive = int(np.count_nonzero(mask & wins))
    routed = int(mask.sum())
    total_wins = int(wins.sum())
    return {
        "route_n": routed,
        "route_fraction": float(mask.mean()),
        "true_positive_n": true_positive,
        "false_positive_n": int(np.count_nonzero(mask & ~wins)),
        "false_negative_n": int(np.count_nonzero(~mask & wins)),
        "precision": float(true_positive / routed) if routed else None,
        "recall": float(true_positive / total_wins) if total_wins else None,
    }


def _downside_summary(mask: np.ndarray, gain: np.ndarray) -> dict[str, object]:
    downside = np.maximum(-gain[mask], 0.0)
    if len(downside) == 0:
        return {"n": 0}
    return {
        "n": int(len(downside)),
        "wrong_route_n": int(np.count_nonzero(downside > 0)),
        "wrong_route_fraction": float(np.mean(downside > 0)),
        "over_0.01_fraction": float(np.mean(downside > 0.01)),
        "over_0.05_fraction": float(np.mean(downside > 0.05)),
        "mean": float(downside.mean()),
        "max": float(downside.max()),
        "p90": float(np.quantile(downside, 0.90)),
        "p95": float(np.quantile(downside, 0.95)),
        "p99": float(np.quantile(downside, 0.99)),
    }


def oracle_router_analysis(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    expert_pred: np.ndarray,
    fixed_route: np.ndarray,
    *,
    target_names: Sequence[str],
    weights: Sequence[float],
    win_deltas: Sequence[float] = (0.0, 0.001, 0.002),
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Compare base, fixed routing, unrestricted Oracle, and budget Oracle."""
    y_true = np.asarray(y_true, dtype=np.float64)
    base_pred = np.asarray(base_pred, dtype=np.float64)
    expert_pred = np.asarray(expert_pred, dtype=np.float64)
    fixed_route = np.asarray(fixed_route, dtype=bool).reshape(-1)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        base_pred = base_pred[:, None]
        expert_pred = expert_pred[:, None]
    if not (y_true.shape == base_pred.shape == expert_pred.shape):
        raise ValueError("True, base, and expert arrays must have identical shapes")
    if len(fixed_route) != len(y_true):
        raise ValueError("Route mask length does not match predictions")

    base_loss = per_molecule_loss(y_true, base_pred, weights)
    expert_loss = per_molecule_loss(y_true, expert_pred, weights)
    gain = base_loss - expert_loss
    oracle_route = gain > 0

    budget_n = int(fixed_route.sum())
    budget_route = np.zeros(len(gain), dtype=bool)
    if budget_n:
        top = np.argsort(-gain, kind="stable")[:budget_n]
        budget_route[top] = True

    routes = {
        "base": np.zeros(len(gain), dtype=bool),
        "fixed": fixed_route,
        "oracle": oracle_route,
        "budget_oracle": budget_route,
    }
    predictions: dict[str, np.ndarray] = {}
    losses: dict[str, np.ndarray] = {}
    metrics: dict[str, object] = {}
    for name, mask in routes.items():
        pred = base_pred.copy()
        pred[mask] = expert_pred[mask]
        predictions[name] = pred
        losses[name] = per_molecule_loss(y_true, pred, weights)
        metrics[name] = _prediction_metrics(y_true, pred, target_names, weights)

    fixed_gap_delta = np.abs(predictions["fixed"] - y_true)
    budget_gap_delta = np.abs(predictions["budget_oracle"] - y_true)
    oracle_gap_delta = np.abs(predictions["oracle"] - y_true)
    target_index = {name: i for i, name in enumerate(target_names)}
    gap_index = target_index.get("gap")

    bootstraps: dict[str, object] = {
        "fixed_minus_base_weighted": paired_bootstrap_mean(
            losses["fixed"] - losses["base"], n_bootstrap=n_bootstrap, seed=seed
        ),
        "budget_oracle_minus_fixed_weighted": paired_bootstrap_mean(
            losses["budget_oracle"] - losses["fixed"],
            n_bootstrap=n_bootstrap,
            seed=seed + 1,
        ),
        "oracle_minus_fixed_weighted": paired_bootstrap_mean(
            losses["oracle"] - losses["fixed"],
            n_bootstrap=n_bootstrap,
            seed=seed + 2,
        ),
    }
    if gap_index is not None:
        base_gap_error = np.abs(base_pred[:, gap_index] - y_true[:, gap_index])
        bootstraps["budget_oracle_minus_fixed_gap"] = paired_bootstrap_mean(
            budget_gap_delta[:, gap_index] - fixed_gap_delta[:, gap_index],
            n_bootstrap=n_bootstrap,
            seed=seed + 3,
        )
        bootstraps["oracle_minus_fixed_gap"] = paired_bootstrap_mean(
            oracle_gap_delta[:, gap_index] - fixed_gap_delta[:, gap_index],
            n_bootstrap=n_bootstrap,
            seed=seed + 4,
        )
        bootstraps["fixed_minus_base_gap"] = paired_bootstrap_mean(
            fixed_gap_delta[:, gap_index] - base_gap_error,
            n_bootstrap=n_bootstrap,
            seed=seed + 5,
        )

    positive = np.sort(np.maximum(gain, 0.0))[::-1]
    positive_total = float(positive.sum())
    concentration = {}
    for fraction in (0.01, 0.10):
        n_top = max(1, int(np.ceil(len(gain) * fraction)))
        concentration[f"top_{int(100 * fraction)}pct_positive_gain_share"] = (
            float(positive[:n_top].sum() / positive_total) if positive_total else 0.0
        )

    result = {
        "n": int(len(y_true)),
        "target_names": list(target_names),
        "target_weights": [float(value) for value in weights],
        "methods": metrics,
        "expert_win_rates": {
            f"gain_gt_{delta:g}": float(np.mean(gain > delta)) for delta in win_deltas
        },
        "routes": {
            "fixed": {
                f"delta_{delta:g}": _route_diagnostics(fixed_route, gain, delta)
                for delta in win_deltas
            },
            "oracle": _route_diagnostics(oracle_route, gain, 0.0),
            "budget_oracle": _route_diagnostics(budget_route, gain, 0.0),
            "fixed_oracle_overlap_fraction": float(np.mean(fixed_route == oracle_route)),
        },
        "regret": {
            "fixed_minus_oracle_weighted": float(
                losses["fixed"].mean() - losses["oracle"].mean()
            ),
            "fixed_minus_budget_oracle_weighted": float(
                losses["fixed"].mean() - losses["budget_oracle"].mean()
            ),
        },
        "gain_distribution": {
            "mean": float(gain.mean()),
            "median": float(np.median(gain)),
            "p01": float(np.quantile(gain, 0.01)),
            "p10": float(np.quantile(gain, 0.10)),
            "p90": float(np.quantile(gain, 0.90)),
            "p99": float(np.quantile(gain, 0.99)),
            **concentration,
        },
        "safety": {
            "fixed": _downside_summary(fixed_route, gain),
            "budget_oracle": _downside_summary(budget_route, gain),
        },
        "bootstrap": bootstraps,
    }
    arrays = {
        "base_loss": base_loss,
        "expert_loss": expert_loss,
        "gain": gain,
        "fixed_route": fixed_route,
        "oracle_route": oracle_route,
        "budget_oracle_route": budget_route,
        "fixed_loss": losses["fixed"],
    }
    return result, arrays


def route_policy_metrics(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    expert_pred: np.ndarray,
    route_mask: np.ndarray,
    *,
    target_names: Sequence[str],
    weights: Sequence[float],
    reference_route: np.ndarray | None = None,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate an arbitrary learned route against base and an optional control."""
    y_true = np.asarray(y_true, dtype=np.float64)
    base_pred = np.asarray(base_pred, dtype=np.float64)
    expert_pred = np.asarray(expert_pred, dtype=np.float64)
    route_mask = np.asarray(route_mask, dtype=bool)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        base_pred = base_pred[:, None]
        expert_pred = expert_pred[:, None]
    routed_pred = base_pred.copy()
    routed_pred[route_mask] = expert_pred[route_mask]
    base_loss = per_molecule_loss(y_true, base_pred, weights)
    expert_loss = per_molecule_loss(y_true, expert_pred, weights)
    routed_loss = per_molecule_loss(y_true, routed_pred, weights)
    gain = base_loss - expert_loss
    result = {
        "n": int(len(y_true)),
        "metrics": _prediction_metrics(y_true, routed_pred, target_names, weights),
        "route": _route_diagnostics(route_mask, gain, 0.0),
        "safety": _downside_summary(route_mask, gain),
        "weighted_delta_vs_base": float((routed_loss - base_loss).mean()),
    }
    if reference_route is not None:
        reference_route = np.asarray(reference_route, dtype=bool)
        reference_pred = base_pred.copy()
        reference_pred[reference_route] = expert_pred[reference_route]
        reference_loss = per_molecule_loss(y_true, reference_pred, weights)
        errors = np.abs(routed_pred - y_true)
        reference_errors = np.abs(reference_pred - y_true)
        result["weighted_delta_vs_reference"] = float(
            (routed_loss - reference_loss).mean()
        )
        result["bootstrap_weighted_vs_reference"] = paired_bootstrap_mean(
            routed_loss - reference_loss, n_bootstrap=n_bootstrap, seed=seed
        )
        if "gap" in target_names:
            gap_index = list(target_names).index("gap")
            result["bootstrap_gap_vs_reference"] = paired_bootstrap_mean(
                errors[:, gap_index] - reference_errors[:, gap_index],
                n_bootstrap=n_bootstrap,
                seed=seed + 1,
            )
        result["route_overlap_fraction"] = float(
            np.mean(route_mask == reference_route)
        )
    return result


def select_top_budget(score: np.ndarray, n_routes: int) -> np.ndarray:
    """Select the highest-scoring rows under an exact route budget."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    mask = np.zeros(len(score), dtype=bool)
    if n_routes:
        mask[np.argsort(-score, kind="stable")[:n_routes]] = True
    return mask


def apply_utility_policy(
    predicted_gain: np.ndarray,
    predicted_downside: np.ndarray,
    probability: np.ndarray,
    policy: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a calibrated utility threshold and optional batch route cap."""
    utility = np.asarray(predicted_gain) - policy["alpha"] * np.asarray(predicted_downside)
    route = (
        (utility >= policy["utility_threshold"])
        & (np.asarray(probability) >= policy["p_min"])
    )
    max_fraction = policy.get("max_route_fraction")
    if max_fraction is not None:
        max_routes = int(round(float(max_fraction) * len(route)))
        if route.sum() > max_routes:
            route = select_top_budget(np.where(route, utility, -np.inf), max_routes)
    return route, utility


def router_descriptor_row(smiles: object) -> dict[str, float]:
    """Return the interpretable, low-cost descriptors used by router analysis."""
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    names = (
        "mw", "heavy_atoms", "ring_count", "aromatic_rings", "rotatable_bonds",
        "tpsa", "logp", "fraction_csp3", "hbd", "hba", "formal_charge",
        "conjugated_bonds", "aromatic_atom_fraction", "n_N", "n_O", "n_S",
        "n_F", "n_Cl", "n_Br", "n_B", "n_P", "n_Si",
    )
    mol = safe_mol(smiles)
    if mol is None:
        return {name: float("nan") for name in names}

    atoms = list(mol.GetAtoms())
    counts: dict[str, int] = {}
    for atom in atoms:
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
    heavy_atoms = mol.GetNumHeavyAtoms()
    formal_charge = sum(atom.GetFormalCharge() for atom in atoms)
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "heavy_atoms": float(heavy_atoms),
        "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "formal_charge": float(formal_charge),
        "conjugated_bonds": float(sum(bond.GetIsConjugated() for bond in mol.GetBonds())),
        "aromatic_atom_fraction": float(
            sum(atom.GetIsAromatic() for atom in atoms) / max(heavy_atoms, 1)
        ),
        **{f"n_{symbol}": float(counts.get(symbol, 0)) for symbol in (
            "N", "O", "S", "F", "Cl", "Br", "B", "P", "Si"
        )},
    }
