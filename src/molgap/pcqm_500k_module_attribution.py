"""No-inference structural attribution for accepted matched-500K predictions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ROWS = 50_000
EXPECTED_SOURCE_START = 500_000
BOOTSTRAP_SEED = 20_260_918


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _payload_array(payload: dict[str, Any], *names: str) -> np.ndarray:
    for name in names:
        if name in payload:
            value = payload[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value).reshape(-1)
    raise ValueError(f"Prediction payload lacks all aliases: {names}")


def load_prediction(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Prediction payload is not a mapping: {path}")
    result = {
        "prediction": _payload_array(payload, "prediction_eV", "prediction"),
        "target": _payload_array(payload, "target_eV", "target"),
        "source_idx": _payload_array(payload, "source_idx").astype(np.int64),
        "sha256": sha256_file(path),
    }
    if any(len(result[key]) != EXPECTED_ROWS for key in ("prediction", "target", "source_idx")):
        raise ValueError(f"Prediction payload must contain {EXPECTED_ROWS} rows: {path}")
    if not np.isfinite(result["prediction"]).all() or not np.isfinite(result["target"]).all():
        raise ValueError(f"Prediction payload contains non-finite values: {path}")
    expected = np.arange(EXPECTED_SOURCE_START, EXPECTED_SOURCE_START + EXPECTED_ROWS)
    if not np.array_equal(result["source_idx"], expected):
        raise ValueError(f"Unexpected development source_idx identity: {path}")
    return result


def _graph_distances(num_nodes: int, edge_index: np.ndarray) -> tuple[int, float, int]:
    neighbors = [[] for _ in range(num_nodes)]
    for source, target in edge_index.T:
        source_i, target_i = int(source), int(target)
        if target_i not in neighbors[source_i]:
            neighbors[source_i].append(target_i)

    component = [-1] * num_nodes
    component_count = 0
    diameter = 0
    distance_total = 0
    pair_count = 0
    for root in range(num_nodes):
        if component[root] < 0:
            queue = [root]
            component[root] = component_count
            for node in queue:
                for neighbor in neighbors[node]:
                    if component[neighbor] < 0:
                        component[neighbor] = component_count
                        queue.append(neighbor)
            component_count += 1

        distances = [-1] * num_nodes
        distances[root] = 0
        queue = [root]
        for node in queue:
            for neighbor in neighbors[node]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        reachable = [distance for distance in distances[root + 1 :] if distance >= 0]
        if reachable:
            diameter = max(diameter, max(reachable))
            distance_total += sum(reachable)
            pair_count += len(reachable)
    mean_distance = distance_total / pair_count if pair_count else 0.0
    return diameter, mean_distance, component_count


def extract_graph_features(graph_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    loaded = torch.load(graph_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise TypeError("Expected a PyG collated (data, slices) tuple")
    data, slices = loaded
    required = {
        "x",
        "edge_index",
        "edge_attr",
        "y",
        "source_idx",
        "random_walk_pe",
    }
    if not required.issubset(slices):
        raise ValueError(f"Graph shard lacks fields: {sorted(required - set(slices))}")
    rows = len(slices["source_idx"]) - 1
    if rows != EXPECTED_ROWS:
        raise ValueError(f"Graph shard must contain {EXPECTED_ROWS} rows, got {rows}")
    source_idx = data.source_idx.detach().cpu().numpy().reshape(-1).astype(np.int64)
    expected = np.arange(EXPECTED_SOURCE_START, EXPECTED_SOURCE_START + EXPECTED_ROWS)
    if not np.array_equal(source_idx, expected):
        raise ValueError("Graph shard development source_idx identity mismatch")

    feature_names = (
        "atom_count",
        "bond_count",
        "mean_degree",
        "max_degree",
        "leaf_fraction",
        "branch_atom_fraction",
        "component_count",
        "cycle_rank",
        "diameter",
        "mean_shortest_path",
        "heteroatom_fraction",
        "aromatic_atom_fraction",
        "ring_atom_fraction",
        "charged_atom_fraction",
        "radical_atom_fraction",
        "non_single_bond_fraction",
        "aromatic_bond_fraction",
        "conjugated_bond_fraction",
        "stereo_bond_fraction",
        "rwse_mean",
        "rwse_last_mean",
        "rwse_node_variance",
    )
    values = {name: np.empty(rows, dtype=np.float64) for name in feature_names}
    targets = data.y.detach().cpu().numpy().reshape(-1).astype(np.float64)
    x_all = data.x.detach().cpu().numpy()
    edges_all = data.edge_index.detach().cpu().numpy()
    edge_attr_all = data.edge_attr.detach().cpu().numpy()
    rwse_all = data.random_walk_pe.detach().cpu().numpy().astype(np.float64)

    for index in range(rows):
        node_start, node_stop = (int(v) for v in slices["x"][index : index + 2])
        edge_start, edge_stop = (
            int(v) for v in slices["edge_index"][index : index + 2]
        )
        x = x_all[node_start:node_stop]
        edge_index = edges_all[:, edge_start:edge_stop]
        edge_attr = edge_attr_all[edge_start:edge_stop]
        rwse = rwse_all[node_start:node_stop]
        num_nodes = len(x)
        directed_edges = edge_index.shape[1]
        num_bonds = directed_edges / 2.0
        degree = np.bincount(edge_index[0], minlength=num_nodes)
        diameter, mean_distance, components = _graph_distances(num_nodes, edge_index)
        cycle_rank = max(0.0, num_bonds - num_nodes + components)

        # OGB categorical schema: carbon is atomic-number index 5; aromatic and
        # ring flags are columns 7 and 8. Bond type 0 is single and 3 aromatic.
        values["atom_count"][index] = num_nodes
        values["bond_count"][index] = num_bonds
        values["mean_degree"][index] = degree.mean() if num_nodes else 0.0
        values["max_degree"][index] = degree.max() if num_nodes else 0.0
        values["leaf_fraction"][index] = np.mean(degree == 1) if num_nodes else 0.0
        values["branch_atom_fraction"][index] = np.mean(degree >= 3) if num_nodes else 0.0
        values["component_count"][index] = components
        values["cycle_rank"][index] = cycle_rank
        values["diameter"][index] = diameter
        values["mean_shortest_path"][index] = mean_distance
        values["heteroatom_fraction"][index] = np.mean(x[:, 0] != 5)
        values["aromatic_atom_fraction"][index] = np.mean(x[:, 7] == 1)
        values["ring_atom_fraction"][index] = np.mean(x[:, 8] == 1)
        values["charged_atom_fraction"][index] = np.mean(x[:, 3] != 5)
        values["radical_atom_fraction"][index] = np.mean(x[:, 5] != 0)
        if directed_edges:
            values["non_single_bond_fraction"][index] = np.mean(edge_attr[:, 0] != 0)
            values["aromatic_bond_fraction"][index] = np.mean(edge_attr[:, 0] == 3)
            values["conjugated_bond_fraction"][index] = np.mean(edge_attr[:, 2] == 1)
            values["stereo_bond_fraction"][index] = np.mean(edge_attr[:, 1] != 0)
        else:
            for name in (
                "non_single_bond_fraction",
                "aromatic_bond_fraction",
                "conjugated_bond_fraction",
                "stereo_bond_fraction",
            ):
                values[name][index] = 0.0
        values["rwse_mean"][index] = rwse.mean() if rwse.size else 0.0
        values["rwse_last_mean"][index] = rwse[:, -1].mean() if rwse.size else 0.0
        values["rwse_node_variance"][index] = (
            rwse.var(axis=0).mean() if rwse.size else 0.0
        )

    identity = {
        "graph_sha256": sha256_file(graph_path),
        "rows": rows,
        "source_idx_min": int(source_idx.min()),
        "source_idx_max": int(source_idx.max()),
        "target": targets,
        "source_idx": source_idx,
    }
    return values, identity


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.ptp(first) == 0 or np.ptp(second) == 0:
        return 0.0
    result = spearmanr(first, second).statistic
    return 0.0 if not np.isfinite(result) else float(result)


def _quantile_bins(values: np.ndarray, maximum_bins: int = 4) -> list[np.ndarray]:
    unique = np.unique(values)
    if len(unique) <= maximum_bins:
        return [values == item for item in unique]
    boundaries = np.unique(np.quantile(values, np.linspace(0.0, 1.0, maximum_bins + 1)))
    if len(boundaries) <= 2:
        return [np.ones(len(values), dtype=bool)]
    bins = []
    for index in range(len(boundaries) - 1):
        if index == len(boundaries) - 2:
            mask = (values >= boundaries[index]) & (values <= boundaries[index + 1])
        else:
            mask = (values >= boundaries[index]) & (values < boundaries[index + 1])
        if mask.any():
            bins.append(mask)
    return bins


def _feature_bins(
    values: np.ndarray,
    errors: dict[str, np.ndarray],
    gains: dict[str, np.ndarray],
) -> dict[str, Any]:
    records = []
    for mask in _quantile_bins(values):
        records.append(
            {
                "rows": int(mask.sum()),
                "feature_min": float(values[mask].min()),
                "feature_max": float(values[mask].max()),
                "feature_mean": float(values[mask].mean()),
                "mae_eV": {name: float(error[mask].mean()) for name, error in errors.items()},
                "mean_gain_eV": {name: float(gain[mask].mean()) for name, gain in gains.items()},
            }
        )
    spreads = {
        name: float(max(row["mean_gain_eV"][name] for row in records) - min(row["mean_gain_eV"][name] for row in records))
        for name in gains
    }
    return {"bins": records, "gain_spread_eV": spreads}


def _tail_overlap(errors: dict[str, np.ndarray], fraction: float) -> dict[str, float]:
    count = max(1, int(round(len(next(iter(errors.values()))) * fraction)))
    tails = {
        name: set(np.argpartition(error, -count)[-count:].tolist())
        for name, error in errors.items()
    }
    result = {}
    names = sorted(errors)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            union = tails[first] | tails[second]
            result[f"{first}__{second}"] = len(tails[first] & tails[second]) / len(union)
    return result


def _diagnostic_crossfit(
    features: dict[str, np.ndarray], gain: np.ndarray, source_idx: np.ndarray
) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score

    names = sorted(features)
    matrix = np.column_stack([features[name] for name in names])
    oof_gain = np.empty_like(gain)
    oof_probability = np.empty_like(gain)
    winner = gain > 0
    folds = source_idx % 5
    for fold in range(5):
        train = folds != fold
        held_out = ~train
        regressor = HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=15, learning_rate=0.05, random_state=42
        )
        classifier = HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, learning_rate=0.05, random_state=42
        )
        regressor.fit(matrix[train], gain[train])
        classifier.fit(matrix[train], winner[train])
        oof_gain[held_out] = regressor.predict(matrix[held_out])
        oof_probability[held_out] = classifier.predict_proba(matrix[held_out])[:, 1]
    denominator = float(np.sum((gain - gain.mean()) ** 2))
    r2 = 1.0 - float(np.sum((gain - oof_gain) ** 2)) / denominator if denominator else 0.0
    return {
        "feature_names": names,
        "gain_regression_oof_r2": r2,
        "gain_regression_oof_pearson": float(np.corrcoef(gain, oof_gain)[0, 1]),
        "winner_classifier_oof_auc": float(roc_auc_score(winner, oof_probability)),
        "diagnostic_fit_only": True,
    }


def analyze(
    graph_path: Path,
    prediction_paths: dict[str, Path],
    *,
    expected_hashes: dict[str, str] | None = None,
    expected_graph_hash: str | None = None,
) -> dict[str, Any]:
    features, graph_identity = extract_graph_features(graph_path)
    payloads = {name: load_prediction(path) for name, path in prediction_paths.items()}
    if expected_graph_hash and graph_identity["graph_sha256"] != expected_graph_hash:
        raise ValueError("Graph shard SHA256 mismatch")
    if expected_hashes:
        for name, expected in expected_hashes.items():
            if payloads[name]["sha256"] != expected:
                raise ValueError(f"Prediction SHA256 mismatch: {name}")

    if "edge_state" not in payloads or "gptrans_t" not in payloads:
        raise ValueError("edge_state and gptrans_t are required")
    reference = payloads["edge_state"]
    for name, payload in payloads.items():
        if not np.array_equal(payload["source_idx"], reference["source_idx"]):
            raise ValueError(f"Prediction source_idx mismatch: {name}")
        if not np.array_equal(payload["target"], reference["target"]):
            raise ValueError(f"Prediction target mismatch: {name}")
    if not np.array_equal(reference["source_idx"], graph_identity["source_idx"]):
        raise ValueError("Graph/prediction source_idx mismatch")
    if not np.array_equal(reference["target"].astype(np.float32), graph_identity["target"].astype(np.float32)):
        raise ValueError("Graph/prediction target mismatch")

    target = reference["target"].astype(np.float64)
    predictions = {
        name: payload["prediction"].astype(np.float64) for name, payload in payloads.items()
    }
    errors = {name: np.abs(prediction - target) for name, prediction in predictions.items()}
    gains = {
        "gptrans_t_over_edge_state": errors["edge_state"] - errors["gptrans_t"]
    }
    if "k1" in errors:
        gains.update(
            {
                "k1_over_gptrans_t": errors["gptrans_t"] - errors["k1"],
                "k1_over_edge_state": errors["edge_state"] - errors["k1"],
            }
        )
    correlations = {
        feature_name: {
            gain_name: _spearman(feature, gain) for gain_name, gain in gains.items()
        }
        for feature_name, feature in features.items()
    }
    bins = {
        name: _feature_bins(value, errors, gains) for name, value in features.items()
    }
    ranked = {
        gain_name: sorted(
            (
                {
                    "feature": feature_name,
                    "spearman": correlations[feature_name][gain_name],
                    "absolute_spearman": abs(correlations[feature_name][gain_name]),
                    "quartile_gain_spread_eV": bins[feature_name]["gain_spread_eV"][gain_name],
                }
                for feature_name in features
            ),
            key=lambda item: (item["absolute_spearman"], item["quartile_gain_spread_eV"]),
            reverse=True,
        )
        for gain_name in gains
    }
    diagnostics = {
        "gptrans_t_vs_edge_state": _diagnostic_crossfit(
            features, gains["gptrans_t_over_edge_state"], reference["source_idx"]
        )
    }
    if "k1" in errors:
        diagnostics.update(
            {
                "k1_vs_gptrans_t": _diagnostic_crossfit(
                    features, gains["k1_over_gptrans_t"], reference["source_idx"]
                ),
                "k1_vs_edge_state": _diagnostic_crossfit(
                    features, gains["k1_over_edge_state"], reference["source_idx"]
                ),
            }
        )
    return {
        "format": "molgap-pcqm-500k-module-attribution-v1",
        "benchmark_id": "pcqm-fixed500k-dev50k-matched60-v4",
        "rows": EXPECTED_ROWS,
        "model_inference_executed": False,
        "encoder_training_executed": False,
        "diagnostic_fit_executed": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "identity": {
            "graph_sha256": graph_identity["graph_sha256"],
            "prediction_sha256": {
                name: payload["sha256"] for name, payload in payloads.items()
            },
            "source_idx_min": int(reference["source_idx"].min()),
            "source_idx_max": int(reference["source_idx"].max()),
            "source_idx_and_target_alignment": True,
        },
        "mae_eV": {name: float(error.mean()) for name, error in errors.items()},
        "signed_residual_correlation": {
            "edge_state__gptrans_t": float(
                np.corrcoef(
                    predictions["edge_state"] - target,
                    predictions["gptrans_t"] - target,
                )[0, 1]
            )
        },
        "tail_jaccard": {
            "top_1pct": _tail_overlap(errors, 0.01),
            "top_5pct": _tail_overlap(errors, 0.05),
            "top_10pct": _tail_overlap(errors, 0.10),
        },
        "feature_correlations": correlations,
        "feature_rankings": ranked,
        "feature_bins": bins,
        "structural_diagnostic_crossfit": diagnostics,
    }
