"""Post-hoc, label-aware diagnostics for the frozen PCQM K1 V4 screen.

This module never trains or constructs a model.  It joins already-frozen
development predictions to the accepted graph cache and reports where each
candidate changes absolute error relative to K1.  The output is hypothesis
generation only: it cannot promote a model or authorize a sealed-role read.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


EXPECTED_MANIFEST_SHA256 = (
    "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
)
EXPECTED_GEOMETRY_SHA256 = (
    "bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5"
)
REFERENCE = "neural_atom_k1_v4"
DEVELOPMENT_ROWS = 50_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _segment_sum(values: torch.Tensor, counts: torch.Tensor) -> np.ndarray:
    graph_index = torch.repeat_interleave(torch.arange(len(counts)), counts)
    result = torch.zeros(len(counts), dtype=torch.float64)
    result.index_add_(0, graph_index, values.to(torch.float64))
    return result.numpy()


def _segment_mean(values: torch.Tensor, counts: torch.Tensor) -> np.ndarray:
    denominator = counts.clamp_min(1).numpy().astype(np.float64)
    return _segment_sum(values, counts) / denominator


def _load_development_descriptors(cache_root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    manifest_path = cache_root / "manifest.json"
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Fixed 100K manifest content changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("geometry_aggregate_sha256") != EXPECTED_GEOMETRY_SHA256:
        raise RuntimeError("Fixed geometry aggregate changed")
    for role in (
        "official_validation_role_read",
        "test_dev_role_read",
        "test_challenge_role_read",
    ):
        if manifest.get(role) is not False:
            raise RuntimeError(f"Sealed role flag changed: {role}")

    development = [
        item for item in manifest["geometry_shards"] if item["role"] == "development"
    ]
    if len(development) != 1:
        raise RuntimeError(f"Expected one development shard, found {development}")
    shard = development[0]
    shard_path = cache_root / shard["file"]
    if sha256_file(shard_path) != shard["sha256"]:
        raise RuntimeError("Development shard changed")
    data, slices = torch.load(shard_path, map_location="cpu")

    source_idx = data.source_idx.view(-1).long().numpy()
    target = data.y.view(-1).float().numpy()
    expected = np.arange(100_000, 150_000, dtype=np.int64)
    if not np.array_equal(source_idx, expected):
        raise RuntimeError("Development source order changed")

    node_counts = (slices["x"][1:] - slices["x"][:-1]).long()
    edge_counts = (slices["edge_index"][1:] - slices["edge_index"][:-1]).long()
    if len(node_counts) != DEVELOPMENT_ROWS or len(edge_counts) != DEVELOPMENT_ROWS:
        raise RuntimeError("Development graph count changed")
    x = data.x.long()
    edge_attr = data.edge_attr.long()
    rwse = data.random_walk_pe.float()

    descriptors = {
        "atom_count": node_counts.numpy().astype(np.float64),
        "bond_count": (edge_counts.numpy() / 2.0).astype(np.float64),
        "hetero_atom_fraction": _segment_mean((x[:, 0] != 5).float(), node_counts),
        "aromatic_atom_fraction": _segment_mean((x[:, 7] != 0).float(), node_counts),
        "ring_atom_fraction": _segment_mean((x[:, 8] != 0).float(), node_counts),
        "charged_atom_fraction": _segment_mean((x[:, 3] != 5).float(), node_counts),
        "radical_atom_fraction": _segment_mean((x[:, 5] != 0).float(), node_counts),
        "multiple_bond_fraction": _segment_mean((edge_attr[:, 0] != 0).float(), edge_counts),
        "conjugated_bond_fraction": _segment_mean((edge_attr[:, 2] != 0).float(), edge_counts),
        "rwse_mean": _segment_mean(rwse.mean(dim=1), node_counts),
        "rwse_terminal_mean": _segment_mean(rwse[:, -1], node_counts),
        "target_gap_eV": target.astype(np.float64),
    }
    descriptors["bond_per_atom"] = descriptors["bond_count"] / np.maximum(
        descriptors["atom_count"], 1.0
    )
    return source_idx, target, descriptors, manifest


def _load_payloads(payload_root: Path, expected_source: np.ndarray, expected_target: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    manifest_path = payload_root / "payload_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions: dict[str, np.ndarray] = {}
    for item in manifest["payloads"]:
        path = payload_root / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Prediction payload changed: {item['mode']}")
        payload = torch.load(path, map_location="cpu")
        source_idx = payload["source_idx"].view(-1).long().numpy()
        target = payload["target_eV"].view(-1).float().numpy()
        prediction = payload["prediction_eV"].view(-1).float().numpy()
        if not np.array_equal(source_idx, expected_source):
            raise RuntimeError(f"Source order differs for {item['mode']}")
        if not np.array_equal(target, expected_target):
            raise RuntimeError(f"Target values differ for {item['mode']}")
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Non-finite prediction for {item['mode']}")
        predictions[item["mode"]] = prediction.astype(np.float64)
    if REFERENCE not in predictions:
        raise RuntimeError("Frozen K1-v4 reference payload is absent")
    return predictions, manifest


def _quantile_edges(values: np.ndarray, bins: int = 5) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return np.asarray([float(values.min()), float(values.max()) + 1.0])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _paired_normal_interval(values: np.ndarray) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    mean = float(values.mean())
    if len(values) == 1:
        return [mean, mean]
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    return [mean - 1.96 * standard_error, mean + 1.96 * standard_error]


def run_error_audit(cache_root: Path, payload_root: Path, output_root: Path) -> dict:
    source_idx, target, descriptors, cache_manifest = _load_development_descriptors(cache_root)
    predictions, payload_manifest = _load_payloads(payload_root, source_idx, target)
    absolute_error = {name: np.abs(value - target) for name, value in predictions.items()}
    reference_error = absolute_error[REFERENCE]

    overall = {}
    for name, errors in sorted(absolute_error.items()):
        paired_gain = reference_error - errors
        overall[name] = {
            "mae_eV": float(errors.mean()),
            "gain_vs_reference_eV": float(paired_gain.mean()),
            "paired_gain_normal95_eV": _paired_normal_interval(paired_gain),
        }

    slice_rows = []
    for descriptor_name, values in descriptors.items():
        edges = _quantile_edges(values)
        groups = np.digitize(values, edges[1:-1], right=True)
        for group in range(len(edges) - 1):
            mask = groups == group
            if int(mask.sum()) < 100:
                continue
            for name, errors in sorted(absolute_error.items()):
                paired_gain = reference_error[mask] - errors[mask]
                slice_rows.append(
                    {
                        "descriptor": descriptor_name,
                        "bin": group,
                        "lower": None if np.isneginf(edges[group]) else float(edges[group]),
                        "upper": None if np.isposinf(edges[group + 1]) else float(edges[group + 1]),
                        "rows": int(mask.sum()),
                        "mode": name,
                        "mae_eV": float(errors[mask].mean()),
                        "gain_vs_reference_eV": float(paired_gain.mean()),
                        "paired_gain_normal95_eV": _paired_normal_interval(paired_gain),
                    }
                )

    candidate_names = [name for name in sorted(absolute_error) if name != REFERENCE]
    stacked = np.stack([absolute_error[name] for name in candidate_names], axis=0)
    oracle = np.minimum(reference_error, stacked.min(axis=0))
    reference_hard = reference_error >= np.quantile(reference_error, 0.9)
    candidate_best = np.argmin(stacked, axis=0)
    winner_counts = {
        name: int((candidate_best == index).sum())
        for index, name in enumerate(candidate_names)
    }
    result = {
        "format": "molgap-pcqm-k1-error-attribution-v1",
        "complete": True,
        "analysis_scope": "accepted-official-train-derived-development-role",
        "promotion_authorized": False,
        "model_inference_executed": False,
        "training_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "geometry_aggregate_sha256": cache_manifest["geometry_aggregate_sha256"],
        "payload_manifest_sha256": sha256_file(payload_root / "payload_manifest.json"),
        "development_rows": int(len(target)),
        "reference": REFERENCE,
        "overall": overall,
        "slice_rows": slice_rows,
        "diagnostics": {
            "reference_hard_decile_rows": int(reference_hard.sum()),
            "reference_hard_decile_mae_eV": float(reference_error[reference_hard].mean()),
            "oracle_minimum_across_existing_variants_mae_eV": float(oracle.mean()),
            "oracle_headroom_vs_reference_eV": float(reference_error.mean() - oracle.mean()),
            "oracle_is_not_a_deployable_fusion_result": True,
            "candidate_lowest_error_row_counts": winner_counts,
        },
        "artifact_sha256": {},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "error_attribution.json", result)
    result["artifact_sha256"] = {
        "error_attribution.json": sha256_file(output_root / "error_attribution.json")
    }
    _atomic_json(output_root / "completion_manifest.json", result)
    return result
