"""Frozen-checkpoint causal diagnostics for the PCQM K1 global exchanges.

This module evaluates the accepted K1-v4 checkpoint on the reused development
role. It never trains or mutates weights. Counterfactuals suppress one or all
of the layer-3/6/9 global updates while every local EdgeState operation stays
unchanged.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from .pcqm_k1_explainability import (
    DEVELOPMENT_ROWS,
    EXPECTED_GEOMETRY_SHA256,
    EXPECTED_MANIFEST_SHA256,
    sha256_file,
)


EXPECTED_CHECKPOINT_SHA256 = (
    "53f9118f34a95e02e3f0d798a56389af53ff55739753b4208c9f53c36d116d95"
)
EXPECTED_PAYLOAD_SHA256 = (
    "966ed31ba25e024aa82d8032e7ab6e2797402e5845a01888c8f3cfd4c084da91"
)
EXPECTED_PARAMETERS = 3_658_817
MIXER_LAYERS = (3, 6, 9)
FORBIDDEN_MODEL_FIELDS = (
    "pos",
    "edge_distance",
    "wedge_angle_cos",
    "wedge_edge_ids",
    "geometry_valid",
)
ABLATIONS = {
    "baseline": (),
    "drop_layer3": (3,),
    "drop_layer6": (6,),
    "drop_layer9": (9,),
    "drop_all_global": MIXER_LAYERS,
}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _normal_interval(values: np.ndarray) -> list[float]:
    mean = float(values.mean())
    if len(values) < 2:
        return [mean, mean]
    error = 1.96 * float(values.std(ddof=1) / math.sqrt(len(values)))
    return [mean - error, mean + error]


def _segment_mean(values, graph_index, graph_count):
    import torch

    if values.is_cuda or graph_index.is_cuda:
        raise RuntimeError("Diagnostic segment reductions must remain on CPU")
    output = torch.zeros(
        (graph_count,) + tuple(values.shape[1:]),
        dtype=values.dtype,
        device=values.device,
    )
    output.index_add_(0, graph_index, values)
    count = torch.bincount(graph_index, minlength=graph_count).clamp_min(1)
    return output / count.reshape((graph_count,) + (1,) * (values.ndim - 1))


def _graph_dispersion(hidden, batch, graph_count):
    mean = _segment_mean(hidden, batch, graph_count)
    centered_square = (hidden - mean[batch]).square().mean(dim=1)
    return _segment_mean(centered_square, batch, graph_count)


class _PackedGraphDatasetFactory:
    @staticmethod
    def load(path: Path):
        import torch
        from torch_geometric.data import InMemoryDataset

        from .pcqm_wedge import WedgeData  # noqa: F401 -- pickle dependency

        class PackedGraphDataset(InMemoryDataset):
            def __init__(self, source: Path):
                super().__init__(root=None)
                self.data, self.slices = torch.load(source, map_location="cpu")

        dataset = PackedGraphDataset(path)
        for field in FORBIDDEN_MODEL_FIELDS:
            if field in dataset._data:
                del dataset._data[field]
                dataset.slices.pop(field, None)
        return dataset


def _load_contract(cache_root: Path):
    import torch
    from torch.utils.data import ConcatDataset

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

    roles = {"train": [], "development": []}
    expected_start = {"train": 0, "development": 100_000}
    for item in manifest["geometry_shards"]:
        if item["role"] not in roles:
            continue
        path = cache_root / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Fixed shard changed: {item['file']}")
        dataset = _PackedGraphDatasetFactory.load(path)
        if len(dataset) != item["rows"]:
            raise RuntimeError(f"Fixed shard row count changed: {item['file']}")
        source = dataset._data.source_idx.view(-1).long()
        start = expected_start[item["role"]]
        expected = torch.arange(start, start + len(dataset), dtype=torch.long)
        if not torch.equal(source, expected):
            raise RuntimeError(f"Source order changed: {item['file']}")
        expected_start[item["role"]] += len(dataset)
        roles[item["role"]].append(dataset)
    combined = {name: ConcatDataset(parts) for name, parts in roles.items()}
    if len(combined["train"]) != 100_000:
        raise RuntimeError("Training row count changed")
    if len(combined["development"]) != DEVELOPMENT_ROWS:
        raise RuntimeError("Development row count changed")
    train_target = torch.cat(
        [part._data.y.view(-1).float() for part in roles["train"]]
    )
    return combined, manifest, float(train_target.mean()), float(train_target.std())


def _descriptors(batch):
    import torch

    graph_count = int(batch.num_graphs)
    node_batch = batch.batch.detach().cpu()
    edge_source = batch.edge_index[0].detach().cpu()
    edge_batch = node_batch[edge_source]
    node_count = torch.bincount(node_batch, minlength=graph_count).float()
    directed_edge_count = torch.bincount(edge_batch, minlength=graph_count).float()
    edge_attr = batch.edge_attr.detach().cpu()
    node_features = batch.x.detach().cpu()
    rwse = batch.random_walk_pe.detach().float().cpu()
    return {
        "atom_count": node_count,
        "bond_count": directed_edge_count / 2.0,
        "conjugated_bond_fraction": _segment_mean(
            (edge_attr[:, 2] != 0).float(), edge_batch, graph_count
        ),
        "ring_atom_fraction": _segment_mean(
            (node_features[:, 8] != 0).float(), node_batch, graph_count
        ),
        "rwse_mean": _segment_mean(
            rwse.mean(dim=1), node_batch, graph_count
        ),
    }


def _forward_counterfactual(model, batch, dropped_layers, collect_diagnostics):
    import torch

    h = model._embed_nodes(batch.x)
    h = h + model.rwse_encoder(batch.random_walk_pe.float())
    edge_state = model._embed_edges(batch.edge_attr)
    diagnostics = {}
    graph_count = int(batch.num_graphs)
    for layer, (edge_update, block) in enumerate(
        zip(model.edge_updates, model.local_blocks), start=1
    ):
        edge_state = edge_update(h, batch.edge_index, edge_state)
        h = block(h, batch.edge_index, batch.batch, edge_attr=edge_state)
        if layer not in MIXER_LAYERS:
            continue
        mixer = model.neural_atom_mixers[str(layer)]
        update, details = mixer.compute_update(h, batch.batch)
        if collect_diagnostics:
            assignment = details["assignment"][:, 0].detach().float().cpu()
            valid = details["valid"].detach().cpu()
            hidden_for_stats = h.detach().float().cpu()
            update_for_stats = update.detach().float().cpu()
            batch_for_stats = batch.batch.detach().cpu()
            safe = assignment.clamp_min(torch.finfo(assignment.dtype).tiny)
            entropy = -(assignment * safe.log()).sum(dim=-1)
            valid_count = valid.sum(dim=-1).clamp_min(1)
            normalized_entropy = torch.where(
                valid_count > 1,
                entropy / valid_count.float().log(),
                torch.zeros_like(entropy),
            )
            hidden_rms = _segment_mean(
                hidden_for_stats.square().mean(dim=1),
                batch_for_stats,
                graph_count,
            ).sqrt()
            update_rms = _segment_mean(
                update_for_stats.square().mean(dim=1),
                batch_for_stats,
                graph_count,
            ).sqrt()
            before_dispersion = _graph_dispersion(
                hidden_for_stats, batch_for_stats, graph_count
            )
            after_dispersion = _graph_dispersion(
                hidden_for_stats + update_for_stats,
                batch_for_stats,
                graph_count,
            )
            diagnostics[layer] = {
                "assignment_entropy_normalized": normalized_entropy,
                "assignment_effective_atoms": entropy.exp(),
                "assignment_max_mass": assignment.max(dim=-1).values,
                "update_to_hidden_rms": update_rms / hidden_rms.clamp_min(1e-12),
                "dispersion_ratio": after_dispersion
                / before_dispersion.clamp_min(1e-12),
            }
        if layer not in dropped_layers:
            h = h + update
    prediction = model.head(model._pool(h, batch.batch)).view(-1)
    return prediction, diagnostics


def _summarize(target, predictions, descriptors, diagnostics):
    masks = {
        "all": np.ones(len(target), dtype=bool),
        "small_sparse_low_conjugation": (
            (descriptors["atom_count"] <= 12)
            & (descriptors["bond_count"] <= 12)
            & (descriptors["conjugated_bond_fraction"] <= 0.2727272727272727)
        ),
        "high_ring_fraction": descriptors["ring_atom_fraction"] > 0.75,
        "high_rwse_mean": descriptors["rwse_mean"] > 0.14420703817158936,
    }
    masks["topology_extreme_union"] = (
        masks["small_sparse_low_conjugation"]
        | masks["high_ring_fraction"]
        | masks["high_rwse_mean"]
    )
    masks["middle_control"] = ~masks["topology_extreme_union"]

    baseline_error = np.abs(predictions["baseline"] - target)
    intervention = {}
    for stratum, mask in masks.items():
        if int(mask.sum()) < 100:
            raise RuntimeError(f"Causal stratum is too small: {stratum}")
        row = {
            "rows": int(mask.sum()),
            "baseline_mae_eV": float(baseline_error[mask].mean()),
            "counterfactuals": {},
        }
        for name, prediction in predictions.items():
            if name == "baseline":
                continue
            error = np.abs(prediction - target)
            gain = baseline_error[mask] - error[mask]
            row["counterfactuals"][name] = {
                "mae_eV": float(error[mask].mean()),
                "gain_vs_baseline_eV": float(gain.mean()),
                "paired_gain_normal95_eV": _normal_interval(gain),
                "row_fraction_improved": float((gain > 0).mean()),
            }
        intervention[stratum] = row

    diagnostic_summary = {}
    for layer, values in diagnostics.items():
        layer_rows = {}
        for stratum, mask in masks.items():
            layer_rows[stratum] = {
                name: {
                    "mean": float(array[mask].mean()),
                    "p10": float(np.quantile(array[mask], 0.1)),
                    "median": float(np.median(array[mask])),
                    "p90": float(np.quantile(array[mask], 0.9)),
                }
                for name, array in values.items()
            }
        diagnostic_summary[str(layer)] = layer_rows
    return intervention, diagnostic_summary


def run_causal_audit(
    cache_root: Path,
    checkpoint_path: Path,
    reference_payload_path: Path,
    output_root: Path,
    batch_size: int = 128,
) -> dict:
    import torch
    from torch_geometric.loader import DataLoader

    from .pcqm_k1_variants import make_encoder

    if sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen K1 checkpoint changed")
    if sha256_file(reference_payload_path) != EXPECTED_PAYLOAD_SHA256:
        raise RuntimeError("Frozen K1 development payload changed")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    roles, manifest, target_mean, target_std = _load_contract(cache_root)
    model = make_encoder("neural_atom_k1_v4").to("cuda").eval()
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Frozen K1 parameter count changed: {parameters}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    loader = DataLoader(
        roles["development"],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    targets, source_indices = [], []
    prediction_parts = {name: [] for name in ABLATIONS}
    descriptor_parts = {}
    diagnostic_parts = {
        layer: {
            "assignment_entropy_normalized": [],
            "assignment_effective_atoms": [],
            "assignment_max_mass": [],
            "update_to_hidden_rms": [],
            "dispersion_ratio": [],
        }
        for layer in MIXER_LAYERS
    }
    with torch.no_grad():
        for batch in loader:
            batch = batch.to("cuda", non_blocking=True)
            for name, values in _descriptors(batch).items():
                descriptor_parts.setdefault(name, []).append(values.float().cpu())
            for name, dropped in ABLATIONS.items():
                normalized, layer_diagnostics = _forward_counterfactual(
                    model,
                    batch,
                    dropped_layers=set(dropped),
                    collect_diagnostics=name == "baseline",
                )
                prediction_parts[name].append(
                    (normalized * target_std + target_mean).float().cpu()
                )
                if name == "baseline":
                    for layer, values in layer_diagnostics.items():
                        for metric, tensor in values.items():
                            diagnostic_parts[layer][metric].append(
                                tensor.float().cpu()
                            )
            targets.append(batch.y.view(-1).float().cpu())
            source_indices.append(batch.source_idx.view(-1).long().cpu())

    target = torch.cat(targets)
    source_idx = torch.cat(source_indices)
    predictions = {
        name: torch.cat(parts).numpy().astype(np.float64)
        for name, parts in prediction_parts.items()
    }
    descriptor_arrays = {
        name: torch.cat(parts).numpy().astype(np.float64)
        for name, parts in descriptor_parts.items()
    }
    diagnostic_arrays = {
        layer: {
            name: torch.cat(parts).numpy().astype(np.float64)
            for name, parts in values.items()
        }
        for layer, values in diagnostic_parts.items()
    }

    reference = torch.load(reference_payload_path, map_location="cpu")
    if not torch.equal(source_idx, reference["source_idx"].view(-1).long()):
        raise RuntimeError("Frozen payload source order changed")
    if not torch.equal(target, reference["target_eV"].view(-1).float()):
        raise RuntimeError("Frozen payload targets changed")
    frozen_prediction = reference["prediction_eV"].view(-1).float().numpy()
    baseline_difference = np.abs(predictions["baseline"] - frozen_prediction)
    baseline_mae = float(np.abs(predictions["baseline"] - target.numpy()).mean())
    frozen_mae = float(np.abs(frozen_prediction - target.numpy()).mean())
    if float(baseline_difference.max()) > 5e-4 or abs(baseline_mae - frozen_mae) > 5e-6:
        raise RuntimeError("Kunshan inference did not reproduce the frozen payload")

    intervention, diagnostic_summary = _summarize(
        target.numpy().astype(np.float64),
        predictions,
        descriptor_arrays,
        diagnostic_arrays,
    )
    result = {
        "format": "molgap-pcqm-k1-causal-audit-v1",
        "complete": True,
        "training_executed": False,
        "model_inference_executed": True,
        "promotion_authorized": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "development_rows": int(len(target)),
        "batch_size": int(batch_size),
        "parameter_count": parameters,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "reference_payload_sha256": sha256_file(reference_payload_path),
        "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "geometry_aggregate_sha256": manifest["geometry_aggregate_sha256"],
        "target_stats": {"mean_eV": target_mean, "sample_std_eV": target_std},
        "baseline_reproduction": {
            "frozen_mae_eV": frozen_mae,
            "kunshan_mae_eV": baseline_mae,
            "mae_absolute_difference_eV": abs(baseline_mae - frozen_mae),
            "prediction_max_absolute_difference_eV": float(
                baseline_difference.max()
            ),
            "prediction_mean_absolute_difference_eV": float(
                baseline_difference.mean()
            ),
        },
        "interventions": intervention,
        "diagnostics": diagnostic_summary,
        "artifact_sha256": {},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "causal_audit.json", result)
    result["artifact_sha256"] = {
        "causal_audit.json": sha256_file(output_root / "causal_audit.json")
    }
    _atomic_json(output_root / "completion_manifest.json", result)
    return result
