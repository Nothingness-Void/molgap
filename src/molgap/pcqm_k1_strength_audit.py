"""Frozen K1 layer-6 residual-strength counterfactuals.

Round 1 found no dispensable global exchange, but layer 6 had the strongest
topology-dependent update-magnitude inflation. This module changes only the
coefficient applied to that already-trained update and never trains weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pcqm_k1_causal_audit import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_PARAMETERS,
    EXPECTED_PAYLOAD_SHA256,
    MIXER_LAYERS,
    _atomic_json,
    _descriptors,
    _load_contract,
    _summarize,
)
from .pcqm_k1_explainability import sha256_file


TARGET_LAYER = 6
STRENGTHS = {
    "layer6_scale_0p50": 0.50,
    "layer6_scale_0p75": 0.75,
    "baseline": 1.00,
    "layer6_scale_1p25": 1.25,
    "layer6_scale_1p50": 1.50,
}


def _forward_strength(model, batch, layer6_strength: float):
    hidden = model._embed_nodes(batch.x)
    hidden = hidden + model.rwse_encoder(batch.random_walk_pe.float())
    edge_state = model._embed_edges(batch.edge_attr)
    for layer, (edge_update, block) in enumerate(
        zip(model.edge_updates, model.local_blocks), start=1
    ):
        edge_state = edge_update(hidden, batch.edge_index, edge_state)
        hidden = block(
            hidden,
            batch.edge_index,
            batch.batch,
            edge_attr=edge_state,
        )
        if layer not in MIXER_LAYERS:
            continue
        mixer = model.neural_atom_mixers[str(layer)]
        update, _ = mixer.compute_update(hidden, batch.batch)
        strength = layer6_strength if layer == TARGET_LAYER else 1.0
        hidden = hidden + strength * update
    return model.head(model._pool(hidden, batch.batch)).view(-1)


def run_strength_audit(
    cache_root: Path,
    checkpoint_path: Path,
    reference_payload_path: Path,
    output_root: Path,
    batch_size: int = 128,
) -> dict:
    import torch
    from torch_geometric.loader import DataLoader

    from .pcqm_k1_variants import make_encoder

    if batch_size != 128:
        raise RuntimeError("Round-2 physical inference batch must be 128")
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
    prediction_parts = {name: [] for name in STRENGTHS}
    descriptor_parts = {}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to("cuda", non_blocking=True)
            for name, values in _descriptors(batch).items():
                descriptor_parts.setdefault(name, []).append(values.float().cpu())
            for name, strength in STRENGTHS.items():
                normalized = _forward_strength(model, batch, strength)
                prediction_parts[name].append(
                    (normalized * target_std + target_mean).float().cpu()
                )
            targets.append(batch.y.view(-1).float().cpu())
            source_indices.append(batch.source_idx.view(-1).long().cpu())

    target = torch.cat(targets)
    source_idx = torch.cat(source_indices)
    predictions = {
        name: torch.cat(parts).numpy().astype(np.float64)
        for name, parts in prediction_parts.items()
    }
    descriptors = {
        name: torch.cat(parts).numpy().astype(np.float64)
        for name, parts in descriptor_parts.items()
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

    interventions, _ = _summarize(
        target.numpy().astype(np.float64),
        predictions,
        descriptors,
        {},
    )
    result = {
        "format": "molgap-pcqm-k1-layer6-strength-audit-v1",
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
        "target_layer": TARGET_LAYER,
        "strengths": STRENGTHS,
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
        "interventions": interventions,
        "artifact_sha256": {},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "strength_audit.json", result)
    result["artifact_sha256"] = {
        "strength_audit.json": sha256_file(output_root / "strength_audit.json")
    }
    _atomic_json(output_root / "completion_manifest.json", result)
    return result
