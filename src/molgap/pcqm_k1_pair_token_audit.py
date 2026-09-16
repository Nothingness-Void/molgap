"""Frozen-checkpoint causal interventions for the accepted K1 PairToken."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .k1_pair_token import HIDDEN_CHANNELS, PAIR_CHANNELS, TARGET_LAYER
from .pcqm_k1_causal_audit import (
    _atomic_json,
    _descriptors,
    _load_contract,
    _summarize,
)
from .pcqm_k1_explainability import sha256_file


EXPECTED_PARAMETERS = 3_681_665
EXPECTED_CHECKPOINT_SHA256 = (
    "ac8b576a3c7e44d50012a885e13e4c5904ad17c3574319796b13518efe98d1cc"
)
EXPECTED_PAYLOAD_SHA256 = (
    "b67be3f9ba0c67ca0eb1cf7dffa00ad535e9f04f5f8371aadad1e6a2711e0331"
)
MODES = (
    "baseline",
    "disabled",
    "uniform_all_pairs",
    "diagonal_only",
    "off_diagonal_only",
    "no_pair_norm",
)


def _relation_update(module, hidden, batch, mode: str):
    import torch
    import torch.nn.functional as functional
    from torch_geometric.utils import to_dense_batch

    if mode == "disabled":
        return torch.zeros_like(hidden)
    dense, valid = to_dense_batch(hidden, batch)
    source = module.source(dense).unsqueeze(2)
    target = module.target(dense).unsqueeze(1)
    pair = functional.silu(source + target)
    if mode != "no_pair_norm":
        pair = module.pair_norm(pair)
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    diagonal = torch.eye(
        pair.shape[1], dtype=torch.bool, device=pair.device
    ).unsqueeze(0)
    if mode == "diagonal_only":
        pair_valid = pair_valid & diagonal
    elif mode == "off_diagonal_only":
        off_diagonal = pair_valid & ~diagonal
        # DTK's frozen PyTorch build does not support tuple dimensions for
        # torch.any; flattening the two pair axes is exactly equivalent.
        has_off_diagonal = off_diagonal.flatten(1).any(dim=1).view(-1, 1, 1)
        pair_valid = torch.where(
            has_off_diagonal,
            off_diagonal,
            pair_valid & diagonal,
        )
    if mode == "uniform_all_pairs":
        assignment = pair_valid.to(pair.dtype)
        assignment = assignment / assignment.sum(
            dim=(1, 2), keepdim=True
        ).clamp_min(1)
    else:
        logits = torch.einsum("bijd,d->bij", pair, module.query)
        logits = logits / math.sqrt(PAIR_CHANNELS)
        logits = logits.masked_fill(~pair_valid, float("-inf"))
        assignment = torch.softmax(logits.flatten(1), dim=-1).reshape_as(logits)
    token = torch.einsum("bij,bijd->bd", assignment, pair)
    token = module.token_norm(token + module.token_ffn(token))
    return module.return_projection(token)[batch]


def _forward(model, batch, mode: str):
    hidden = model.base._embed_nodes(batch.x)
    hidden = hidden + model.base.rwse_encoder(batch.random_walk_pe.float())
    edge_state = model.base._embed_edges(batch.edge_attr)
    for layer, (edge_update, block) in enumerate(
        zip(model.base.edge_updates, model.base.local_blocks), start=1
    ):
        edge_state = edge_update(hidden, batch.edge_index, edge_state)
        hidden = block(
            hidden, batch.edge_index, batch.batch, edge_attr=edge_state
        )
        if str(layer) in model.base.neural_atom_mixers:
            hidden = model.base.neural_atom_mixers[str(layer)](
                hidden, batch.batch
            )
        if layer == TARGET_LAYER:
            hidden = hidden + _relation_update(
                model.relation_token, hidden, batch.batch, mode
            )
    return model.base.head(model.base._pool(hidden, batch.batch)).view(-1)


def run_audit(
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
        raise RuntimeError("PairToken audit requires physical batch 128")
    if sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Accepted PairToken checkpoint changed")
    if sha256_file(reference_payload_path) != EXPECTED_PAYLOAD_SHA256:
        raise RuntimeError("Accepted PairToken payload changed")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    roles, manifest, target_mean, target_std = _load_contract(cache_root)
    model = make_encoder("neural_atom_k1_pair_token").to("cuda").eval()
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"PairToken parameter count changed: {parameters}")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu"), strict=True
    )
    loader = DataLoader(
        roles["development"],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    targets, source_indices = [], []
    prediction_parts = {mode: [] for mode in MODES}
    descriptor_parts = {}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to("cuda", non_blocking=True)
            for name, values in _descriptors(batch).items():
                descriptor_parts.setdefault(name, []).append(
                    values.float().cpu()
                )
            for mode in MODES:
                normalized = _forward(model, batch, mode)
                prediction_parts[mode].append(
                    (normalized * target_std + target_mean).float().cpu()
                )
            targets.append(batch.y.view(-1).float().cpu())
            source_indices.append(batch.source_idx.view(-1).long().cpu())

    target = torch.cat(targets)
    source_idx = torch.cat(source_indices)
    predictions = {
        mode: torch.cat(parts).numpy().astype(np.float64)
        for mode, parts in prediction_parts.items()
    }
    descriptors = {
        name: torch.cat(parts).numpy().astype(np.float64)
        for name, parts in descriptor_parts.items()
    }
    reference = torch.load(reference_payload_path, map_location="cpu")
    if not torch.equal(source_idx, reference["source_idx"].view(-1).long()):
        raise RuntimeError("PairToken payload source order changed")
    if not torch.equal(target, reference["target_eV"].view(-1).float()):
        raise RuntimeError("PairToken payload targets changed")
    frozen_prediction = reference["prediction_eV"].view(-1).float().numpy()
    baseline_difference = np.abs(predictions["baseline"] - frozen_prediction)
    baseline_mae = float(np.abs(predictions["baseline"] - target.numpy()).mean())
    frozen_mae = float(np.abs(frozen_prediction - target.numpy()).mean())
    if (
        float(baseline_difference.max()) > 5e-4
        or abs(baseline_mae - frozen_mae) > 5e-6
    ):
        raise RuntimeError("Kunshan inference did not reproduce PairToken payload")

    interventions, _ = _summarize(
        target.numpy().astype(np.float64), predictions, descriptors, {}
    )
    result = {
        "format": "molgap-pcqm-k1-pair-token-causal-audit-v1",
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
        "modes": list(MODES),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "reference_payload_sha256": sha256_file(reference_payload_path),
        "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "geometry_aggregate_sha256": manifest["geometry_aggregate_sha256"],
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
    _atomic_json(output_root / "causal_audit.json", result)
    result["artifact_sha256"] = {
        "causal_audit.json": sha256_file(output_root / "causal_audit.json")
    }
    _atomic_json(output_root / "completion_manifest.json", result)
    return result
