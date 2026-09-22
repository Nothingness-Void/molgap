"""Frozen PairToken attention concentration on the fixed PCQM development role."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from .pcqm_k1_causal_audit import _descriptors, _load_contract, _atomic_json
from .pcqm_k1_explainability import sha256_file
from .pcqm_k1_pair_token_audit import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_PAYLOAD_SHA256,
    EXPECTED_PARAMETERS,
)


K1_PAYLOAD_SHA256 = "966ed31ba25e024aa82d8032e7ab6e2797402e5845a01888c8f3cfd4c084da91"
CHUNK_ROWS = 5_000
DEVELOPMENT_ROWS = 50_000
METRICS = (
    "top20_mass", "effective_pair_fraction", "diagonal_mass",
    "bonded_mass", "two_hop_mass", "nonlocal_mass", "atom_count",
    "conjugated_bond_fraction", "k1_absolute_error", "paired_gain_eV",
)


def attention_metrics(assignment, valid, edge_index, node_batch):
    """Return per-graph masses; topology is computed from existing real bonds."""
    import torch
    from torch_geometric.utils import to_dense_adj

    b, n, _ = assignment.shape
    counts = valid.sum(1).long()
    pairs = counts.square()
    flat = assignment.flatten(1)
    sorted_mass = flat.sort(dim=1, descending=True).values.cumsum(dim=1)
    k = torch.ceil(pairs.float() * 0.20).long().clamp_min(1)
    top20 = sorted_mass.gather(1, (k - 1).unsqueeze(1)).squeeze(1)
    positive = assignment.clamp_min(torch.finfo(assignment.dtype).tiny)
    entropy = -(assignment * positive.log()).sum((1, 2))
    effective = entropy.exp() / pairs.float()
    diag_mask = torch.eye(n, device=assignment.device, dtype=torch.bool)[None]
    diagonal = (assignment * diag_mask).sum((1, 2))

    adjacency = to_dense_adj(
        edge_index, batch=node_batch, max_num_nodes=n
    ) > 0
    adjacency &= valid[:, :, None] & valid[:, None, :]
    two_hop = torch.bmm(adjacency.float(), adjacency.float()) > 0
    two_hop &= ~adjacency & ~diag_mask
    bonded = (assignment * adjacency).sum((1, 2))
    distance_two = (assignment * two_hop).sum((1, 2))
    nonlocal_mass = (assignment * (~adjacency & ~two_hop & ~diag_mask)).sum((1, 2))
    result = {
        "top20_mass": top20,
        "effective_pair_fraction": effective,
        "diagonal_mass": diagonal,
        "bonded_mass": bonded,
        "two_hop_mass": distance_two,
        "nonlocal_mass": nonlocal_mass,
    }
    if not torch.allclose(assignment.sum((1, 2)), torch.ones(b, device=assignment.device), atol=1e-5, rtol=0):
        raise RuntimeError("PairToken assignment mass is not one")
    if any(not bool(torch.isfinite(x).all()) for x in result.values()):
        raise RuntimeError("Nonfinite assignment statistic")
    if not bool(torch.allclose(diagonal + bonded + distance_two + nonlocal_mass,
                               torch.ones_like(diagonal), atol=1e-5, rtol=0)):
        raise RuntimeError("Topology pair categories are not a partition")
    return result


def _forward_with_assignment(model, graph_batch):
    hidden = model.base._embed_nodes(graph_batch.x)
    hidden = hidden + model.base.rwse_encoder(graph_batch.random_walk_pe.float())
    edge_state = model.base._embed_edges(graph_batch.edge_attr)
    selected = None
    for layer, (edge_update, block) in enumerate(
        zip(model.base.edge_updates, model.base.local_blocks), start=1
    ):
        edge_state = edge_update(hidden, graph_batch.edge_index, edge_state)
        hidden = block(hidden, graph_batch.edge_index, graph_batch.batch,
                       edge_attr=edge_state)
        if str(layer) in model.base.neural_atom_mixers:
            hidden = model.base.neural_atom_mixers[str(layer)](hidden, graph_batch.batch)
        if layer == 6:
            update, details = model.relation_token.compute_update(hidden, graph_batch.batch)
            hidden = hidden + update
            selected = attention_metrics(details["assignment"], details["valid"],
                                         graph_batch.edge_index, graph_batch.batch)
    prediction = model.base.head(model.base._pool(hidden, graph_batch.batch)).view(-1)
    return prediction, selected


def _read_payload(path, expected_sha):
    import torch

    if sha256_file(path) != expected_sha:
        raise RuntimeError(f"Frozen payload changed: {path.name}")
    return torch.load(path, map_location="cpu")


def _write_chunk(path, arrays):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _summarize(values):
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


def run_diagnostic(cache_root: Path, checkpoint: Path, pair_payload: Path,
                   k1_payload: Path, output_root: Path) -> dict:
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    from .pcqm_k1_variants import make_encoder

    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen PairToken checkpoint changed")
    pair_reference = _read_payload(pair_payload, EXPECTED_PAYLOAD_SHA256)
    k1_reference = _read_payload(k1_payload, K1_PAYLOAD_SHA256)
    for field in ("source_idx", "target_eV"):
        if not torch.equal(pair_reference[field].view(-1), k1_reference[field].view(-1)):
            raise RuntimeError(f"Frozen payload {field} differs")
    expected_sources = torch.arange(100_000, 150_000)
    if not torch.equal(pair_reference["source_idx"].view(-1).long(), expected_sources):
        raise RuntimeError("Development source order differs")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    roles, manifest, mean, std = _load_contract(cache_root)
    if len(roles["development"]) != DEVELOPMENT_ROWS:
        raise RuntimeError("Development row count differs")
    model = make_encoder("neural_atom_k1_pair_token").to("cuda").eval()
    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("Model parameter count differs")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    identities = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "pair_payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "k1_payload_sha256": K1_PAYLOAD_SHA256,
        "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "implementation_sha256": sha256_file(Path(__file__)),
        "pair_token_implementation_sha256": sha256_file(
            Path(__file__).with_name("k1_pair_token.py")
        ),
    }
    progress = {"format": "molgap-k1-pair-selection-progress-v1",
                "input_identity": identities, "chunks": []}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("input_identity") != identities:
            raise RuntimeError("Resume input or implementation identity changed")
    completed = {int(x["start"]): x for x in progress["chunks"]}
    for start in range(0, DEVELOPMENT_ROWS, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, DEVELOPMENT_ROWS)
        chunk_path = output_root / f"rows_{start:05d}_{end:05d}.npz"
        previous = completed.get(start)
        if previous:
            if previous["end"] != end or sha256_file(chunk_path) != previous["sha256"]:
                raise RuntimeError(f"Completed chunk changed: {start}")
            continue
        if chunk_path.exists():
            raise RuntimeError(f"Unregistered chunk exists: {chunk_path.name}")
        loader = DataLoader(Subset(roles["development"], range(start, end)),
                            batch_size=128, shuffle=False, drop_last=False,
                            num_workers=0, pin_memory=True)
        parts = {name: [] for name in METRICS}
        actual_pred = []
        actual_target = []
        actual_source = []
        with torch.no_grad():
            for graph_batch in loader:
                graph_batch = graph_batch.to("cuda", non_blocking=True)
                normalized, assignment = _forward_with_assignment(model, graph_batch)
                prediction = (normalized * std + mean).float().cpu()
                source = graph_batch.source_idx.view(-1).long().cpu()
                target = graph_batch.y.view(-1).float().cpu()
                offsets = (source - 100_000).numpy()
                k1_prediction = k1_reference["prediction_eV"].view(-1)[offsets].float()
                pair_prediction = pair_reference["prediction_eV"].view(-1)[offsets].float()
                if not torch.equal(target, pair_reference["target_eV"].view(-1)[offsets].float()):
                    raise RuntimeError("Target mismatch against frozen payload")
                descriptor = _descriptors(graph_batch)
                for name in assignment:
                    parts[name].append(assignment[name].float().cpu().numpy())
                for name in ("atom_count", "conjugated_bond_fraction"):
                    parts[name].append(descriptor[name].numpy())
                parts["k1_absolute_error"].append((k1_prediction - target).abs().numpy())
                parts["paired_gain_eV"].append(
                    ((k1_prediction - target).abs() - (pair_prediction - target).abs()).numpy()
                )
                actual_pred.append(prediction)
                actual_target.append(target)
                actual_source.append(source)
        sources = torch.cat(actual_source)
        expected = torch.arange(100_000 + start, 100_000 + end)
        if not torch.equal(sources, expected):
            raise RuntimeError("Processed source order differs")
        predictions = torch.cat(actual_pred)
        frozen = pair_reference["prediction_eV"].view(-1)[start:end].float()
        target = torch.cat(actual_target)
        maximum = float((predictions - frozen).abs().max())
        mae_difference = abs(float((predictions - target).abs().mean()) -
                             float((frozen - target).abs().mean()))
        if maximum > 5e-4 or mae_difference > 5e-6:
            raise RuntimeError(f"Frozen prediction reproduction failed: {maximum}, {mae_difference}")
        arrays = {name: np.concatenate(parts[name]).astype(np.float32) for name in METRICS}
        _write_chunk(chunk_path, arrays)
        previous = {"start": start, "end": end, "sha256": sha256_file(chunk_path),
                    "prediction_max_absolute_difference_eV": maximum,
                    "mae_absolute_difference_eV": mae_difference}
        progress["chunks"].append(previous)
        _atomic_json(progress_path, progress)
        completed[start] = previous

    arrays = {name: [] for name in METRICS}
    for start in range(0, DEVELOPMENT_ROWS, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, DEVELOPMENT_ROWS)
        with np.load(output_root / f"rows_{start:05d}_{end:05d}.npz") as chunk:
            for name in METRICS:
                arrays[name].append(chunk[name])
    arrays = {name: np.concatenate(parts).astype(np.float64)
              for name, parts in arrays.items()}
    if any(len(x) != DEVELOPMENT_ROWS or not np.isfinite(x).all() for x in arrays.values()):
        raise RuntimeError("Incomplete or nonfinite diagnostic arrays")
    hard_cut = np.quantile(arrays["k1_absolute_error"], 0.9)
    masks = {
        "all": np.ones(DEVELOPMENT_ROWS, dtype=bool),
        "more_than_12_atoms": arrays["atom_count"] > 12,
        "hardest_k1_error_decile": arrays["k1_absolute_error"] >= hard_cut,
        "pairtoken_improved": arrays["paired_gain_eV"] > 0,
        "pairtoken_regressed": arrays["paired_gain_eV"] < 0,
    }
    strata = {
        name: {"rows": int(mask.sum()), **{metric: _summarize(arrays[metric][mask])
               for metric in METRICS if metric not in ("atom_count", "k1_absolute_error")}}
        for name, mask in masks.items()
    }
    concentration = arrays["top20_mass"]
    gate = {
        "median_top20_at_least_0_80": bool(np.median(concentration) >= 0.8),
        "fraction_at_least_0_80_ge_0_70": bool(np.mean(concentration >= 0.8) >= 0.7),
        "more_than_12_atoms_median_at_least_0_80": bool(
            np.median(concentration[masks["more_than_12_atoms"]]) >= 0.8
        ),
        "hardest_k1_decile_median_at_least_0_80": bool(
            np.median(concentration[masks["hardest_k1_error_decile"]]) >= 0.8
        ),
    }
    result = {
        "format": "molgap-k1-pair-selection-diagnostic-v1",
        "complete": True,
        "training_executed": False,
        "model_inference_executed": True,
        "development_rows": DEVELOPMENT_ROWS,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "promotion_authorized": False,
        "candidate_question_qualified": all(gate.values()),
        "gate": gate,
        "strata": strata,
        "checkpoint_sha256": sha256_file(checkpoint),
        "pair_payload_sha256": sha256_file(pair_payload),
        "k1_payload_sha256": sha256_file(k1_payload),
        "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "geometry_aggregate_sha256": manifest["geometry_aggregate_sha256"],
        "chunk_sha256": {x["start"]: x["sha256"] for x in progress["chunks"]},
    }
    _atomic_json(output_root / "diagnostic.json", result)
    return result
