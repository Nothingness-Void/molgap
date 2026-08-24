"""Paired external evaluation for the repaired-2M hierarchical Fusion."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .graphs import smiles_to_pyg
from .hierarchical_fusion import (
    ConservativeFusionConfig,
    ConservativeHierarchicalResidualHead,
    HierarchicalBoundedResidualHead,
    HierarchicalFusionConfig,
    hierarchical_context,
    predict_conservative_hierarchical_fusion,
    predict_hierarchical_fusion,
)
from .multi2d_router_fusion import (
    DenseSoftGate,
    GateTrainingConfig,
    dense_gate_features,
    predict_dense_gate,
)
from .router import paired_bootstrap_mean
from .schnet import SchNetWrapper


TARGETS = ("homo", "lumo", "gap")
SCHNET_CONFIG = {
    "hidden_channels": 176,
    "num_filters": 160,
    "num_interactions": 6,
    "num_gaussians": 50,
    "cutoff": 10.0,
    "dropout": 0.05,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _build_graph(item: tuple[int, str, tuple[float, ...]]) -> tuple[int, object | None]:
    row, smiles, target = item
    seed = int((42 * 1_000_003 + row * 97) % 2_147_483_647)
    graph = smiles_to_pyg(
        smiles,
        use_charges=True,
        mmff_iters=200,
        max_embed_attempts=2,
        random_seed=seed,
    )
    if graph is not None:
        graph.source_idx = torch.tensor([row], dtype=torch.long)
        graph.y = torch.tensor(target, dtype=torch.float32).view(1, 3)
    return row, graph


def build_external_graph_cache(
    table: pd.DataFrame,
    cache_dir: Path,
    *,
    shard_size: int = 100,
    workers: int = 8,
) -> dict:
    """Build deterministic ETKDGv3+MMFF200 shards with atomic resume."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for start in range(0, len(table), shard_size):
        stop = min(start + shard_size, len(table))
        graph_path = cache_dir / f"graphs_{start:07d}_{stop:07d}.pt"
        report_path = graph_path.with_suffix(".json")
        if graph_path.is_file() and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("sha256") == sha256(graph_path):
                reports.append(report)
                continue
        work = [
            (
                row,
                str(table.iloc[row].smiles),
                tuple(float(table.iloc[row][target]) for target in TARGETS),
            )
            for row in range(start, stop)
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            built = list(pool.map(_build_graph, work, chunksize=4))
        graphs = [graph for _, graph in built if graph is not None]
        failed = [row for row, graph in built if graph is None]
        _atomic_torch(graphs, graph_path)
        report = {
            "status": "complete",
            "start": start,
            "stop": stop,
            "requested_rows": stop - start,
            "accepted_rows": len(graphs),
            "failed_source_idx": failed,
            "path": graph_path.name,
            "sha256": sha256(graph_path),
        }
        _atomic_json(report, report_path)
        reports.append(report)
        print(
            f"graph shard {start:,}:{stop:,} "
            f"accepted={len(graphs):,} failed={len(failed):,}",
            flush=True,
        )
    accepted = sum(int(report["accepted_rows"]) for report in reports)
    completion = {
        "status": "complete",
        "protocol": "ETKDGv3+MMFF200",
        "source_rows": len(table),
        "accepted_rows": accepted,
        "failed_rows": len(table) - accepted,
        "parts": len(reports),
        "reports": reports,
    }
    _atomic_json(completion, cache_dir / "completion.json")
    return completion


def _aligned_table(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    keys = ["eval_set", "cid"]
    if reference.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise ValueError(f"{name} contains duplicate eval_set/CID identities")
    indexed = candidate.set_index(keys)
    index = pd.MultiIndex.from_frame(reference[keys])
    missing = index.difference(indexed.index)
    if len(missing):
        raise ValueError(f"{name} misses {len(missing):,} reference rows")
    aligned = indexed.loc[index].reset_index()
    if not np.array_equal(
        aligned["smiles"].astype(str).to_numpy(),
        reference["smiles"].astype(str).to_numpy(),
    ):
        raise ValueError(f"{name} SMILES identities differ")
    if not np.allclose(
        aligned.loc[:, TARGETS].to_numpy(np.float64),
        reference.loc[:, TARGETS].to_numpy(np.float64),
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError(f"{name} targets differ")
    return aligned


def load_paired_prediction_tables(
    *,
    two_m_predictions: Path,
    routed_v4_predictions: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    two_m = pd.read_csv(two_m_predictions)
    reference = two_m.loc[:, ["eval_set", "cid", "smiles", *TARGETS]].copy()
    routed = _aligned_table(reference, pd.read_csv(routed_v4_predictions), "routed v4")
    experts = np.stack(
        [
            two_m.loc[
                :, [f"{expert}_{target}" for target in TARGETS]
            ].to_numpy(np.float32)
            for expert in ("gps7", "gps9", "gps11_160")
        ],
        axis=1,
    )
    routed_prediction = np.stack(
        [
            routed[f"routed_v4_500k_{target}"].to_numpy(np.float32)
            for target in TARGETS
        ],
        axis=1,
    )
    for name, values in (
        ("2M expert", experts),
        ("routed v4", routed_prediction),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} predictions contain non-finite values")
    return reference, experts, routed_prediction


def load_dense_ensemble(
    paths: list[Path],
    experts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictions, weights = [], []
    features = dense_gate_features(experts)
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = GateTrainingConfig(**payload["config"])
        state = payload["state_dict"]
        model = DenseSoftGate(
            features.shape[1],
            state["feature_mean"].numpy(),
            state["feature_std"].numpy(),
            config,
        )
        model.load_state_dict(state)
        prediction, weight = predict_dense_gate(model.eval(), experts)
        predictions.append(prediction)
        weights.append(weight)
    return (
        np.mean(predictions, axis=0).astype(np.float32),
        np.mean(weights, axis=0).astype(np.float32),
    )


@torch.inference_mode()
def extract_external_schnet_embeddings(
    *,
    cache_dir: Path,
    primary_checkpoint: Path,
    augmented_checkpoint: Path,
    device: str,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    models = []
    for checkpoint in (primary_checkpoint, augmented_checkpoint):
        model = SchNetWrapper(**SCHNET_CONFIG, use_charges=True).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        models.append(model.eval())
    source_indices, primary, augmented = [], [], []
    for path in sorted(cache_dir.glob("graphs_*.pt")):
        graphs = torch.load(path, map_location="cpu", weights_only=False)
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            source_indices.append(batch.source_idx.view(-1).cpu())
            for model, output in zip(models, (primary, augmented)):
                output.append(
                    model.encode(
                        batch.z,
                        batch.pos,
                        batch.batch,
                        charges=getattr(batch, "charges", None),
                    ).float().cpu()
                )
    source_idx = torch.cat(source_indices).numpy().astype(np.int64)
    if len(np.unique(source_idx)) != len(source_idx):
        raise ValueError("External graph cache contains duplicate source_idx")
    order = np.argsort(source_idx)
    return (
        source_idx[order],
        torch.cat(primary).numpy()[order],
        torch.cat(augmented).numpy()[order],
    )


def load_hierarchical_ensemble(paths: list[Path]) -> list[torch.nn.Module]:
    models = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        if payload.get("kind") == "conservative_hierarchical_2d_3d_residual":
            config = ConservativeFusionConfig(**payload["config"])
            model = ConservativeHierarchicalResidualHead(
                len(state["feature_center"]),
                state["feature_center"].numpy(),
                state["feature_scale"].numpy(),
                config,
            )
        else:
            config = HierarchicalFusionConfig(**payload["config"])
            model = HierarchicalBoundedResidualHead(
                len(state["feature_mean"]),
                state["feature_mean"].numpy(),
                state["feature_std"].numpy(),
                config,
            )
        model.load_state_dict(state)
        models.append(model.eval())
    return models


def _metric_block(targets: np.ndarray, prediction: np.ndarray) -> dict:
    errors = np.abs(prediction - targets)
    return {
        **{
            target: {"mae_eV": float(errors[:, index].mean())}
            for index, target in enumerate(TARGETS)
        },
        "average": {"mae_eV": float(errors.mean())},
    }


def _paired_delta(
    targets: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict:
    result = {}
    for index, target in enumerate((*TARGETS, "average")):
        if target == "average":
            delta = np.abs(candidate - targets).mean(axis=1) - np.abs(
                baseline - targets
            ).mean(axis=1)
        else:
            delta = np.abs(candidate[:, index] - targets[:, index]) - np.abs(
                baseline[:, index] - targets[:, index]
            )
        result[target] = paired_bootstrap_mean(
            delta,
            n_bootstrap=10_000,
            seed=42 + index,
        )
    return result


def evaluate_paired_external(
    *,
    reference: pd.DataFrame,
    experts: np.ndarray,
    routed_v4: np.ndarray,
    source_idx: np.ndarray,
    primary_embeddings: np.ndarray,
    augmented_embeddings: np.ndarray,
    dense_gate_paths: list[Path],
    hierarchical_root: Path,
) -> tuple[dict, pd.DataFrame]:
    dense, dense_weights = load_dense_ensemble(dense_gate_paths, experts)
    equal = experts[:, :2].mean(axis=1)
    context = hierarchical_context(
        experts[source_idx],
        dense_weights[source_idx],
        primary_embeddings,
        augmented_embeddings,
    )
    methods = {
        "routed_v4_500k": routed_v4[source_idx],
        "repaired_2m_equal_2d": equal[source_idx],
        "repaired_2m_dense_2d": dense[source_idx],
    }
    for base_name, base in (
        ("equal", equal[source_idx]),
        ("dense", dense[source_idx]),
    ):
        models = load_hierarchical_ensemble(
            [
                hierarchical_root / f"{'equal_gps7_gps9' if base_name == 'equal' else 'dense'}_seed{seed}.pt"
                for seed in (42, 43, 44)
            ]
        )
        predictions = []
        for model in models:
            if isinstance(model, ConservativeHierarchicalResidualHead):
                prediction = predict_conservative_hierarchical_fusion(
                    model, base, context
                )[0]
            else:
                prediction = predict_hierarchical_fusion(model, base, context)[0]
            predictions.append(prediction)
        methods[f"repaired_2m_{base_name}_dual_schnet"] = np.mean(
            predictions, axis=0
        )
    aligned = reference.iloc[source_idx].reset_index(drop=True)
    targets = aligned.loc[:, TARGETS].to_numpy(np.float32)
    scopes = {
        "all": np.ones(len(aligned), dtype=bool),
        "ood1000": aligned.eval_set.eq("ood1000").to_numpy(),
        "p8_targeted_hard": aligned.eval_set.eq("p8_targeted_hard").to_numpy(),
    }
    metrics = {}
    for scope, mask in scopes.items():
        metrics[scope] = {
            "rows": int(mask.sum()),
            "methods": {
                name: _metric_block(targets[mask], prediction[mask])
                for name, prediction in methods.items()
            },
            "delta_vs_routed_v4_500k": {
                name: _paired_delta(
                    targets[mask],
                    methods["routed_v4_500k"][mask],
                    prediction[mask],
                )
                for name, prediction in methods.items()
                if name != "routed_v4_500k"
            },
            "fusion_delta_vs_own_2d": {
                identity: _paired_delta(
                    targets[mask],
                    methods[f"repaired_2m_{identity}_2d"][mask],
                    methods[f"repaired_2m_{identity}_dual_schnet"][mask],
                )
                for identity in ("equal", "dense")
            },
        }
    output = aligned.copy()
    for name, prediction in methods.items():
        for index, target in enumerate(TARGETS):
            output[f"{name}_{target}"] = prediction[:, index]
    return metrics, output
