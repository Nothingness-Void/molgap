"""Fixed official-validation evaluation for the frozen PCQM Route B fusion."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .gps import GPSWrapper
from .pcqm_route_b import build_route_b_row
from .pcqm_route_b_fusion import (
    ENCODER_NAMES,
    FrozenGapReadout,
    PCQMBoundedFusion,
)
from .pcqm_route_b_training import atomic_json, atomic_torch, sha256_file
from .schnet import SchNetWrapper


EXPECTED_LABEL_ROWS = 5_000
EXPECTED_ACCEPTED_ROWS = 4_981
GINE_V7_FIXED_VALID_GAP_MAE_EV = 0.1846183411


@dataclass(frozen=True)
class OfficialValidConfig:
    graph_shard_rows: int = 500
    workers: int = 12
    gps_batch_size: int = 256
    schnet_batch_size: int = 128
    fusion_batch_size: int = 4096


def _install_portable_radius_if_needed() -> str:
    try:
        import torch_cluster

        return getattr(torch_cluster, "BACKEND", "torch_cluster_extension")
    except ImportError:
        from . import portable_radius

        sys.modules["torch_cluster"] = portable_radius
        return portable_radius.BACKEND


def _validate_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"idx", "smiles", "gap_true"}
    if set(frame.columns) != required:
        raise RuntimeError(
            f"Official-valid columns differ: {list(frame.columns)}"
        )
    if (
        len(frame) != EXPECTED_LABEL_ROWS
        or frame["idx"].duplicated().any()
        or not np.isfinite(frame["gap_true"].to_numpy(dtype=np.float64)).all()
    ):
        raise RuntimeError("Official-valid identity or target contract differs")
    return frame


def _graph_contract(labels_path: Path, config: OfficialValidConfig) -> dict:
    return {
        "format": "molgap-pcqm-route-b-official-valid-graphs-v1",
        "labels_sha256": sha256_file(labels_path),
        "source_rows": EXPECTED_LABEL_ROWS,
        "expected_accepted_rows": EXPECTED_ACCEPTED_ROWS,
        "graph_shard_rows": config.graph_shard_rows,
        "conformer_contract": {
            "method": "ETKDGv3+MMFF200",
            "primary_seed": 42,
            "secondary_seed": 43,
            "secondary_built_for_acceptance_only": True,
        },
    }


def build_official_valid_graph_cache(
    *,
    labels_path: Path,
    cache_dir: Path,
    config: OfficialValidConfig | None = None,
) -> dict:
    """Build atomic aligned GPS/primary shards using the training graph contract."""
    config = config or OfficialValidConfig()
    frame = _validate_labels(labels_path)
    contract = _graph_contract(labels_path, config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = {
        **contract,
        "status": "building",
        "processed_rows": 0,
        "accepted_rows": 0,
        "failed_source_idx": [],
        "shards": [],
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in contract.items():
            if existing.get(key) != value:
                raise RuntimeError(f"Graph-cache contract differs at {key}")
        manifest = existing
        if manifest.get("status") == "complete":
            return manifest

    start = int(manifest["processed_rows"])
    expected_start = len(manifest["shards"]) * config.graph_shard_rows
    if start != expected_start:
        raise RuntimeError("Graph-cache resume boundary differs")

    context = mp.get_context("spawn")
    with context.Pool(processes=config.workers) as pool:
        for begin in range(start, len(frame), config.graph_shard_rows):
            end = min(begin + config.graph_shard_rows, len(frame))
            rows = [
                (
                    int(row.idx),
                    str(row.smiles),
                    float(row.gap_true),
                    2,
                )
                for row in frame.iloc[begin:end].itertuples(index=False)
            ]
            gps_graphs, primary_graphs, failures = [], [], []
            for source_idx, result in pool.imap(
                build_route_b_row, rows, chunksize=10
            ):
                if result is None:
                    failures.append(int(source_idx))
                    continue
                gps, primary, _secondary = result
                gps_graphs.append(gps)
                primary_graphs.append(primary)
            if len(gps_graphs) != len(primary_graphs):
                raise RuntimeError("Official-valid graph modalities misaligned")
            payload = {
                "format": "molgap-pcqm-route-b-official-valid-shard-v1",
                "begin": begin,
                "end": end,
                "gps": gps_graphs,
                "primary": primary_graphs,
                "source_idx": torch.tensor(
                    [int(graph.source_idx.item()) for graph in gps_graphs],
                    dtype=torch.long,
                ),
                "targets": torch.tensor(
                    [float(graph.y.item()) for graph in gps_graphs],
                    dtype=torch.float32,
                ),
                "failed_source_idx": failures,
            }
            path = cache_dir / f"official_{begin:05d}_{end:05d}.pt"
            atomic_torch(path, payload)
            manifest["processed_rows"] = end
            manifest["accepted_rows"] += len(gps_graphs)
            manifest["failed_source_idx"].extend(failures)
            manifest["shards"].append(
                {
                    "path": path.name,
                    "begin": begin,
                    "end": end,
                    "rows": len(gps_graphs),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
            atomic_json(manifest_path, manifest)
            print(
                f"official-valid graph shard {begin}:{end} "
                f"accepted={len(gps_graphs)} failed={len(failures)}",
                flush=True,
            )

    if (
        manifest["processed_rows"] != EXPECTED_LABEL_ROWS
        or manifest["accepted_rows"] != EXPECTED_ACCEPTED_ROWS
        or len(manifest["failed_source_idx"])
        != EXPECTED_LABEL_ROWS - EXPECTED_ACCEPTED_ROWS
    ):
        raise RuntimeError(
            "Official-valid accepted-row contract differs: "
            f"{manifest['accepted_rows']} != {EXPECTED_ACCEPTED_ROWS}"
        )
    manifest["status"] = "complete"
    atomic_json(manifest_path, manifest)
    return manifest


def _load_graph_cache(cache_dir: Path) -> tuple[list, list, torch.Tensor, torch.Tensor]:
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("accepted_rows") != EXPECTED_ACCEPTED_ROWS
    ):
        raise RuntimeError("Official-valid graph cache is not accepted")
    gps_graphs, primary_graphs, indices, targets = [], [], [], []
    for item in manifest["shards"]:
        path = cache_dir / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise RuntimeError(f"Official-valid graph shard differs: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        gps_graphs.extend(payload["gps"])
        primary_graphs.extend(payload["primary"])
        indices.append(payload["source_idx"].view(-1).long())
        targets.append(payload["targets"].view(-1).float())
    source_idx = torch.cat(indices)
    target = torch.cat(targets)
    if (
        len(source_idx) != EXPECTED_ACCEPTED_ROWS
        or len(source_idx.unique()) != len(source_idx)
        or len(gps_graphs) != len(source_idx)
        or len(primary_graphs) != len(source_idx)
        or not torch.isfinite(target).all()
    ):
        raise RuntimeError("Official-valid graph-cache alignment differs")
    return gps_graphs, primary_graphs, source_idx, target


def _verify_manifest_artifacts(root: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Incomplete artifact manifest: {manifest_path}")
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise RuntimeError(f"Artifact differs: {path}")
    return manifest


def verify_downloaded_fusion(
    *, payload_dir: Path, fusion_dir: Path
) -> tuple[dict[str, Path], list[Path], dict]:
    """Verify encoder and selected fusion artifacts before opening valid labels."""
    payload_manifest = _verify_manifest_artifacts(
        payload_dir, payload_dir / "manifest.json"
    )
    fusion_manifest = _verify_manifest_artifacts(
        fusion_dir, fusion_dir / "completion_manifest.json"
    )
    selection_path = fusion_dir / "development_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("status") != "complete"
        or selection.get("selection_scope") != "scaffold-development-only"
        or selection.get("selected_base_identity") != "augmented_schnet"
        or selection.get("official_valid_metric_read") is not False
        or fusion_manifest.get("selected_base_identity") != "augmented_schnet"
    ):
        raise RuntimeError("Frozen development selection contract differs")
    checkpoint_paths = {
        name: payload_dir / "checkpoints" / f"{name}_best.pt"
        for name in ENCODER_NAMES
    }
    fusion_paths = [
        fusion_dir / "augmented_schnet" / f"seed_{seed}" / "best.pt"
        for seed in selection["seeds"]
    ]
    if any(not path.is_file() for path in fusion_paths):
        raise RuntimeError("Selected fusion seed artifact is missing")
    return checkpoint_paths, fusion_paths, selection


def _load_encoder(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["kind"] == "gps":
        overrides = checkpoint.get("training_overrides") or {}
        model = GPSWrapper(
            in_channels=18,
            edge_dim=4,
            hidden_channels=int(config["hidden_channels"]),
            num_layers=int(config["num_layers"]),
            num_heads=4,
            dropout=float(overrides.get("dropout", 0.05)),
            n_targets=3,
        )
    else:
        model_config = checkpoint["warm_start_report"]["model_config"]
        model = SchNetWrapper(**model_config, use_charges=True, n_targets=3)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def _encode_gps(models: dict, graphs: list, batch_size: int, device: torch.device):
    output = {name: [] for name in models}
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            for name, model in models.items():
                value = model.encode(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                output[name].append(value.to(torch.float16).cpu())
    return {name: torch.cat(parts) for name, parts in output.items()}


@torch.no_grad()
def _encode_schnet(
    models: dict, graphs: list, batch_size: int, device: torch.device
):
    output = {name: [] for name in models}
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            for name, model in models.items():
                value = model.encode(
                    batch.z,
                    batch.pos,
                    batch.batch,
                    charges=batch.charges,
                )
                output[name].append(value.to(torch.float16).cpu())
    return {name: torch.cat(parts) for name, parts in output.items()}


@torch.no_grad()
def evaluate_official_valid(
    *,
    labels_path: Path,
    payload_dir: Path,
    fusion_dir: Path,
    graph_cache_dir: Path,
    output_dir: Path,
    config: OfficialValidConfig | None = None,
) -> dict:
    """Evaluate the frozen development-selected fusion once on official valid."""
    config = config or OfficialValidConfig()
    checkpoint_paths, fusion_paths, selection = verify_downloaded_fusion(
        payload_dir=payload_dir, fusion_dir=fusion_dir
    )
    graph_manifest = build_official_valid_graph_cache(
        labels_path=labels_path,
        cache_dir=graph_cache_dir,
        config=config,
    )
    gps_graphs, primary_graphs, source_idx, target = _load_graph_cache(
        graph_cache_dir
    )
    labels = _validate_labels(labels_path).set_index("idx")
    expected_target = torch.tensor(
        labels.loc[source_idx.numpy(), "gap_true"].to_numpy(),
        dtype=torch.float32,
    )
    if not torch.allclose(target, expected_target, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("Graph targets differ from fixed official-valid labels")

    if not torch.cuda.is_available():
        raise RuntimeError("Official-valid Route B evaluation requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    radius_backend = _install_portable_radius_if_needed()
    device = torch.device("cuda")
    models, checkpoints = {}, {}
    for name, path in checkpoint_paths.items():
        models[name], checkpoints[name] = _load_encoder(path, device)
    embeddings = _encode_gps(
        {name: models[name] for name in ("gps9", "gps11_160")},
        gps_graphs,
        config.gps_batch_size,
        device,
    )
    embeddings.update(
        _encode_schnet(
            {
                name: models[name]
                for name in ("primary_schnet", "augmented_schnet")
            },
            primary_graphs,
            config.schnet_batch_size,
            device,
        )
    )
    del models
    torch.cuda.empty_cache()

    component_predictions = {}
    for name in ENCODER_NAMES:
        readout = FrozenGapReadout(checkpoints[name]).to(device).eval()
        component_predictions[name] = (
            readout(embeddings[name].float().to(device)).cpu()
        )

    seed_predictions = {}
    for path in fusion_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("base_name") != "augmented_schnet":
            raise RuntimeError(f"Unexpected Fusion identity: {path}")
        fusion = PCQMBoundedFusion(
            checkpoints["augmented_schnet"],
            base_name="augmented_schnet",
            hidden=int(checkpoint["config"]["hidden"]),
            correction_scale_eV=float(
                checkpoint["config"]["correction_scale_eV"]
            ),
        ).to(device)
        fusion.load_state_dict(checkpoint["model"], strict=True)
        fusion.eval()
        chunks = []
        for begin in range(0, len(target), config.fusion_batch_size):
            index = slice(begin, begin + config.fusion_batch_size)
            values = {
                name: embeddings[name][index].float().to(device)
                for name in ENCODER_NAMES
            }
            chunks.append(fusion(values).cpu())
        seed_predictions[str(checkpoint["seed"])] = torch.cat(chunks)
    ensemble = torch.stack(list(seed_predictions.values())).mean(dim=0)

    def mae(prediction: torch.Tensor) -> float:
        return float((prediction - target).abs().mean())

    metrics = {
        "format": "molgap-pcqm-route-b-official-valid-evaluation-v1",
        "status": "complete",
        "selection_scope": selection["selection_scope"],
        "selected_base_identity": selection["selected_base_identity"],
        "n_source": EXPECTED_LABEL_ROWS,
        "n_valid": len(target),
        "gap_mae_eV": {
            "components": {
                name: mae(prediction)
                for name, prediction in component_predictions.items()
            },
            "fusion_seeds": {
                seed: mae(prediction)
                for seed, prediction in seed_predictions.items()
            },
            "fusion_equal_seed_ensemble": mae(ensemble),
            "gine_v7_fixed_valid_reference": GINE_V7_FIXED_VALID_GAP_MAE_EV,
        },
        "fusion_ensemble_minus_gine_v7_eV": (
            mae(ensemble) - GINE_V7_FIXED_VALID_GAP_MAE_EV
        ),
        "passes_track_b_gate": mae(ensemble)
        < GINE_V7_FIXED_VALID_GAP_MAE_EV,
        "artifacts": {
            "labels_sha256": sha256_file(labels_path),
            "payload_manifest_sha256": sha256_file(
                payload_dir / "manifest.json"
            ),
            "fusion_manifest_sha256": sha256_file(
                fusion_dir / "completion_manifest.json"
            ),
            "development_selection_sha256": sha256_file(
                fusion_dir / "development_selection.json"
            ),
            "graph_manifest_sha256": sha256_file(
                graph_cache_dir / "manifest.json"
            ),
            "encoder_checkpoints": {
                name: sha256_file(path)
                for name, path in checkpoint_paths.items()
            },
            "fusion_checkpoints": {
                path.parent.name: sha256_file(path) for path in fusion_paths
            },
        },
        "graph_contract": graph_manifest["conformer_contract"],
        "radius_backend": radius_backend,
        "official_valid_metric_read": True,
        "official_test_used": False,
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = labels.loc[source_idx.numpy()].reset_index()[
        ["idx", "smiles", "gap_true"]
    ]
    for name, prediction in component_predictions.items():
        prediction_frame[f"{name}_gap"] = prediction.numpy()
    for seed, prediction in seed_predictions.items():
        prediction_frame[f"fusion_seed_{seed}_gap"] = prediction.numpy()
    prediction_frame["fusion_equal_seed_ensemble_gap"] = ensemble.numpy()
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    atomic_json(output_dir / "metrics.json", metrics)
    return metrics
