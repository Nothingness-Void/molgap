"""Local ETKDG geometry supervision for the pure-2D PCQM EdgeState encoder."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .pcqm_local_hierarchy import (
    PARENT_GEOMETRY_CACHE_SHA256,
    PARENT_GRAPH_CACHE_SHA256,
    _atomic_torch_save,
    _find_cache,
    _state_sha256,
    _torch_load,
    _verify_parent_manifest,
    atomic_json,
    sha256_file,
)


MODEL_SEED = 42
BATCH_SIZE = 96
LEARNING_RATE = 1.6e-4
WEIGHT_DECAY = 1e-6
SCRATCH_EPOCHS = 40
PRETRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 20
EXPECTED_MODEL_PARAMETERS = 4_771_073
MIN_PAIRED_GAIN_EV = 0.003
TRAIN_GRAPHS = 100_000
VALIDATION_GRAPHS = 10_000
DISTANCE_CENTER_ANGSTROM = 1.5
DISTANCE_SCALE_ANGSTROM = 0.3


def _set_seed(seed: int) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng_state(shuffle_generator) -> dict:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "shuffle_generator": shuffle_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict, shuffle_generator) -> None:
    import torch

    required = {"python", "numpy", "torch", "shuffle_generator"}
    if not required.issubset(state):
        raise RuntimeError("Resume checkpoint is missing RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    shuffle_generator.set_state(state["shuffle_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _make_encoder():
    from .pcqm_gap_architecture import make_pcqm_gap_encoder

    return make_pcqm_gap_encoder("ogb_edge_state_structural_gps9")


def _forward(model, batch):
    # Geometry is deliberately excluded: the accepted inference architecture
    # remains a pure-2D EdgeState GPS9 after the training-only heads are removed.
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    ).view(-1)


class LocalGeometryHeads:
    """Training-only bond-length and bonded-angle heads."""

    def __init__(self, model):
        import torch.nn as nn

        self.node_state = None
        self.edge_state = None
        self.distance_head = nn.Sequential(
            nn.LayerNorm(64), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.angle_head = nn.Sequential(
            nn.LayerNorm(320), nn.Linear(320, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        self.module = nn.ModuleDict(
            {"distance_head": self.distance_head, "angle_head": self.angle_head}
        )
        self.handles = [
            model.convs[-1].register_forward_hook(self._capture_node),
            model.edge_updates[-1].register_forward_hook(self._capture_edge),
        ]

    def _capture_node(self, _module, _inputs, output):
        self.node_state = output

    def _capture_edge(self, _module, _inputs, output):
        self.edge_state = output

    def predictions(self, batch):
        import torch

        first, second = batch.wedge_edge_ids.unbind(dim=1)
        centers = batch.edge_index[1, first]
        distance = self.distance_head(self.edge_state).view(-1)
        angle_context = torch.cat(
            [self.edge_state[first], self.node_state[centers], self.edge_state[second]],
            dim=-1,
        )
        angle = self.angle_head(angle_context).view(-1)
        return distance, angle, centers

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _loader(graphs, *, shuffle: bool, generator=None):
    import torch
    from torch_geometric.loader import DataLoader

    if generator is None:
        generator = torch.Generator().manual_seed(MODEL_SEED)
    return DataLoader(
        graphs,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )


def load_roles(graph_root: Optional[Path] = None):
    """Load the accepted geometry cache while retaining all frozen row roles."""
    graph_root, manifest = _find_cache(
        "molgap-pcqm-gap100k-etkdg-geometry-cache-v1", graph_root
    )
    _verify_parent_manifest(manifest)
    roles = {"train": [], "validation": []}
    for item in manifest["shards"]:
        path = graph_root / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Geometry shard hash changed: {item['file']}")
        graphs = _torch_load(path, map_location="cpu", weights_only=False)
        if len(graphs) != int(item["graph_count"]):
            raise RuntimeError(f"Geometry shard count changed: {item['file']}")
        for graph in graphs:
            graph.row_id = graph.row_index.clone()
            required = {
                "edge_distance",
                "wedge_angle_cos",
                "wedge_edge_ids",
                "geometry_valid",
            }
            if not required.issubset(set(graph.keys())):
                raise RuntimeError(f"Geometry fields missing at row {graph.row_id}")
        roles[item["role"]].extend(graphs)
    counts = {key: len(value) for key, value in roles.items()}
    if counts != {"train": TRAIN_GRAPHS, "validation": VALIDATION_GRAPHS}:
        raise RuntimeError(f"Geometry role counts changed: {counts}")
    return roles, manifest


def _target_stats(graphs) -> tuple[float, float]:
    import torch

    target = torch.tensor([float(graph.y.view(-1)[0]) for graph in graphs])
    return float(target.mean()), float(target.std(unbiased=False).clamp_min(1e-6))


def _evaluate(model, loader, mean, std, device) -> dict:
    import torch

    model.eval()
    predictions = []
    targets = []
    row_ids = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            predictions.append((_forward(model, batch) * std + mean).cpu())
            targets.append(batch.y.view(-1).cpu())
            row_ids.append(batch.row_id.view(-1).cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return {
        "mae_eV": float((prediction - target).abs().mean()),
        "prediction": prediction,
        "target": target,
        "row_id": torch.cat(row_ids),
    }


def _train_gap(model, roles, run_dir: Path, *, epochs: int, stage: str, initial_hash: str):
    import torch
    import torch.nn.functional as functional

    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    model = model.to(device)
    mean, std = _target_stats(roles["train"])
    mean_tensor = torch.tensor(mean, device=device)
    std_tensor = torch.tensor(std, device=device)
    shuffle_generator = torch.Generator().manual_seed(MODEL_SEED)
    train_loader = _loader(roles["train"], shuffle=True, generator=shuffle_generator)
    validation_loader = _loader(roles["validation"], shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    trace = []
    best = float("inf")
    best_epoch = -1
    start_epoch = 0
    checkpoint_path = run_dir / f"{stage}_checkpoint.pt"
    best_model_path = run_dir / f"{stage}_best_model.pt"
    if checkpoint_path.is_file():
        checkpoint = _torch_load(checkpoint_path, map_location=device, weights_only=False)
        checks = {
            "stage": checkpoint.get("stage") == stage,
            "seed": checkpoint.get("seed") == MODEL_SEED,
            "epochs": checkpoint.get("max_epochs") == epochs,
            "initial": checkpoint.get("initial_encoder_sha256") == initial_hash,
            "cache": checkpoint.get("geometry_cache_aggregate_sha256")
            == PARENT_GEOMETRY_CACHE_SHA256,
            "batch": checkpoint.get("batch_size") == BATCH_SIZE,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Gap resume contract changed: {checks}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trace = list(checkpoint["trace"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if len(trace) != start_epoch or start_epoch > epochs:
            raise RuntimeError("Gap resume trace does not match epoch")
        if trace:
            best_row = min(trace, key=lambda row: row["validation_gap_mae_eV"])
            best = float(best_row["validation_gap_mae_eV"])
            best_epoch = int(best_row["epoch"])
            if not best_model_path.is_file():
                raise RuntimeError("Gap checkpoint has no best model")
        _restore_rng_state(checkpoint.get("rng_state", {}), shuffle_generator)
        print(f"resuming {stage} at epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        started = time.perf_counter()
        absolute = 0.0
        rows = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, batch)
            target = (batch.y.view(-1) - mean_tensor) / std_tensor
            loss = functional.l1_loss(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite Gap loss at {stage} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            absolute += float(
                (prediction.detach() * std_tensor + mean_tensor - batch.y.view(-1))
                .abs()
                .sum()
            )
            rows += int(batch.y.numel())
        scheduler.step()
        validation = _evaluate(
            model, validation_loader, mean_tensor, std_tensor, device
        )
        elapsed = time.perf_counter() - started
        improved = validation["mae_eV"] < best
        if improved:
            best = validation["mae_eV"]
            best_epoch = epoch
            _atomic_torch_save(best_model_path, model.state_dict())
        row = {
            "epoch": epoch,
            "train_gap_mae_eV": absolute / rows,
            "validation_gap_mae_eV": validation["mae_eV"],
            "elapsed_s": elapsed,
            "graphs_per_s": TRAIN_GRAPHS / elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "improved": improved,
        }
        trace.append(row)
        atomic_json(run_dir / f"{stage}_trace.json", {"epochs": trace})
        _atomic_torch_save(
            checkpoint_path,
            {
                "stage": stage,
                "epoch": epoch,
                "max_epochs": epochs,
                "seed": MODEL_SEED,
                "batch_size": BATCH_SIZE,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "initial_encoder_sha256": initial_hash,
                "geometry_cache_aggregate_sha256": PARENT_GEOMETRY_CACHE_SHA256,
                "trace": trace,
                "rng_state": _capture_rng_state(shuffle_generator),
            },
        )
        print(
            f"{stage} ep{epoch:02d} train={row['train_gap_mae_eV']:.6f} "
            f"val={validation['mae_eV']:.6f}eV {elapsed:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
    model.load_state_dict(
        _torch_load(best_model_path, map_location=device, weights_only=True)
    )
    validation = _evaluate(model, validation_loader, mean_tensor, std_tensor, device)
    payload_path = run_dir / f"{stage}_validation_payload.pt"
    _atomic_torch_save(
        payload_path,
        {
            "prediction_eV": validation["prediction"],
            "target_eV": validation["target"],
            "row_id": validation["row_id"],
            "stage": stage,
            "seed": MODEL_SEED,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        },
    )
    return model, {
        "best_epoch": best_epoch,
        "best_validation_gap_mae_eV": validation["mae_eV"],
        "epochs_completed": len(trace),
        "mean_epoch_seconds": float(np.mean([row["elapsed_s"] for row in trace])),
        "mean_graphs_per_s": float(np.mean([row["graphs_per_s"] for row in trace])),
        "best_model_sha256": sha256_file(best_model_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_payload_sha256": sha256_file(payload_path),
    }


def _geometry_loss(heads: LocalGeometryHeads, batch):
    import torch
    import torch.nn.functional as functional

    distance_prediction, angle_prediction, centers = heads.predictions(batch)
    geometry_valid = batch.geometry_valid.view(-1).bool()
    edge_valid = geometry_valid[batch.batch[batch.edge_index[0]]]
    wedge_valid = geometry_valid[batch.batch[centers]]
    if not bool(edge_valid.any()) or not bool(wedge_valid.any()):
        raise RuntimeError("A pretraining batch has no valid local geometry")
    distance_target = (
        batch.edge_distance.view(-1) - DISTANCE_CENTER_ANGSTROM
    ) / DISTANCE_SCALE_ANGSTROM
    angle_target = batch.wedge_angle_cos.view(-1)
    distance_loss = functional.smooth_l1_loss(
        distance_prediction[edge_valid], distance_target[edge_valid]
    )
    angle_loss = functional.smooth_l1_loss(
        angle_prediction[wedge_valid], angle_target[wedge_valid]
    )
    loss = distance_loss + angle_loss
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Local geometry objective became non-finite")
    return loss, distance_loss, angle_loss


def _pretrain(model, roles, run_dir: Path, *, initial_hash: str):
    import torch

    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    model = model.to(device)
    heads = LocalGeometryHeads(model)
    heads.module = heads.module.to(device)
    parameters = list(model.parameters()) + list(heads.module.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PRETRAIN_EPOCHS, eta_min=1e-6
    )
    shuffle_generator = torch.Generator().manual_seed(MODEL_SEED)
    loader = _loader(roles["train"], shuffle=True, generator=shuffle_generator)
    trace = []
    start_epoch = 0
    checkpoint_path = run_dir / "pretrain_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = _torch_load(checkpoint_path, map_location=device, weights_only=False)
        checks = {
            "stage": checkpoint.get("stage") == "local_geometry_pretrain",
            "seed": checkpoint.get("seed") == MODEL_SEED,
            "epochs": checkpoint.get("max_epochs") == PRETRAIN_EPOCHS,
            "initial": checkpoint.get("initial_encoder_sha256") == initial_hash,
            "cache": checkpoint.get("geometry_cache_aggregate_sha256")
            == PARENT_GEOMETRY_CACHE_SHA256,
            "batch": checkpoint.get("batch_size") == BATCH_SIZE,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Pretraining resume contract changed: {checks}")
        model.load_state_dict(checkpoint["model"])
        heads.module.load_state_dict(checkpoint["heads"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trace = list(checkpoint["trace"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if len(trace) != start_epoch or start_epoch > PRETRAIN_EPOCHS:
            raise RuntimeError("Pretraining resume trace does not match epoch")
        _restore_rng_state(checkpoint.get("rng_state", {}), shuffle_generator)
        print(f"resuming local geometry at epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, PRETRAIN_EPOCHS):
        model.train()
        heads.module.train()
        started = time.perf_counter()
        totals = {"loss": 0.0, "distance": 0.0, "angle": 0.0}
        batches = 0
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _forward(model, batch)
            loss, distance_loss, angle_loss = _geometry_loss(heads, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            for key, value in (
                ("loss", loss),
                ("distance", distance_loss),
                ("angle", angle_loss),
            ):
                totals[key] += float(value.detach())
            batches += 1
        scheduler.step()
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            **{key: value / batches for key, value in totals.items()},
            "elapsed_s": elapsed,
            "graphs_per_s": TRAIN_GRAPHS / elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        trace.append(row)
        atomic_json(run_dir / "pretrain_trace.json", {"epochs": trace})
        _atomic_torch_save(
            checkpoint_path,
            {
                "stage": "local_geometry_pretrain",
                "epoch": epoch,
                "max_epochs": PRETRAIN_EPOCHS,
                "seed": MODEL_SEED,
                "batch_size": BATCH_SIZE,
                "model": model.state_dict(),
                "heads": heads.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "initial_encoder_sha256": initial_hash,
                "geometry_cache_aggregate_sha256": PARENT_GEOMETRY_CACHE_SHA256,
                "gap_labels_read": False,
                "trace": trace,
                "rng_state": _capture_rng_state(shuffle_generator),
            },
        )
        print(
            f"local_geometry ep{epoch:02d} loss={row['loss']:.6f} "
            f"distance={row['distance']:.6f} angle={row['angle']:.6f} "
            f"{elapsed:.1f}s",
            flush=True,
        )
    result = {
        "epochs_completed": len(trace),
        "mean_epoch_seconds": float(np.mean([row["elapsed_s"] for row in trace])),
        "mean_graphs_per_s": float(np.mean([row["graphs_per_s"] for row in trace])),
        "training_head_parameters": sum(p.numel() for p in heads.module.parameters()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "final_encoder_sha256": _state_sha256(model),
        "gap_labels_read": False,
    }
    heads.close()
    del heads
    return model, result


def run_preflight(graph_root: Path, output_root: Path, *, source_commit: str) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SCNet did not expose a DCU")
    roles, manifest = load_roles(graph_root)
    batch = next(iter(_loader(roles["train"][:BATCH_SIZE], shuffle=False)))
    device = torch.device("cuda:0")
    _set_seed(MODEL_SEED)
    model = _make_encoder().to(device)
    if sum(p.numel() for p in model.parameters()) != EXPECTED_MODEL_PARAMETERS:
        raise RuntimeError("EdgeState parameter contract changed")
    heads = LocalGeometryHeads(model)
    heads.module = heads.module.to(device)
    batch = batch.to(device)
    _forward(model, batch)
    loss, distance_loss, angle_loss = _geometry_loss(heads, batch)
    loss.backward()
    finite = bool(torch.isfinite(loss)) and all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in list(model.parameters()) + list(heads.module.parameters())
    )
    result = {
        "format": "molgap-pcqm-local-geometry-preflight-v1",
        "accepted": finite,
        "source_commit": source_commit,
        "architecture": "ogb_edge_state_structural_gps9",
        "geometry_cache_aggregate_sha256": manifest["aggregate_sha256"],
        "batch_size": BATCH_SIZE,
        "gpu": torch.cuda.get_device_name(0),
        "finite_forward_backward": finite,
        "distance_loss": float(distance_loss.detach()),
        "angle_loss": float(angle_loss.detach()),
        "inference_parameter_count": sum(p.numel() for p in model.parameters()),
        "training_head_parameter_count": sum(p.numel() for p in heads.module.parameters()),
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "preflight.json", result)
    heads.close()
    if not finite:
        raise RuntimeError("Local geometry preflight failed")
    return result


def run_pair(
    graph_root: Path,
    output_root: Path,
    *,
    source_commit: str,
    replicate: int,
    preflight_path: Path,
) -> dict:
    import torch

    if replicate not in (1, 2):
        raise ValueError("replicate must be 1 or 2")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each paired SCNet run must see exactly one DCU")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    checks = {
        "format": preflight.get("format") == "molgap-pcqm-local-geometry-preflight-v1",
        "accepted": preflight.get("accepted") is True,
        "source": preflight.get("source_commit") == source_commit,
        "cache": preflight.get("geometry_cache_aggregate_sha256")
        == PARENT_GEOMETRY_CACHE_SHA256,
        "batch": preflight.get("batch_size") == BATCH_SIZE,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Preflight contract changed: {checks}")
    roles, manifest = load_roles(graph_root)
    output_root.mkdir(parents=True, exist_ok=True)

    _set_seed(MODEL_SEED)
    initial = _make_encoder()
    initial_hash = _state_sha256(initial)
    if sum(p.numel() for p in initial.parameters()) != EXPECTED_MODEL_PARAMETERS:
        raise RuntimeError("EdgeState parameter contract changed")
    del initial

    _set_seed(MODEL_SEED)
    scratch = _make_encoder()
    if _state_sha256(scratch) != initial_hash:
        raise RuntimeError("Scratch initialization changed")
    scratch, scratch_result = _train_gap(
        scratch,
        roles,
        output_root / "scratch",
        epochs=SCRATCH_EPOCHS,
        stage="scratch_gap",
        initial_hash=initial_hash,
    )
    del scratch
    torch.cuda.empty_cache()

    _set_seed(MODEL_SEED)
    candidate = _make_encoder()
    if _state_sha256(candidate) != initial_hash:
        raise RuntimeError("Pretrained initialization changed")
    candidate, pretraining = _pretrain(
        candidate, roles, output_root / "pretrained", initial_hash=initial_hash
    )
    candidate, finetune = _train_gap(
        candidate,
        roles,
        output_root / "pretrained",
        epochs=FINETUNE_EPOCHS,
        stage="finetune_gap",
        initial_hash=initial_hash,
    )
    del candidate
    torch.cuda.empty_cache()

    delta = (
        finetune["best_validation_gap_mae_eV"]
        - scratch_result["best_validation_gap_mae_eV"]
    )
    summary = {
        "format": "molgap-pcqm-local-geometry-paired-screen-v1",
        "complete": True,
        "source_commit": source_commit,
        "replicate": replicate,
        "execution_platform": os.environ.get("MOLGAP_EXECUTION_PLATFORM", "unknown"),
        "gpu": torch.cuda.get_device_name(0),
        "architecture": "ogb_edge_state_structural_gps9",
        "inference_parameter_count": EXPECTED_MODEL_PARAMETERS,
        "initial_encoder_sha256": initial_hash,
        "geometry_cache_aggregate_sha256": manifest["aggregate_sha256"],
        "training_contract": {
            "batch_size": BATCH_SIZE,
            "scratch_epochs": SCRATCH_EPOCHS,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "distance_center_angstrom": DISTANCE_CENTER_ANGSTROM,
            "distance_scale_angstrom": DISTANCE_SCALE_ANGSTROM,
        },
        "scratch": scratch_result,
        "pretraining": pretraining,
        "finetune": finetune,
        "pretrained_minus_scratch_eV": delta,
        "replicate_gate_passed": delta <= -MIN_PAIRED_GAIN_EV,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "molecular_research_server_accessed": False,
    }
    atomic_json(output_root / "summary.json", summary)
    summary["artifact_sha256"] = {
        str(path.relative_to(output_root)): sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "completion_manifest.json"
    }
    atomic_json(output_root / "completion_manifest.json", summary)
    return summary


def accept_pair(
    output_root: Path, *, expected_source_commit: str, expected_replicate: int
) -> dict:
    import torch

    summary = json.loads((output_root / "completion_manifest.json").read_text())
    checks = {
        "format": summary.get("format") == "molgap-pcqm-local-geometry-paired-screen-v1",
        "complete": summary.get("complete") is True,
        "source": summary.get("source_commit") == expected_source_commit,
        "replicate": summary.get("replicate") == expected_replicate,
        "architecture": summary.get("architecture") == "ogb_edge_state_structural_gps9",
        "parameters": summary.get("inference_parameter_count") == EXPECTED_MODEL_PARAMETERS,
        "cache": summary.get("geometry_cache_aggregate_sha256")
        == PARENT_GEOMETRY_CACHE_SHA256,
        "batch": summary.get("training_contract", {}).get("batch_size") == BATCH_SIZE,
        "scratch_epochs": summary.get("scratch", {}).get("epochs_completed")
        == SCRATCH_EPOCHS,
        "pretrain_epochs": summary.get("pretraining", {}).get("epochs_completed")
        == PRETRAIN_EPOCHS,
        "finetune_epochs": summary.get("finetune", {}).get("epochs_completed")
        == FINETUNE_EPOCHS,
        "gap_not_read_pretrain": summary.get("pretraining", {}).get("gap_labels_read")
        is False,
        "official_validation_sealed": summary.get("official_validation_role_read")
        is False,
        "test_dev_sealed": summary.get("test_dev_role_read") is False,
        "ims_unaccessed": summary.get("molecular_research_server_accessed") is False,
    }
    artifact_checks = {}
    for relative, expected in summary.get("artifact_sha256", {}).items():
        path = output_root / relative
        artifact_checks[relative] = path.is_file() and sha256_file(path) == expected
    payloads = {}
    for role, stage in (("scratch", "scratch_gap"), ("pretrained", "finetune_gap")):
        path = output_root / role / f"{stage}_validation_payload.pt"
        payload = _torch_load(path, map_location="cpu", weights_only=False)
        mae = float((payload["prediction_eV"] - payload["target_eV"]).abs().mean())
        payloads[role] = {"payload": payload, "mae_eV": mae}
        checks[f"{role}_rows"] = int(payload["row_id"].numel()) == VALIDATION_GRAPHS
        checks[f"{role}_metric"] = abs(
            mae
            - summary["scratch" if role == "scratch" else "finetune"][
                "best_validation_gap_mae_eV"
            ]
        ) < 1e-7
    checks["aligned_rows"] = bool(
        torch.equal(payloads["scratch"]["payload"]["row_id"], payloads["pretrained"]["payload"]["row_id"])
    )
    checks["aligned_targets"] = bool(
        torch.equal(payloads["scratch"]["payload"]["target_eV"], payloads["pretrained"]["payload"]["target_eV"])
    )
    delta = payloads["pretrained"]["mae_eV"] - payloads["scratch"]["mae_eV"]
    checks["delta"] = abs(delta - summary["pretrained_minus_scratch_eV"]) < 1e-7
    accepted = all(checks.values()) and all(artifact_checks.values())
    result = {
        "format": "molgap-pcqm-local-geometry-paired-acceptance-v1",
        "accepted": accepted,
        "checks": checks,
        "artifact_checks": artifact_checks,
        "replicate": expected_replicate,
        "scratch_validation_gap_mae_eV": payloads["scratch"]["mae_eV"],
        "pretrained_validation_gap_mae_eV": payloads["pretrained"]["mae_eV"],
        "pretrained_minus_scratch_eV": delta,
        "replicate_gate_passed": delta <= -MIN_PAIRED_GAIN_EV,
    }
    atomic_json(output_root / "acceptance.json", result)
    if not accepted:
        raise RuntimeError(f"Local geometry acceptance failed: {result}")
    return result
