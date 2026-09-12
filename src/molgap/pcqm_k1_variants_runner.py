"""V4 PCQM-100K training and acceptance records for isolated K1 variants."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import struct
import time
from pathlib import Path

import numpy as np

from .pcqm_k1_variants import (
    ARCHITECTURE_CONFIGS,
    MIXER_LAYERS,
    _cluster_mixer_update,
    _dynamic_query_single_slot_update,
    _multihead_single_slot_update,
    make_encoder,
)
from .screen_policy import canonical_fingerprint, validate_screen_arm
from .training_reproducibility import (
    atomic_json,
    atomic_torch_save,
    build_runtime_manifest,
    configure_fp32_determinism,
    sha256_file,
)


SEED = 42
TRAIN_ROWS = 100_000
DEVELOPMENT_ROWS = 50_000
BATCH_SIZE = 128
EPOCHS = 40
STEPS_PER_EPOCH = TRAIN_ROWS // BATCH_SIZE
ROWS_PER_EPOCH = STEPS_PER_EPOCH * BATCH_SIZE
SAMPLE_EXPOSURE = ROWS_PER_EPOCH * EPOCHS
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-5
MINIMUM_GAIN_EV = 0.003
STOCHASTICITY_FLOOR_EV = 0.003
FIXED_DATASET = "kaseichou/pcqm4mv2-ogb-fixed-100k-v1"
FIXED_MANIFEST_SHA256 = (
    "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
)
FIXED_GEOMETRY_SHA256 = (
    "bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5"
)
BENCHMARK_ID = "pcqm4mv2-ogb-fixed-100k-gap-v4"
PLATFORM_ID = "kaggle2"
FORBIDDEN_MODEL_FIELDS = (
    "pos",
    "edge_distance",
    "wedge_angle_cos",
    "wedge_edge_ids",
    "geometry_valid",
)


def _hash_mapping(value: dict) -> str:
    return canonical_fingerprint(value)


FEATURE_FINGERPRINT = _hash_mapping(
    {
        "atom": "ogb-nine-categorical",
        "bond": "ogb-three-categorical-real-bonds",
        "rwse": 16,
        "geometry_model_input": False,
    }
)
TARGET_FINGERPRINT = _hash_mapping(
    {"name": "pcqm4mv2-homo-lumo-gap", "column": "gap", "unit": "eV"}
)
OPTIMIZER_FINGERPRINT = _hash_mapping(
    {"name": "AdamW", "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "clip": 1.0}
)
SCHEDULE_FINGERPRINT = _hash_mapping(
    {"name": "CosineAnnealingLR", "epochs": EPOCHS, "eta_min": 1e-6}
)
LOSS_FINGERPRINT = _hash_mapping({"name": "L1", "space": "normalized-gap"})
TARGET_TRANSFORM_FINGERPRINT = _hash_mapping(
    {"mean": "all-100k-train", "std": "all-100k-train-sample-std"}
)
SELECTION_FINGERPRINT = _hash_mapping(
    {"role": "official-train-derived-next-50k", "criterion": "minimum-gap-mae-each-epoch"}
)
ROLE_ACCESS_FINGERPRINT = _hash_mapping(
    {
        "train": [0, 100_000],
        "development": [100_000, 150_000],
        "official_validation": False,
        "test_dev": False,
        "test_challenge": False,
    }
)


def epoch_order(epoch: int) -> list[int]:
    """Return a Python-version-stable row order for one complete-batch pass."""
    if not 0 <= epoch < EPOCHS:
        raise ValueError(f"epoch outside frozen range: {epoch}")
    values = list(range(TRAIN_ROWS))
    random.Random(SEED * 1_000_003 + epoch).shuffle(values)
    return values[:ROWS_PER_EPOCH]


def compute_row_order_fingerprint() -> str:
    digest = hashlib.sha256()
    for epoch in range(EPOCHS):
        digest.update(struct.pack("<I", epoch))
        for index in epoch_order(epoch):
            digest.update(struct.pack("<I", index))
    return digest.hexdigest()


ROW_ORDER_FINGERPRINT = (
    "e85736669a04029e0fa40e993a085b2e0226b4be98a923a141f999a672ce3f34"
)


class _PackedGraphDatasetFactory:
    @staticmethod
    def load(path: Path):
        import torch
        from torch_geometric.data import InMemoryDataset

        class PackedGraphDataset(InMemoryDataset):
            def __init__(self, source: Path):
                super().__init__(root=None)
                self.data, self.slices = torch.load(
                    source, map_location="cpu", weights_only=False
                )

        payload = PackedGraphDataset(path)
        for field in FORBIDDEN_MODEL_FIELDS:
            if field in payload._data:
                del payload._data[field]
                payload.slices.pop(field, None)
        return payload


def find_fixed_cache() -> tuple[Path, dict]:
    candidates = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("format") == "molgap-pcqm4mv2-kaggle-fixed-subset-v1"
            and payload.get("identity", {}).get("name") == "ogb-train-100k"
        ):
            candidates.append((path.parent, payload))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one fixed 100K cache, found {candidates}")
    root, manifest = candidates[0]
    if sha256_file(root / "manifest.json") != FIXED_MANIFEST_SHA256:
        raise RuntimeError("Fixed 100K manifest content changed")
    expected_identity = {
        "name": "ogb-train-100k",
        "train_rows": TRAIN_ROWS,
        "development_rows": DEVELOPMENT_ROWS,
        "kaggle1": True,
        "scnet_compatible": False,
    }
    if manifest.get("status") != "complete" or manifest.get("identity") != expected_identity:
        raise RuntimeError("Fixed 100K identity changed")
    if manifest.get("geometry_aggregate_sha256") != FIXED_GEOMETRY_SHA256:
        raise RuntimeError("Fixed 100K payload aggregate changed")
    expected_graph = {
        "feature_schema": "ogb",
        "node_feature_dim": 9,
        "edge_feature_dim": 3,
        "rwse_dim": 16,
        "geometry_method": "ETKDGv3",
        "optimization_method": "MMFF94s",
    }
    if manifest.get("graph_contract") != expected_graph:
        raise RuntimeError("Fixed 100K graph contract changed")
    if manifest.get("source_indices_embedded_in_graphs") is not True:
        raise RuntimeError("Fixed cache lost source indices")
    for role in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        if manifest.get(role) is not False:
            raise RuntimeError(f"Sealed role changed: {role}")
    return root, manifest


def load_roles(root: Path, manifest: dict):
    import torch
    from torch.utils.data import ConcatDataset

    roles = {"train": [], "development": []}
    aggregate = hashlib.sha256()
    expected_start = {"train": 0, "development": TRAIN_ROWS}
    for item in manifest["geometry_shards"]:
        path = root / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Fixed shard changed: {item['file']}")
        payload = _PackedGraphDatasetFactory.load(path)
        if len(payload) != item["rows"]:
            raise RuntimeError(f"Fixed shard row count changed: {item['file']}")
        role = item["role"]
        source_idx = payload._data.source_idx.view(-1).long()
        start = expected_start[role]
        expected = torch.arange(start, start + len(payload), dtype=torch.long)
        if not torch.equal(source_idx, expected):
            raise RuntimeError(f"Source order changed: {item['file']}")
        expected_start[role] += len(payload)
        roles[role].append(payload)
        aggregate.update(
            f"{role}\tstore/geometry/{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    if aggregate.hexdigest() != FIXED_GEOMETRY_SHA256:
        raise RuntimeError("Fixed aggregate recomputation changed")
    combined = {name: ConcatDataset(parts) for name, parts in roles.items()}
    if len(combined["train"]) != TRAIN_ROWS or len(combined["development"]) != DEVELOPMENT_ROWS:
        raise RuntimeError("Fixed role counts changed")
    return combined


def _target_stats(graphs) -> tuple[float, float]:
    import torch

    values = torch.cat(
        [dataset._data.y.view(-1).float() for dataset in graphs.datasets]
    )
    return float(values.mean()), float(values.std())


def _train_loader(graphs, epoch: int):
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    selected = Subset(graphs, epoch_order(epoch))
    return DataLoader(
        selected,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )


def _development_loader(graphs):
    from torch_geometric.loader import DataLoader

    return DataLoader(
        graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )


def _forward(model, batch):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    ).view(-1)


def _state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _batch_sha256(batch) -> str:
    digest = hashlib.sha256()
    for name in ("x", "edge_index", "edge_attr", "batch", "random_walk_pe", "y", "source_idx"):
        value = getattr(batch, name).detach().cpu().contiguous()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _optimizer_step(model, optimizer, batch, mean, std):
    import torch.nn.functional as functional

    optimizer.zero_grad(set_to_none=True)
    prediction = _forward(model, batch)
    target = (batch.y.view(-1) - mean) / std
    loss = functional.l1_loss(prediction, target)
    loss.backward()
    import torch

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.detach()


def build_runtime_certificate(roles, output: Path) -> tuple[dict, dict]:
    import torch

    determinism = configure_fp32_determinism(SEED)
    runtime = build_runtime_manifest(determinism)
    mean_value, std_value = _target_stats(roles["train"])
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    batch = next(iter(_train_loader(roles["train"], 0))).to("cuda", non_blocking=True)
    if int(batch.num_graphs) != BATCH_SIZE:
        raise RuntimeError("Runtime calibration did not receive physical batch 128")
    fixture_sha = _batch_sha256(batch)
    hashes = []
    losses = []
    for _ in range(2):
        configure_fp32_determinism(SEED)
        model = make_encoder("neural_atom_k1_v4").to("cuda").train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        losses.append(float(_optimizer_step(model, optimizer, batch, mean, std).cpu()))
        torch.cuda.synchronize()
        hashes.append(_state_sha256(model))
        del model, optimizer
        torch.cuda.empty_cache()
    if hashes[0] != hashes[1] or losses[0] != losses[1]:
        raise RuntimeError("Optimizer-step calibration is not deterministic")
    certificate = {
        "format": "molgap-runtime-certificate-v1",
        "status": "accepted",
        "platform_id": PLATFORM_ID,
        "accelerator": torch.cuda.get_device_name(0),
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": BATCH_SIZE,
        "tail_batch_policy": "drop_last",
        "software_fingerprint": runtime["installed_distributions_sha256"],
        "determinism_fingerprint": canonical_fingerprint(determinism),
        "calibration_fixture_sha256": fixture_sha,
        "calibration_output_sha256": hashes[0],
        "runtime_fingerprint": runtime["runtime_fingerprint"],
        "calibration_checks_passed": True,
    }
    certificate_id = canonical_fingerprint(certificate)
    atomic_json(output / "runtime_manifest.json", runtime)
    atomic_json(output / "runtime_certificate.json", certificate)
    return certificate, {
        "runtime_certificate_id": certificate_id,
        "target_stats": {"mean_eV": mean_value, "sample_std_eV": std_value},
        "calibration_loss": losses[0],
    }


def _evaluate(model, loader, mean, std):
    import torch

    model.eval()
    targets = []
    predictions = []
    source_indices = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to("cuda", non_blocking=True)
            predictions.append((_forward(model, batch) * std + mean).float().cpu())
            targets.append(batch.y.view(-1).float().cpu())
            source_indices.append(batch.source_idx.view(-1).long().cpu())
    target = torch.cat(targets)
    prediction = torch.cat(predictions)
    source_idx = torch.cat(source_indices)
    return float((prediction - target).abs().mean()), target, prediction, source_idx


def _base_state(model):
    return model if not hasattr(model, "base") else model.base


def _architecture_preflight(mode: str, roles, target_stats: dict) -> dict:
    import torch

    configure_fp32_determinism(SEED)
    baseline = make_encoder("neural_atom_k1_v4").to("cuda").eval()
    baseline_sha = _state_sha256(baseline)
    configure_fp32_determinism(SEED)
    model = make_encoder(mode).to("cuda").eval()
    shared_sha = _state_sha256(_base_state(model))
    if shared_sha != baseline_sha:
        raise RuntimeError(f"K1 shared initialization changed for {mode}")
    batch = next(iter(_train_loader(roles["train"], 0))).to("cuda", non_blocking=True)
    with torch.no_grad():
        baseline_prediction = _forward(baseline, batch)
        candidate_prediction = _forward(model, batch)
    exact_nested_initialization = bool(torch.equal(baseline_prediction, candidate_prediction))
    if mode != "neural_atom_k1_v4" and not exact_nested_initialization:
        raise RuntimeError(f"Candidate is not functionally nested in K1: {mode}")
    mechanism_checks = {}
    if mode == "neural_atom_k4_cluster":
        mixer = model.base.neural_atom_mixers[str(MIXER_LAYERS[0])]
        probe = torch.linspace(
            -1.0,
            1.0,
            steps=int(batch.num_nodes) * 192,
            device="cuda",
        ).reshape(int(batch.num_nodes), 192)
        update, _, assignment, valid, diagnostics = _cluster_mixer_update(
            mixer, probe, batch.batch
        )
        valid_mass = assignment.sum(dim=1).masked_select(valid)
        padding_mass = assignment.masked_select(~valid.unsqueeze(1)).abs().sum()
        mechanism_checks = {
            "active_slots": diagnostics["active_slots"],
            "allocation_normalization_axis": diagnostics[
                "allocation_normalization_axis"
            ],
            "valid_atom_allocation_mass_one": bool(
                torch.allclose(
                    valid_mass,
                    torch.ones_like(valid_mass),
                    atol=1e-6,
                    rtol=0,
                )
            ),
            "padding_allocation_mass_zero": bool(padding_mass.item() == 0.0),
            "zero_return_exact": bool(torch.count_nonzero(update).item() == 0),
        }
        if mechanism_checks != {
            "active_slots": 4,
            "allocation_normalization_axis": "slots",
            "valid_atom_allocation_mass_one": True,
            "padding_allocation_mass_zero": True,
            "zero_return_exact": True,
        }:
            raise RuntimeError(
                f"Paper-allocation invariant failed: {mechanism_checks}"
            )
    elif mode == "neural_atom_k1_h4":
        mixer = model.base.neural_atom_mixers[str(MIXER_LAYERS[0])]
        probe = torch.linspace(
            -1.0,
            1.0,
            steps=int(batch.num_nodes) * 192,
            device="cuda",
        ).reshape(int(batch.num_nodes), 192)
        update, _, assignment, valid, diagnostics = _multihead_single_slot_update(
            mixer, probe, batch.batch
        )
        valid_head_mass = assignment.sum(dim=-1)
        padding_mass = assignment.masked_select(
            ~valid[:, None, None, :]
        ).abs().sum()
        mechanism_checks = {
            "active_slots": diagnostics["active_slots"],
            "allocation_heads": diagnostics["allocation_heads"],
            "head_channels": diagnostics["head_channels"],
            "allocation_normalization_axis": diagnostics[
                "allocation_normalization_axis"
            ],
            "valid_head_allocation_mass_one": bool(
                torch.allclose(
                    valid_head_mass,
                    torch.ones_like(valid_head_mass),
                    atol=1e-6,
                    rtol=0,
                )
            ),
            "padding_allocation_mass_zero": bool(padding_mass.item() == 0.0),
            "zero_return_exact": bool(torch.count_nonzero(update).item() == 0),
        }
        if mechanism_checks != {
            "active_slots": 1,
            "allocation_heads": 4,
            "head_channels": 16,
            "allocation_normalization_axis": "atoms-per-head",
            "valid_head_allocation_mass_one": True,
            "padding_allocation_mass_zero": True,
            "zero_return_exact": True,
        }:
            raise RuntimeError(
                f"Multi-view allocation invariant failed: {mechanism_checks}"
            )
    elif mode == "neural_atom_k1_dynamic_query":
        layer = str(MIXER_LAYERS[0])
        mixer = model.base.neural_atom_mixers[layer]
        conditioner = model.query_conditioners[layer]
        probe = torch.linspace(
            -1.0,
            1.0,
            steps=int(batch.num_nodes) * 192,
            device="cuda",
        ).reshape(int(batch.num_nodes), 192)
        update, _, assignment, valid, diagnostics = (
            _dynamic_query_single_slot_update(
                mixer, conditioner, probe, batch.batch
            )
        )
        valid_mass = assignment.sum(dim=-1)
        padding_mass = assignment.masked_select(
            ~valid.unsqueeze(1)
        ).abs().sum()
        mechanism_checks = {
            "active_slots": diagnostics["active_slots"],
            "allocation_heads": diagnostics["allocation_heads"],
            "query_context": diagnostics["query_context"],
            "allocation_normalization_axis": diagnostics[
                "allocation_normalization_axis"
            ],
            "conditioner_zero_at_initialization": bool(
                torch.count_nonzero(conditioner.weight).item() == 0
            ),
            "valid_allocation_mass_one": bool(
                torch.allclose(
                    valid_mass,
                    torch.ones_like(valid_mass),
                    atol=1e-6,
                    rtol=0,
                )
            ),
            "padding_allocation_mass_zero": bool(padding_mass.item() == 0.0),
            "zero_return_exact": bool(torch.count_nonzero(update).item() == 0),
        }
        if mechanism_checks != {
            "active_slots": 1,
            "allocation_heads": 1,
            "query_context": "mean-current-node-state",
            "allocation_normalization_axis": "atoms",
            "conditioner_zero_at_initialization": True,
            "valid_allocation_mass_one": True,
            "padding_allocation_mass_zero": True,
            "zero_return_exact": True,
        }:
            raise RuntimeError(
                f"Dynamic-query invariant failed: {mechanism_checks}"
            )
    mean = torch.tensor(target_stats["mean_eV"], device="cuda")
    std = torch.tensor(target_stats["sample_std_eV"], device="cuda")
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    _optimizer_step(model, optimizer, batch, mean, std)
    _optimizer_step(model, optimizer, batch, mean, std)
    candidate_parameters = []
    if mode == "neural_atom_k1_g":
        candidate_parameters = list(model.molecule_gates.parameters())
    elif mode == "neural_atom_k1_r":
        candidate_parameters = list(model.relation_slots.parameters())
    elif mode == "neural_atom_k4_cluster":
        candidate_parameters = list(model.base.neural_atom_mixers.parameters())
    elif mode == "neural_atom_k1_h4":
        candidate_parameters = list(model.base.neural_atom_mixers.parameters())
    elif mode == "neural_atom_k1_dynamic_query":
        candidate_parameters = list(model.query_conditioners.parameters())
    candidate_trainable = mode == "neural_atom_k1_v4" or any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0
        for parameter in candidate_parameters
    )
    if not candidate_trainable:
        raise RuntimeError(f"Candidate-only mechanism has no finite gradient: {mode}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del baseline, model, optimizer
    torch.cuda.empty_cache()
    return {
        "mode": mode,
        "parameter_count": parameter_count,
        "shared_k1_initial_state_sha256": shared_sha,
        "exact_k1_function_at_initialization": exact_nested_initialization,
        "candidate_mechanism_trainable_after_two_steps": candidate_trainable,
        "exchange_layers": list(MIXER_LAYERS),
        "mechanism_checks": mechanism_checks,
    }


def _contract(
    *,
    mode: str,
    accelerator: str,
    runtime_certificate_id: str,
    source_archive_sha256: str,
    result_artifact_sha256: str,
) -> dict:
    architecture_fingerprint = canonical_fingerprint(ARCHITECTURE_CONFIGS[mode])
    return {
        "run_id": f"pcqm-k1-variants-100k-s42-v1-{mode}",
        "model_id": mode,
        "architecture_fingerprint": architecture_fingerprint,
        "source_archive_sha256": source_archive_sha256,
        "result_artifact_sha256": result_artifact_sha256,
        "platform_id": PLATFORM_ID,
        "accelerator": accelerator,
        "runtime_certificate_id": runtime_certificate_id,
        "benchmark_id": BENCHMARK_ID,
        "data_role_fingerprint": FIXED_MANIFEST_SHA256,
        "row_order_fingerprint": ROW_ORDER_FINGERPRINT,
        "feature_fingerprint": FEATURE_FINGERPRINT,
        "target_fingerprint": TARGET_FINGERPRINT,
        "seed": SEED,
        "precision": "fp32",
        "optimizer_fingerprint": OPTIMIZER_FINGERPRINT,
        "schedule_fingerprint": SCHEDULE_FINGERPRINT,
        "loss_fingerprint": LOSS_FINGERPRINT,
        "target_transform_fingerprint": TARGET_TRANSFORM_FINGERPRINT,
        "selection_fingerprint": SELECTION_FINGERPRINT,
        "role_access_fingerprint": ROLE_ACCESS_FINGERPRINT,
        "sample_exposure": SAMPLE_EXPOSURE,
        "tail_batch_policy": "drop_last",
        "physical_batch_per_device": BATCH_SIZE,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
    }


def train_arm(
    mode: str,
    output: Path,
    *,
    source_commit: str,
    source_archive_sha256: str,
) -> dict:
    import torch
    import torch.nn.functional as functional

    if mode not in ARCHITECTURE_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}")
    if len(source_commit) != 40 or len(source_archive_sha256) != 64:
        raise ValueError("Committed source and archive identities are required")
    validate_screen_arm(physical_batch_per_device=BATCH_SIZE)
    if compute_row_order_fingerprint() != ROW_ORDER_FINGERPRINT:
        raise RuntimeError("Frozen row order implementation changed")
    configure_fp32_determinism(SEED)
    root, manifest = find_fixed_cache()
    roles = load_roles(root, manifest)
    output.mkdir(parents=True, exist_ok=True)
    certificate, runtime = build_runtime_certificate(roles, output)
    target_stats = runtime["target_stats"]
    preflight = _architecture_preflight(mode, roles, target_stats)
    atomic_json(output / "preflight.json", preflight)
    configure_fp32_determinism(SEED)
    model = make_encoder(mode).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    mean = torch.tensor(target_stats["mean_eV"], device="cuda")
    std = torch.tensor(target_stats["sample_std_eV"], device="cuda")
    development_loader = _development_loader(roles["development"])
    best = math.inf
    best_epoch = -1
    trace = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        model.train()
        absolute = 0.0
        rows = 0
        started = time.perf_counter()
        for batch in _train_loader(roles["train"], epoch):
            batch = batch.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, batch)
            target = (batch.y.view(-1) - mean) / std
            loss = functional.l1_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            absolute += float((prediction.detach() - target).abs().sum())
            rows += int(target.numel())
        if rows != ROWS_PER_EPOCH:
            raise RuntimeError(f"Optimizer exposure changed: {rows}")
        validation_mae, target_eV, prediction_eV, source_idx = _evaluate(
            model, development_loader, mean, std
        )
        improved = validation_mae < best
        if improved:
            best = validation_mae
            best_epoch = epoch
            atomic_torch_save(output / "best_model.pt", model.state_dict())
            atomic_torch_save(
                output / "best_development_payload.pt",
                {
                    "target_eV": target_eV,
                    "prediction_eV": prediction_eV,
                    "source_idx": source_idx,
                },
            )
        row = {
            "epoch": epoch,
            "optimizer_steps": (epoch + 1) * STEPS_PER_EPOCH,
            "sample_presentations": (epoch + 1) * ROWS_PER_EPOCH,
            "train_normalized_mae": absolute / rows,
            "development_gap_mae_eV": validation_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
            "improved": improved,
        }
        trace.append(row)
        scheduler.step()
        atomic_torch_save(
            output / "last_checkpoint.pt",
            {
                "format": "molgap-k1-variant-checkpoint-v1",
                "mode": mode,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "trace": trace,
                "source_commit": source_commit,
                "source_archive_sha256": source_archive_sha256,
                "runtime_certificate_id": runtime["runtime_certificate_id"],
                "official_validation_role_read": False,
                "test_dev_role_read": False,
                "test_challenge_role_read": False,
            },
        )
        atomic_json(output / "trace.json", {"epochs": trace})
        print(
            f"{mode} ep{epoch:02d} train={row['train_normalized_mae']:.6f} "
            f"dev={validation_mae:.6f}eV {row['seconds']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
    payload_sha = sha256_file(output / "best_development_payload.pt")
    contract = _contract(
        mode=mode,
        accelerator=torch.cuda.get_device_name(0),
        runtime_certificate_id=runtime["runtime_certificate_id"],
        source_archive_sha256=source_archive_sha256,
        result_artifact_sha256=payload_sha,
    )
    training = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "development_gap_mae_eV": best,
        "epochs_completed": len(trace),
        "optimizer_steps": EPOCHS * STEPS_PER_EPOCH,
        "sample_presentations": SAMPLE_EXPOSURE,
        "mean_epoch_seconds": float(np.mean([row["seconds"] for row in trace])),
        "mean_graphs_per_second": SAMPLE_EXPOSURE / sum(row["seconds"] for row in trace),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "total_memory_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2,
        "best_model_sha256": sha256_file(output / "best_model.pt"),
        "payload_sha256": payload_sha,
        "checkpoint_sha256": sha256_file(output / "last_checkpoint.pt"),
    }
    record = {
        "format": "molgap-pcqm-k1-variant-arm-v1",
        "complete": True,
        "source_commit": source_commit,
        "fixed_dataset": FIXED_DATASET,
        "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
        "fixed_geometry_sha256": FIXED_GEOMETRY_SHA256,
        "architecture": ARCHITECTURE_CONFIGS[mode],
        "preflight": preflight,
        "runtime_certificate": certificate,
        "contract": contract,
        "training": training,
        "pure_2d": True,
        "geometry_attributes_removed_before_batching": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    if mode == "neural_atom_k1_v4":
        record["contract"].update(
            {
                "frozen_reference": True,
                "stochasticity_floor_eV": STOCHASTICITY_FLOOR_EV,
                "minimum_material_gain_eV": MINIMUM_GAIN_EV,
            }
        )
    atomic_json(output / "arm_record.json", record)
    atomic_json(
        output / "completion_manifest.json",
        {
            "format": "molgap-pcqm-k1-variant-completion-v1",
            "complete": True,
            "mode": mode,
            "artifact_sha256": {
                path.name: sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "completion_manifest.json"
            },
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
        },
    )
    return record
