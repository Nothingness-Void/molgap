"""K1 V4 100K owning trainer extraction and family output event hooks.

Loader/order/stats/forward helpers retained from pcqm_k1_variants_runner.py
at archive commit 8821b5ce893680121260627436dc11a8f7fd8403.
Pure 2D consumes original OGB/RWSE16 shards, stripping retained geometry.
"""
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

from .qm9_neural_atom import make_encoder
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


def find_fixed_cache(input_root: Path) -> tuple[Path, dict]:
    candidates = []
    for path in Path(input_root).rglob("manifest.json"):
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
    if hashlib.sha256(values.contiguous().numpy().tobytes()).hexdigest() != "df5da6a53a51d25edaacfbd72462b53a554c1ac2a667f00718a4df6308d786ce":
        raise RuntimeError("Frozen K1 training target bytes changed")
    # Use the accepted reference asset; CPU reduction partition must not drift it.
    return 5.3383002281188965, 1.275090217590332


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


INITIAL_STATE_SHA256 = "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd"
ARM_MODES = ("ssma", "clean_fingerprint")


def _state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_recipe(recipe: dict, *, mode: str) -> None:
    """Enforce the historical exposure before loading a molecular role."""
    if mode not in ARM_MODES:
        raise ValueError("Unsupported K1 screen arm")
    expected = {"epochs": EPOCHS, "optimizer_steps": EPOCHS * STEPS_PER_EPOCH,
                "sample_presentations": SAMPLE_EXPOSURE,
                "development_rows": DEVELOPMENT_ROWS, "precision": "fp32"}
    requirements = recipe.get("acceptance_requirements", {})
    if any(requirements.get(key) != value for key, value in expected.items()):
        raise ValueError("K1 frozen acceptance exposure changed")
    if recipe.get("mode") != mode:
        raise ValueError("Recipe mode mismatch")
    if recipe.get("row_order_fingerprint") != ROW_ORDER_FINGERPRINT:
        raise ValueError("K1 historical Python sampler identity changed")
    if recipe.get("initialization_sha256") != INITIAL_STATE_SHA256:
        raise ValueError("K1 initialization identity changed")
    expected_training = {"seed": 42, "batch_size": 128, "drop_last": True,
        "optimizer": "AdamW", "learning_rate": 4e-4, "weight_decay": 1e-5,
        "clip_grad_norm": 1.0, "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": 40, "scheduler_eta_min": 1e-6,
        "ema": False, "selection": "best-development-live",
        "auxiliary_weight": 0.1 if mode == "clean_fingerprint" else 0.0}
    if recipe.get("training_recipe") != expected_training:
        raise ValueError("K1 executable training recipe changed")


def _attach_fingerprint_head(model):
    """Separate training head preserves the clean public Gap forward."""
    import torch
    import torch.nn as nn
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        model.clean_fingerprint_head = nn.Sequential(
            nn.Linear(192, 32), nn.SiLU(), nn.Linear(32, 512))
    return model


def _clean_gap_state(model) -> dict:
    return {key: value for key, value in model.state_dict().items()
            if not key.startswith("clean_fingerprint_head.")}


def _evaluate(model, loader, mean, std):
    import torch
    model.eval()
    targets, predictions, source_indices = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to("cuda", non_blocking=True)
            predictions.append((_forward(model, batch) * std + mean).float().cpu())
            targets.append(batch.y.view(-1).float().cpu())
            source_indices.append(batch.source_idx.view(-1).long().cpu())
    target, prediction, source_idx = map(torch.cat, (targets, predictions, source_indices))
    return float((prediction - target).abs().mean()), target, prediction, source_idx


def run_screen_arm(*, spec, package_dir: Path, expected_package_identity: str,
                   arm_id: str, mode: str, recipe_path: Path, initial_state_path: Path,
                   input_root: Path, output: Path, account: str, run_reference: str,
                   trajectory_id: str, label_cache=None) -> dict:
    """Execute the owning V4 epoch loop with shared durable-output hooks.

    Resume is restricted to acknowledged complete epochs. Re-created epoch
    loaders retain the original Python row order and worker setup. No EMA,
    corruption, pretraining, alternate geometry cache or protected role is used.
    """
    import torch
    import torch.nn.functional as functional
    from .experiment_family_workflow import FamilyOutputSession, RunContext
    from .training_reproducibility import capture_rng_state, restore_rng_state

    recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    validate_recipe(recipe, mode=mode)
    context = RunContext.for_training(spec, package_dir,
        expected_package_identity=expected_package_identity, arm_id=arm_id,
        account=account, run_reference=run_reference)
    if (context.family_name, context.family_version) != ("neural_atom_k1", "2"):
        raise ValueError("Expected registered K1 family output adapter")
    declaration = next(arm for arm in spec.to_dict()["arms"] if arm["arm_id"] == arm_id)
    if declaration["initialization"] != {"kind": "frozen_state", "seed": SEED,
                                        "state_sha256": INITIAL_STATE_SHA256}:
        raise ValueError("K1 Spec initialization differs from executable state")
    if mode == "ssma" and label_cache is not None:
        raise ValueError("SSMA does not consume chemical auxiliary labels")
    if mode == "clean_fingerprint":
        if label_cache is None or label_cache.components != ("fingerprints",):
            raise ValueError("Clean fingerprint requires accepted fingerprint-only cache")
        role = label_cache.manifest["role"]
        if (set(label_cache.positions) != set(range(TRAIN_ROWS)) or
            role.get("fixed_manifest_sha256") != FIXED_MANIFEST_SHA256 or
            role.get("row_identity_semantics") != "pcqm-fixed100k-v4-source_idx"):
            raise ValueError("Auxiliary cache differs from original K1 training role")
        if recipe.get("label_cache_identity") != label_cache.identity:
            raise ValueError("Auxiliary cache identity differs from frozen recipe")
    if compute_row_order_fingerprint() != ROW_ORDER_FINGERPRINT:
        raise RuntimeError("Frozen row-order implementation changed")
    determinism = configure_fp32_determinism(SEED)
    root, manifest = find_fixed_cache(input_root)
    roles = load_roles(root, manifest)
    mean_value, std_value = _target_stats(roles["train"])
    # Load the exact backbone before constructing any extra trainable mechanism.
    state = torch.load(initial_state_path, map_location="cpu", weights_only=True)
    if _state_digest(state) != INITIAL_STATE_SHA256:
        raise ValueError("Pinned K1 initial tensor identity mismatch")
    model = make_encoder("neural_atom_k1")
    model.load_state_dict(state, strict=True)
    if mode == "ssma":
        from .k1_joint_aggregation import attach_k1_joint_aggregation
        attach_k1_joint_aggregation(model, mode="ssma", layer=6, seed=SEED)
    else:
        _attach_fingerprint_head(model)
    model = model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    development_loader = _development_loader(roles["development"])
    semantics = {"live_train_metric": {"metric": "MAE", "unit": "eV", "target": "Gap",
                     "role_identity": "pcqm4mv2-ogb-fixed-100k-v1:training-0-100000",
                     "weights": "live", "direction": "minimize",
                     "timing": "online-pre-update; includes dropout; no auxiliary BCE"},
                 "ema_dev_metric": None,
                 "live_dev_metric": {"metric": "MAE", "unit": "eV", "target": "Gap",
                     "role_identity": recipe["development_role_identity"],
                     "weights": "live", "direction": "minimize"}}
    session = FamilyOutputSession(output, context, adapter="k1-screen-v1", contract=recipe_path,
                                 trajectory_id=trajectory_id, metric_semantics=semantics)
    atomic_json(session.root / "runtime_manifest.json", build_runtime_manifest(determinism))
    start_epoch, best = 0, math.inf
    checkpoint = session.root / "last_checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["context"] != context.to_dict():
            raise ValueError("K1 resume source/arm identity mismatch")
        cursor = saved["cursor"]
        start_epoch = cursor["epoch"]
        if (type(start_epoch) is not int or not 1 <= start_epoch <= EPOCHS or
            cursor["next_batch"] != 0 or cursor["sampler_order_sha256"] != ROW_ORDER_FINGERPRINT or
            saved["optimizer_step"] != start_epoch * STEPS_PER_EPOCH or
            saved["sample_presentations"] != start_epoch * ROWS_PER_EPOCH):
            raise ValueError("Only acknowledged complete-epoch K1 resume is supported")
        observations = [row for row in session.stage.recorder.record["observations"]
                        if row["event"] == "observation"]
        if len(observations) != start_epoch:
            raise ValueError("Trace and durable checkpoint disagree; reconcile before resume")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best = min(row["live_dev_metric"] for row in observations)
        restore_rng_state(saved["rng_state"])
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        absolute, rows = 0.0, 0
        started = time.perf_counter()
        for batch in _train_loader(roles["train"], epoch):
            if label_cache is not None:
                label_cache.attach(batch)
            batch = batch.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if mode == "clean_fingerprint":
                representation = model.encode(batch.x, batch.edge_index, batch.edge_attr,
                                               batch.batch, batch.random_walk_pe)
                prediction = model.head(representation).view(-1)
            else:
                prediction = _forward(model, batch)
            target = (batch.y.view(-1) - mean) / std
            loss = functional.l1_loss(prediction, target)
            if mode == "clean_fingerprint":
                loss = loss + 0.1 * functional.binary_cross_entropy_with_logits(
                    model.clean_fingerprint_head(representation), batch.chemical_fingerprint.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            absolute += float((prediction.detach() - target).abs().sum())
            rows += int(target.numel())
        if rows != ROWS_PER_EPOCH:
            raise RuntimeError("Frozen optimizer exposure changed")
        validation_mae, target_eV, prediction_eV, source_idx = _evaluate(
            model, development_loader, mean, std)
        step, presentations = (epoch + 1) * STEPS_PER_EPOCH, (epoch + 1) * ROWS_PER_EPOCH
        if validation_mae < best:
            best = validation_mae
            session.selected(model_state=_clean_gap_state(model), epoch=epoch + 1,
                optimizer_step=step, weights="live", prediction_eV=prediction_eV,
                target_eV=target_eV, source_idx=source_idx)
        session.epoch_finished(epoch=epoch + 1, optimizer_step=step,
            sample_presentations=presentations, live_dev_metric=validation_mae,
            live_train_metric=absolute / rows * std_value,
            learning_rate=optimizer.param_groups[0]["lr"],
            wall_time_seconds=time.perf_counter() - started)
        scheduler.step()
        session.checkpoint(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(), rng_state=capture_rng_state(),
            cursor={"epoch": epoch + 1, "next_batch": 0,
                    "sampler_order_sha256": ROW_ORDER_FINGERPRINT},
            optimizer_step=step, sample_presentations=presentations)
        print(f"{arm_id} epoch={epoch+1} train_normalized_mae={absolute/rows:.6f} "
              f"dev={validation_mae:.6f}eV seconds={time.perf_counter()-started:.1f}", flush=True)
    return session.complete(runtime={"platform": context.platform, "account": context.account,
        "precision": "fp32", "source_commit": context.source_commit,
        "source_archive_sha256": context.source_archive_sha256}, hardware=torch.cuda.get_device_name(0))


