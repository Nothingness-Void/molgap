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
from functools import lru_cache
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
from .v4_runtime import state_dict_sha256, certify_numerical_repeatability, normalized_source_sha256


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


def _validate_seed(seed: int) -> int:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("K1 seed must be an integer in [0, 2**32 - 1]")
    return seed


def epoch_order(epoch: int, seed: int = SEED) -> list[int]:
    """Return a Python-version-stable row order for one complete-batch pass."""
    if not 0 <= epoch < EPOCHS:
        raise ValueError(f"epoch outside frozen range: {epoch}")
    _validate_seed(seed)
    values = list(range(TRAIN_ROWS))
    random.Random(seed * 1_000_003 + epoch).shuffle(values)
    return values[:ROWS_PER_EPOCH]


@lru_cache(maxsize=16, typed=True)
def compute_row_order_fingerprint(seed: int = SEED) -> str:
    _validate_seed(seed)
    digest = hashlib.sha256()
    for epoch in range(EPOCHS):
        digest.update(struct.pack("<I", epoch))
        for index in epoch_order(epoch, seed):
            digest.update(struct.pack("<I", index))
    return digest.hexdigest()


ROW_ORDER_FINGERPRINT = (
    "e85736669a04029e0fa40e993a085b2e0226b4be98a923a141f999a672ce3f34"
)


class _PackedGraphDatasetFactory:
    @staticmethod
    def load(path: Path):
        from .packed_graph_dataset import PackedGraphDataset

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


def load_roles(root: Path, manifest: dict, *, selected_roles=("train", "development")):
    import torch
    from torch.utils.data import ConcatDataset

    if not selected_roles or len(set(selected_roles)) != len(selected_roles) or not set(selected_roles) <= {"train", "development"}:
        raise ValueError("Select distinct fixed-cache roles")
    roles = {name: [] for name in selected_roles}
    aggregate = hashlib.sha256()
    expected_start = {"train": 0, "development": TRAIN_ROWS}
    for item in manifest["geometry_shards"]:
        role = item["role"]
        aggregate.update(
            f"{role}\tstore/geometry/{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
        if role not in roles:
            continue
        path = root / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Fixed shard changed: {item['file']}")
        payload = _PackedGraphDatasetFactory.load(path)
        if len(payload) != item["rows"]:
            raise RuntimeError(f"Fixed shard row count changed: {item['file']}")
        source_idx = payload._data.source_idx.view(-1).long()
        start = expected_start[role]
        expected = torch.arange(start, start + len(payload), dtype=torch.long)
        if not torch.equal(source_idx, expected):
            raise RuntimeError(f"Source order changed: {item['file']}")
        expected_start[role] += len(payload)
        roles[role].append(payload)
    if aggregate.hexdigest() != FIXED_GEOMETRY_SHA256:
        raise RuntimeError("Fixed aggregate recomputation changed")
    combined = {name: ConcatDataset(parts) for name, parts in roles.items()}
    counts = {"train": TRAIN_ROWS, "development": DEVELOPMENT_ROWS}
    if any(len(combined[name]) != counts[name] for name in selected_roles):
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


def _train_loader(graphs, epoch: int, seed: int = SEED):
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    selected = Subset(graphs, epoch_order(epoch, seed))
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
    return state_dict_sha256(model.state_dict())


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
ARM_MODES = ("reference", "ssma", "clean_fingerprint", "mean2", "mean2_clean_second", "fused_layout")


def build_initial_state(seed: int = SEED) -> dict:
    """Construct fresh V4 K1 CPU tensors without consuming caller CPU RNG."""
    import torch
    _validate_seed(seed)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        model = make_encoder("neural_atom_k1")
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _validate_initialization(seed: int, initialization_sha256: str) -> None:
    from .experiment_spec import _digest
    _digest(initialization_sha256, "initialization_sha256")
    if seed == SEED and initialization_sha256 != INITIAL_STATE_SHA256:
        raise ValueError("K1 historical initialization identity changed")


def _load_initial_state(path: Path, recipe: dict) -> dict:
    """Verify transported tensors, never regenerate them on another runtime."""
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state_dict_sha256(state) != recipe["initialization_sha256"]:
        raise ValueError("Pinned K1 initial tensor identity mismatch")
    return state


def _recipe_seed(recipe: dict) -> int:
    return _validate_seed(recipe.get("training_recipe", {}).get("seed"))


def _maximum_overhead(mode):
    # The two-pass control intentionally doubles forwards; it is not an SSMA extension.
    return 1.0 if mode in ("mean2", "mean2_clean_second") else 0.25


def _state_digest(state: dict) -> str:
    return state_dict_sha256(state)


def validate_recipe(recipe: dict, *, mode: str) -> None:
    """Enforce the historical exposure before loading a molecular role."""
    if mode not in ARM_MODES:
        raise ValueError("Unsupported K1 screen arm")
    limit = recipe.get("allocation_wall_limit_seconds")
    if limit is not None and (type(limit) is not int or not 120 <= limit <= 14400):
        raise ValueError("Invalid allocation ceiling")
    expected = {"epochs": EPOCHS, "optimizer_steps": EPOCHS * STEPS_PER_EPOCH,
                "sample_presentations": SAMPLE_EXPOSURE,
                "development_rows": DEVELOPMENT_ROWS, "precision": "fp32"}
    requirements = recipe.get("acceptance_requirements", {})
    if any(requirements.get(key) != value for key, value in expected.items()):
        raise ValueError("K1 frozen acceptance exposure changed")
    if recipe.get("mode") != mode:
        raise ValueError("Recipe mode mismatch")
    seed = _recipe_seed(recipe)
    if recipe.get("row_order_fingerprint") != compute_row_order_fingerprint(seed):
        raise ValueError("K1 historical Python sampler identity changed")
    _validate_initialization(seed, recipe.get("initialization_sha256"))
    expected_training = {"seed": seed, "batch_size": 128, "drop_last": True,
        "optimizer": "AdamW", "learning_rate": 4e-4, "weight_decay": 1e-5,
        "clip_grad_norm": 1.0, "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": 40, "scheduler_eta_min": 1e-6,
        "ema": False, "selection": "best-development-live",
        "auxiliary_weight": 0.1 if mode == "clean_fingerprint" else 0.0}
    if recipe.get("training_recipe") != expected_training:
        raise ValueError("K1 executable training recipe changed")
    policy = recipe.get("execution_policy")
    if mode == "fused_layout":
        if policy != {"adamw_fused": True, "adamw_foreach": False,
                      "cpu_layout": True, "loader": "original-epoch-loader"}:
            raise ValueError("Fused-layout execution policy changed")
    elif policy is not None:
        raise ValueError("Execution overrides require the fused-layout addon")


def build_screen_recipe(mode: str, *, source_idx_sha256: str, target_sha256: str,
                        seed: int = SEED, initialization_sha256: str | None = None) -> dict:
    """Build fixed family constants; callers pin real retained development rows."""
    from .experiment_spec import _digest
    for name, value in (("source_idx_sha256", source_idx_sha256), ("target_sha256", target_sha256)):
        _digest(value, name)
    _validate_seed(seed)
    if initialization_sha256 is None:
        if seed != SEED:
            raise ValueError("Nonhistorical seed requires a frozen initialization_sha256")
        initialization_sha256 = INITIAL_STATE_SHA256
    _validate_initialization(seed, initialization_sha256)
    if mode not in {"reference", "ssma", "mean2", "mean2_clean_second", "fused_layout"}:
        raise ValueError("No executable screen recipe for this K1 addon")
    recipe = {"mode": mode, "row_order_fingerprint": compute_row_order_fingerprint(seed),
        "initialization_sha256": initialization_sha256,
        "development_role_identity": "pcqm4mv2-ogb-fixed-100k-v1:development-100000-150000",
        "training_recipe": {"seed": seed, "batch_size": BATCH_SIZE, "drop_last": True,
            "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "clip_grad_norm": 1.0, "scheduler": "CosineAnnealingLR", "scheduler_t_max": EPOCHS,
            "scheduler_eta_min": 1e-6, "ema": False, "selection": "best-development-live", "auxiliary_weight": 0.0},
        "acceptance_requirements": {"epochs": EPOCHS, "optimizer_steps": EPOCHS * STEPS_PER_EPOCH,
            "sample_presentations": SAMPLE_EXPOSURE, "development_rows": DEVELOPMENT_ROWS,
            "precision": "fp32", "source_idx_sha256": source_idx_sha256, "target_sha256": target_sha256}}
    if mode == "fused_layout":
        recipe["execution_policy"] = {"adamw_fused": True, "adamw_foreach": False,
                                      "cpu_layout": True, "loader": "original-epoch-loader"}
    return recipe


def _attach_fingerprint_head(model, seed=SEED):
    """Separate training head preserves the clean public Gap forward."""
    import torch
    import torch.nn as nn
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
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


def _make_screen_model(state, mode, seed=SEED):
    model = make_encoder("neural_atom_k1")
    model.load_state_dict(state, strict=True)
    if mode == "ssma":
        from .k1_joint_aggregation import attach_k1_joint_aggregation
        attach_k1_joint_aggregation(model, mode="ssma", layer=6, seed=seed)
    elif mode == "clean_fingerprint":
        _attach_fingerprint_head(model, seed)
    return model.to("cuda")


def _optimizer_step(model, optimizer, batch, mean, std, mode="reference"):
    """Original V4 optimizer-inclusive step, with optional clean auxiliary loss."""
    import torch
    import torch.nn.functional as functional
    optimizer.zero_grad(set_to_none=True)
    if mode == "clean_fingerprint":
        representation = model.encode(batch.x, batch.edge_index, batch.edge_attr,
                                       batch.batch, batch.random_walk_pe)
        prediction = model.head(representation).view(-1)
    else:
        prediction = _forward(model, batch)
    target = (batch.y.view(-1) - mean) / std
    loss = functional.l1_loss(prediction, target)
    absolute = (prediction.detach() - target).abs().sum()
    if mode in ("mean2", "mean2_clean_second"):
        from contextlib import nullcontext
        second_view = nullcontext()
        if mode == "mean2_clean_second":
            from .k1_clean_second import clean_second_view
            second_view = clean_second_view(model)
        with second_view:
            second = _forward(model, batch)
        loss = 0.5 * (loss + functional.l1_loss(second, target))
        absolute = 0.5 * (absolute + (second.detach() - target).abs().sum())
    if mode == "clean_fingerprint":
        loss = loss + 0.1 * functional.binary_cross_entropy_with_logits(
            model.clean_fingerprint_head(representation), batch.chemical_fingerprint.float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.detach(), absolute, int(target.numel())


def _screen_optimizer(model, mode):
    import torch
    if mode == "fused_layout":
        from .v4_runtime import make_adamw_compat
        return make_adamw_compat(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY, fused=True, foreach=False)
    # Keep the historical optimizer's native defaults, including foreach=None.
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def _screen_step(model, optimizer, cpu_batch, mean, std, mode):
    from contextlib import nullcontext
    context = nullcontext()
    if mode == "fused_layout":
        from .k1_execution_layout import cpu_layout_context
        context = cpu_layout_context(model, cpu_batch, "cuda")
    with context:
        batch = cpu_batch.to("cuda", non_blocking=True)
        return _optimizer_step(model, optimizer, batch, mean, std, mode)


def _runtime_provenance(context, recipe_path, initial_state_path, runtime, mode):
    if sha256_file(recipe_path) != context.training_recipe_sha256:
        raise ValueError("Runtime recipe bytes differ from Spec")
    recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    validate_recipe(recipe, mode=mode)
    return {"format": "molgap-k1-runtime-provenance-v1", "context": context.to_dict(),
        "mode": mode, "recipe_sha256": sha256_file(recipe_path),
        "initial_state_file_sha256": sha256_file(initial_state_path),
        "initialization_sha256": recipe["initialization_sha256"],
        "runtime_fingerprint": runtime["runtime_fingerprint"],
        "row_order_fingerprint": recipe["row_order_fingerprint"]}


def _validate_arm_binding(spec, context, mode, recipe):
    seed = _recipe_seed(recipe)
    declaration = next(arm for arm in spec.to_dict()["arms"] if arm["arm_id"] == context.arm_id)
    if (declaration["initialization"] != {"kind": "frozen_state", "seed": seed,
                                          "state_sha256": recipe["initialization_sha256"]} or
        declaration["training"]["sampler"]["sha256"] != recipe["row_order_fingerprint"] or
        declaration["training"]["overrides"]):
        raise ValueError("K1 executable initialization/sampler/overrides differ from Spec")
    addons = declaration["addons"]
    if mode == "reference" and addons:
        raise ValueError("Reference must declare no addon")
    if mode == "ssma":
        expected = {"name": "k1_joint_aggregation", "version": "1",
            "source_sha256": normalized_source_sha256(Path(__file__).with_name("k1_joint_aggregation.py")),
            "config": {"degree_policy": "original-sum-above-four", "kappa": 4,
                       "latent_channels": 64, "layer": 6, "seed": seed}}
        if addons != [expected]:
            raise ValueError("SSMA executable addon differs from Spec")
    if mode == "mean2":
        expected = {"name": "k1_two_pass_mean", "version": "1", "config": {},
            "source_sha256": normalized_source_sha256(Path(__file__))}
        if addons != [expected]:
            raise ValueError("Two-pass executable addon differs from Spec")
    if mode == "mean2_clean_second":
        expected = {"name": "k1_mean2_clean_second", "version": "1", "config": {},
            "source_sha256": normalized_source_sha256(Path(__file__).with_name("k1_clean_second.py"))}
        if addons != [expected]:
            raise ValueError("Clean-second executable addon differs from Spec")
    if mode == "fused_layout":
        expected = {"name": "k1_fused_layout", "version": "1", "config": {},
            "source_sha256": normalized_source_sha256(Path(__file__))}
        if addons != [expected]:
            raise ValueError("Fused-layout executable addon differs from Spec")


def validate_screen_recipe(spec, arm_id, recipe):
    """Static family validation shared with preparation; no molecular roles."""
    from types import SimpleNamespace
    from .experiment_execution import training_adapter
    arm = next(a for a in spec.to_dict()["arms"] if a["arm_id"] == arm_id)
    mode = training_adapter(arm).mode(arm)
    validate_recipe(recipe, mode=mode)
    _validate_arm_binding(spec, SimpleNamespace(arm_id=arm_id), mode, recipe)
    if recipe.get("development_role_identity") != "pcqm4mv2-ogb-fixed-100k-v1:development-100000-150000":
        raise ValueError("K1 recipe lacks the fixed development role identity")
    from .experiment_family_workflow import EXPECTED
    if set(recipe["acceptance_requirements"]) != EXPECTED:
        raise ValueError("K1 recipe lacks frozen development row/target hashes")
    return {"family": arm["family"], "mode": mode, "recipe": "frozen_recipe_verified"}


def _validate_prospective(spec, context, input_root, trajectory_id):
    from .experiment_execution import validate_staged_trajectory
    trajectory = validate_staged_trajectory(spec, context.arm_id, Path(input_root))
    if (trajectory["trajectory_id"] != trajectory_id or
        trajectory["state_at_start"]["source_commit"] != context.source_commit):
        raise ValueError("K1 prospective trajectory/source mismatch")


def _allocation_costs(seconds, hardware, *, scope):
    return [{"metric": "wall_seconds", "unit": "seconds", "value": seconds,
             "status": "measured", "semantics": "process_wall", "hardware": hardware, "scope": scope},
            {"metric": "device_seconds", "unit": "seconds", "value": seconds,
             "status": "measured", "semantics": "allocated_device", "hardware": hardware, "scope": scope}]


def validate_runtime_preflight(directory, provenance):
    directory = Path(directory)
    saved = json.loads((directory / "runtime_provenance.json").read_text(encoding="utf-8"))
    certificate = json.loads((directory / "runtime_certificate.json").read_text(encoding="utf-8"))
    architecture = json.loads((directory / "architecture_preflight.json").read_text(encoding="utf-8"))
    runtime = json.loads((directory / "runtime_manifest.json").read_text(encoding="utf-8"))
    if saved != provenance or certificate.get("provenance_sha256") != canonical_fingerprint(provenance):
        raise ValueError("K1 runtime preflight identity changed")
    if (certificate.get("status") != "accepted" or
        certificate.get("calibration_checks_passed") is not True or
        certificate.get("architecture_sha256") != canonical_fingerprint(architecture) or
        architecture.get("accepted") is not True):
        raise ValueError("K1 runtime preflight is not qualified")
    if (runtime.get("runtime_fingerprint") != provenance["runtime_fingerprint"] or
        certificate.get("runtime_fingerprint") != provenance["runtime_fingerprint"] or
        architecture.get("repeatability", {}).get("accepted") is not True or
        architecture.get("resume_roundtrip", {}).get("accepted") is not True or
        architecture.get("zero_initialization_delta") != 0.0 or
        architecture.get("maximum_overhead_fraction") != _maximum_overhead(provenance.get("mode", "reference")) or
        not math.isfinite(architecture.get("synchronized_step_overhead_fraction", math.inf)) or
        architecture["synchronized_step_overhead_fraction"] > architecture["maximum_overhead_fraction"]):
        raise ValueError("K1 runtime calibration evidence differs from the gate")
    return certificate


def run_screen_preflight(*, spec, package_dir: Path, expected_package_identity: str,
                         arm_id: str, mode: str, recipe_path: Path, initial_state_path: Path,
                         input_root: Path, output: Path, account: str, run_reference: str,
                         trajectory_id: str, label_cache=None) -> dict:
    """Isolated real-training-batch qualification; never counts as formal exposure."""
    import torch
    from .experiment_family_workflow import RunContext
    from .training_reproducibility import capture_rng_state, restore_rng_state
    if mode not in ("reference", "ssma", "mean2", "mean2_clean_second", "fused_layout") or label_cache is not None:
        raise ValueError("This paired GPU qualification supports registered screen modes only")
    recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    validate_recipe(recipe, mode=mode)
    seed = _recipe_seed(recipe)
    row_order_sha256 = recipe["row_order_fingerprint"]
    context = RunContext.for_training(spec, package_dir,
        expected_package_identity=expected_package_identity, arm_id=arm_id,
        account=account, run_reference=run_reference)
    _validate_arm_binding(spec, context, mode, recipe)
    _validate_prospective(spec, context, input_root, trajectory_id)
    if (context.family_name, context.family_version) != ("neural_atom_k1", "2"):
        raise ValueError("Preflight requires frozen K1 family/initialization")
    determinism = configure_fp32_determinism(seed)
    runtime = build_runtime_manifest(determinism)
    hardware = torch.cuda.get_device_name(0)
    if "T4" not in hardware:
        raise RuntimeError("Paired qualification requires an exclusive assigned T4")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    provenance = _runtime_provenance(context, recipe_path, initial_state_path, runtime, mode)
    atomic_json(output / "runtime_provenance.json", provenance)
    atomic_json(output / "runtime_manifest.json", runtime)
    state = _load_initial_state(initial_state_path, recipe)
    if compute_row_order_fingerprint(seed) != row_order_sha256:
        raise RuntimeError("Frozen row-order implementation changed")
    root, manifest = find_fixed_cache(input_root)
    roles = load_roles(root, manifest)
    mean_value, std_value = _target_stats(roles["train"])
    cpu_fixture = next(iter(_train_loader(roles["train"], 0, seed)))
    batch = cpu_fixture.clone().to("cuda")
    if int(batch.num_graphs) != BATCH_SIZE:
        raise RuntimeError("Runtime calibration requires physical train batch128")
    mean, std = torch.tensor(mean_value, device="cuda"), torch.tensor(std_value, device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    reference = _make_screen_model(state, "reference", seed).eval()
    candidate = _make_screen_model(state, mode, seed).eval()
    with torch.no_grad():
        if mode == "fused_layout":
            from .k1_execution_layout import cpu_layout_context
            fixture = cpu_fixture.clone()
            with cpu_layout_context(candidate, fixture, "cuda"):
                candidate_prediction = _forward(candidate, fixture.to("cuda"))
        else:
            candidate_prediction = _forward(candidate, batch)
        zero_delta = float((_forward(reference, batch) - candidate_prediction).abs().max())
    if zero_delta != 0.0:
        raise RuntimeError("Zero-added initialization differs from frozen reference")
    del reference, candidate
    losses, states = [], []
    for _ in range(2):
        configure_fp32_determinism(seed)
        model = _make_screen_model(state, mode, seed).train()
        optimizer = _screen_optimizer(model, mode)
        _screen_step(model, optimizer, cpu_fixture.clone(), mean, std, mode)
        loss, _, _ = _screen_step(model, optimizer, cpu_fixture.clone(), mean, std, mode)
        if mode == "ssma" and not any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            and float(parameter.grad.abs().sum()) > 0
            for parameter in model.k1_joint_aggregation.parameters()):
            raise RuntimeError("SSMA mechanism has no finite nonzero training gradient")
        losses.append(float(loss.cpu()))
        states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
        del model, optimizer
    repeated = certify_numerical_repeatability(losses=losses, states=states)
    del states
    # Check the same model/optimizer/scheduler/RNG serialization as epoch resume.
    configure_fp32_determinism(seed)
    model = _make_screen_model(state, mode, seed).train()
    optimizer = _screen_optimizer(model, mode)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    _screen_step(model, optimizer, cpu_fixture.clone(), mean, std, mode)
    scheduler.step()
    snapshot = output / "diagnostic_resume.pt"
    atomic_torch_save(snapshot, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "rng_state": capture_rng_state(),
        "cursor": {"epoch": 1, "next_batch": 0, "sampler_order_sha256": row_order_sha256}})
    uninterrupted, _, _ = _screen_step(model, optimizer, cpu_fixture.clone(), mean, std, mode)
    continuous_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    restored = torch.load(snapshot, map_location="cpu", weights_only=False)
    model.load_state_dict(restored["model"], strict=True)
    optimizer.load_state_dict(restored["optimizer"])
    scheduler.load_state_dict(restored["scheduler"])
    restore_rng_state(restored["rng_state"])
    resumed, _, _ = _screen_step(model, optimizer, cpu_fixture.clone(), mean, std, mode)
    resume = certify_numerical_repeatability(losses=[float(uninterrupted.cpu()), float(resumed.cpu())],
        states=[continuous_state, model.state_dict()])
    selected_path = output / "diagnostic_selected.pt"
    atomic_torch_save(selected_path, _clean_gap_state(model))
    selected_state = torch.load(selected_path, map_location="cpu", weights_only=True)
    selected_model = _make_screen_model(state, mode, seed).eval()
    selected_model.load_state_dict(selected_state, strict=True)
    model.eval()
    with torch.no_grad():
        selected_delta = float((_forward(model, batch) - _forward(selected_model, batch)).abs().max())
    if selected_delta != 0.0:
        raise RuntimeError("Selected Gap state roundtrip changed inference")
    del selected_model, selected_state
    del model, optimizer, scheduler, continuous_state, restored
    timings = {}
    for profile_mode in ("reference", mode):
        configure_fp32_determinism(seed)
        model = _make_screen_model(state, profile_mode, seed).train()
        optimizer = _screen_optimizer(model, profile_mode)
        for _ in range(2):
            _screen_step(model, optimizer, cpu_fixture.clone(), mean, std, profile_mode)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(5):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            _screen_step(model, optimizer, cpu_fixture.clone(), mean, std, profile_mode)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - tick)
        timings[profile_mode] = {"median_step_seconds": float(np.median(samples)),
            "samples_seconds": samples, "peak_memory_bytes": torch.cuda.max_memory_allocated()}
        del model, optimizer
    overhead = timings[mode]["median_step_seconds"] / timings["reference"]["median_step_seconds"] - 1
    maximum_overhead = _maximum_overhead(mode)
    architecture = {"accepted": overhead <= maximum_overhead, "mode": mode, "zero_initialization_delta": zero_delta,
        "repeated_optimizer_steps": 2,
        "repeatability": repeated, "resume_roundtrip": resume, "timings": timings,
        "selected_state_roundtrip_delta": selected_delta,
        "synchronized_step_overhead_fraction": overhead, "maximum_overhead_fraction": maximum_overhead,
        "formal_sample_presentations": 0, "fixture_sha256": _batch_sha256(batch),
        "resume_scope": "model/AdamW/cosine/RNG roundtrip; next step on fixed fixture; no epoch consumption"}
    atomic_json(output / "architecture_preflight.json", architecture)
    torch.cuda.synchronize()
    atomic_json(output / "diagnostic_cost.json", {"costs": _allocation_costs(
        time.perf_counter() - started, hardware, scope="diagnostic_window_excludes_bootstrap_queue"),
        "scope": "one exclusive assigned T4 diagnostic window; bootstrap and queue excluded",
        "cpu_allocation_seconds": {"status": "missing", "value": None},
        "queue_seconds": {"status": "missing", "value": None}})
    if not architecture["accepted"]:
        raise RuntimeError("Registered mode exceeds synchronized optimizer-inclusive overhead gate")
    certificate = {"format": "molgap-runtime-certificate-v1", "status": "accepted",
        "platform_id": recipe.get("runtime_platform_id", context.platform + "-t4x2"),
        "accelerator": hardware, "precision": "fp32", "tf32_enabled": False,
        "deterministic_algorithms": True, "physical_batch_per_device": BATCH_SIZE,
        "tail_batch_policy": "drop_last", "runtime_fingerprint": runtime["runtime_fingerprint"],
        "software_fingerprint": runtime["installed_distributions_sha256"],
        "determinism_fingerprint": canonical_fingerprint(determinism),
        "calibration_fixture_sha256": architecture["fixture_sha256"],
        "calibration_output_sha256": repeated["state_sha256"][0],
        "calibration_checks_passed": True, "provenance_sha256": canonical_fingerprint(provenance),
        "architecture_sha256": canonical_fingerprint(architecture)}
    atomic_json(output / "runtime_certificate.json", certificate)
    return certificate


def run_screen_arm(*, spec, package_dir: Path, expected_package_identity: str,
                   arm_id: str, mode: str, recipe_path: Path, initial_state_path: Path,
                   input_root: Path, output: Path, account: str, run_reference: str,
                   trajectory_id: str, label_cache=None, preflight_dir: Path | None = None) -> dict:
    """Execute the owning V4 epoch loop with shared durable-output hooks.

    Resume is restricted to acknowledged complete epochs. Re-created epoch
    loaders retain the original Python row order and worker setup. No EMA,
    corruption, pretraining, alternate geometry cache or protected role is used.
    """
    import torch
    from .experiment_family_workflow import FamilyOutputSession, RunContext
    from .training_reproducibility import capture_rng_state, restore_rng_state

    recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    validate_recipe(recipe, mode=mode)
    seed = _recipe_seed(recipe)
    row_order_sha256 = recipe["row_order_fingerprint"]
    context = RunContext.for_training(spec, package_dir,
        expected_package_identity=expected_package_identity, arm_id=arm_id,
        account=account, run_reference=run_reference)
    _validate_arm_binding(spec, context, mode, recipe)
    _validate_prospective(spec, context, input_root, trajectory_id)
    if (context.family_name, context.family_version) != ("neural_atom_k1", "2"):
        raise ValueError("Expected registered K1 family output adapter")
    if mode in ("reference", "ssma", "mean2", "mean2_clean_second") and label_cache is not None:
        raise ValueError("Reference/SSMA does not consume chemical auxiliary labels")
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
    if compute_row_order_fingerprint(seed) != row_order_sha256:
        raise RuntimeError("Frozen row-order implementation changed")
    determinism = configure_fp32_determinism(seed)
    runtime = build_runtime_manifest(determinism)
    provenance = _runtime_provenance(context, recipe_path, initial_state_path, runtime, mode)
    certificate = validate_runtime_preflight(preflight_dir or output, provenance)
    hardware = torch.cuda.get_device_name(0)
    if certificate.get("accelerator") != hardware or "T4" not in hardware:
        raise ValueError("Runtime accelerator differs from qualified T4")
    Path(output).mkdir(parents=True, exist_ok=True)
    atomic_json(Path(output) / "runtime_provenance.json", provenance)
    root, manifest = find_fixed_cache(input_root)
    roles = load_roles(root, manifest)
    mean_value, std_value = _target_stats(roles["train"])
    # Load the exact backbone before constructing any extra trainable mechanism.
    state = _load_initial_state(initial_state_path, recipe)
    model = _make_screen_model(state, mode, seed)
    optimizer = _screen_optimizer(model, mode)
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
    atomic_json(session.root / "runtime_manifest.json", runtime)
    atomic_json(session.root / "runtime_certificate.json", certificate)
    start_epoch, best = 0, math.inf
    checkpoint = session.root / "last_checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["context"] != context.to_dict():
            raise ValueError("K1 resume source/arm identity mismatch")
        cursor = saved["cursor"]
        start_epoch = cursor["epoch"]
        if (type(start_epoch) is not int or not 1 <= start_epoch <= EPOCHS or
            cursor["next_batch"] != 0 or cursor["sampler_order_sha256"] != row_order_sha256 or
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
    torch.cuda.synchronize()
    allocation_started = time.perf_counter()
    timing_path = session.root / "execution_times.json"
    timing_rows = json.loads(timing_path.read_text()) if timing_path.exists() else []
    if not timing_path.exists() and start_epoch and mode != "fused_layout":
        timing_rows = [{"epoch": index + 1, "timing": None,
                        "reason": "Historical checkpoint predates execution timing retention"}
                       for index in range(start_epoch)]
    if len(timing_rows) != start_epoch:
        raise ValueError("Execution timing and complete-epoch cursor disagree")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        absolute, rows = 0.0, 0
        started = time.perf_counter()
        for batch in _train_loader(roles["train"], epoch, seed):
            if label_cache is not None:
                label_cache.attach(batch)
            _, batch_absolute, batch_rows = _screen_step(model, optimizer, batch, mean, std, mode)
            absolute += float(batch_absolute)
            rows += batch_rows
        if rows != ROWS_PER_EPOCH:
            raise RuntimeError("Frozen optimizer exposure changed")
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        dev_started = time.perf_counter()
        validation_mae, target_eV, prediction_eV, source_idx = _evaluate(
            model, development_loader, mean, std)
        torch.cuda.synchronize()
        dev_seconds = time.perf_counter() - dev_started
        step, presentations = (epoch + 1) * STEPS_PER_EPOCH, (epoch + 1) * ROWS_PER_EPOCH
        if validation_mae < best:
            best = validation_mae
            session.selected(model_state=_clean_gap_state(model), epoch=epoch + 1,
                optimizer_step=step, weights="live", prediction_eV=prediction_eV,
                target_eV=target_eV, source_idx=source_idx)
        observed_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        session.checkpoint(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(), rng_state=capture_rng_state(),
            cursor={"epoch": epoch + 1, "next_batch": 0,
                    "sampler_order_sha256": row_order_sha256},
            optimizer_step=step, sample_presentations=presentations)
        torch.cuda.synchronize()
        session.epoch_finished(epoch=epoch + 1, optimizer_step=step,
            sample_presentations=presentations, live_dev_metric=validation_mae,
            live_train_metric=absolute / rows * std_value, learning_rate=observed_lr,
            checkpoint_identity="sha256:" + sha256_file(checkpoint),
            wall_time_seconds=time.perf_counter() - started)
        timing_rows.append({"epoch": epoch + 1, "optimizer_steps": step,
            "sample_presentations": presentations, "train_seconds": train_seconds,
            "development_seconds": dev_seconds, "epoch_pipeline_seconds": time.perf_counter() - started,
            "scope": "train includes loader/H2D/optimizer; development is common clean inference; pipeline includes selected/last IO"})
        atomic_json(timing_path, timing_rows)
        print(f"{arm_id} epoch={epoch+1} train_normalized_mae={absolute/rows:.6f} "
              f"dev={validation_mae:.6f}eV seconds={time.perf_counter()-started:.1f}", flush=True)
    torch.cuda.synchronize()
    costs = _allocation_costs(time.perf_counter() - allocation_started, hardware,
        scope="training_invocation_lower_bound_includes_dev_checkpoint_excludes_bootstrap_queue")
    if start_epoch:
        costs[1].update(value=None, status="missing",
            reason="Previous resume segments lack a retained allocated-device ledger")
    atomic_json(session.root / "allocation_cost.json", {"costs": costs,
        "start_epoch": start_epoch, "end_epoch": EPOCHS,
        "cpu_allocation_seconds": {"status": "missing", "value": None},
        "queue_seconds": {"status": "missing", "value": None},
        "scope": "one exclusive assigned T4 training window including development/checkpoint; "
                 "lower bound excludes bootstrap/queue and earlier resume allocations"})
    return session.complete(runtime={"platform": context.platform, "account": context.account,
        "precision": "fp32", "source_commit": context.source_commit,
        "source_archive_sha256": context.source_archive_sha256,
        "runtime_certificate_id": canonical_fingerprint(certificate)}, hardware=hardware,
        observed_costs=costs)
