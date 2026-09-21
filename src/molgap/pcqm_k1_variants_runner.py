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
    _return_allocation_update,
    _single_slot_processor_update,
    _tied_selector_single_slot_update,
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
FIXED_DATASET = os.environ.get(
    "MOLGAP_FIXED_DATASET", "kaseichou/pcqm4mv2-ogb-fixed-100k-v1"
)
FIXED_MANIFEST_SHA256 = (
    "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
)
FIXED_GEOMETRY_SHA256 = (
    "bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5"
)
BENCHMARK_ID = "pcqm4mv2-ogb-fixed-100k-gap-v4"
PLATFORM_ID = os.environ.get("MOLGAP_PLATFORM_ID", "kaggle2")
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
MOSE_FEATURE_FINGERPRINT = _hash_mapping(
    {
        "atom": "ogb-nine-categorical",
        "bond": "ogb-three-categorical-real-bonds",
        "mose": "rooted-all5-plus-c6-log1p-31",
        "rwse": False,
        "geometry_model_input": False,
    }
)
MOSE_RWSE_FEATURE_FINGERPRINT = _hash_mapping(
    {
        "atom": "ogb-nine-categorical",
        "bond": "ogb-three-categorical-real-bonds",
        "rwse": 16,
        "mose": "rooted-all5-plus-c6-log1p-31",
        "geometry_model_input": False,
    }
)
MOSE_REPLACEMENT_MODES = (
    "neural_atom_k1_mose",
    "neural_atom_k1_mose_hidden_bn",
)
MOSE_GATED_DUAL_MODES = (
    "neural_atom_k1_rwse_mose_residual_gate",
    "neural_atom_k1_rwse_mose_context_gate",
)
MOSE_PAIR_TOKEN_MODES = ("neural_atom_k1_pair_token_mose",)
MOSE_DUAL_MODES = MOSE_GATED_DUAL_MODES + MOSE_PAIR_TOKEN_MODES
MOSE_MODES = MOSE_REPLACEMENT_MODES + MOSE_DUAL_MODES
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
    def load(path: Path, *, retain_wedge_topology: bool = False):
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
            if field == "wedge_edge_ids" and retain_wedge_topology:
                continue
            if field in payload._data:
                del payload._data[field]
                payload.slices.pop(field, None)
        return payload


def find_fixed_cache() -> tuple[Path, dict]:
    candidates = []
    explicit_root = os.environ.get("MOLGAP_FIXED_CACHE_ROOT")
    search_roots = [Path(explicit_root)] if explicit_root else [Path("/kaggle/input")]
    for search_root in search_roots:
        paths = (
            [search_root / "manifest.json"]
            if (search_root / "manifest.json").is_file()
            else search_root.rglob("manifest.json")
        )
        for path in paths:
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


def load_roles(root: Path, manifest: dict, *, retain_wedge_topology: bool = False):
    import torch
    from torch.utils.data import ConcatDataset

    roles = {"train": [], "development": []}
    aggregate = hashlib.sha256()
    expected_start = {"train": 0, "development": TRAIN_ROWS}
    for item in manifest["geometry_shards"]:
        path = root / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Fixed shard changed: {item['file']}")
        payload = _PackedGraphDatasetFactory.load(
            path, retain_wedge_topology=retain_wedge_topology
        )
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
    if getattr(model, "requires_wedge_topology", False):
        return model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
            batch.wedge_edge_ids,
        ).view(-1)
    if getattr(model, "requires_functional_group_membership", False):
        return model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
            batch.functional_group_y,
        ).view(-1)
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


def _shared_k1_state_sha256(model, mode: str) -> str:
    """Hash the unchanged K1 state retained by an isolated candidate."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if mode in MOSE_REPLACEMENT_MODES and "rwse_encoder." in name:
            continue
        if mode == "neural_atom_k1_tied_selector" and (
            ".node_key." in name or ".slot_query." in name
        ):
            continue
        if mode in {
            "neural_atom_k1_collapsed_mha",
            "neural_atom_k1_no_slot_attention",
            "neural_atom_k1_no_attention_uniform_return",
            "neural_atom_k1_edge_context_no_slot_attention",
        } and ".slot_attention." in name:
            continue
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _batch_sha256(batch) -> str:
    digest = hashlib.sha256()
    names = ["x", "edge_index", "edge_attr", "batch", "random_walk_pe", "y", "source_idx"]
    if hasattr(batch, "functional_group_y"):
        names.append("functional_group_y")
    if hasattr(batch, "wedge_edge_ids"):
        names.append("wedge_edge_ids")
    for name in names:
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


def build_runtime_certificate(
    roles,
    output: Path,
    *,
    calibration_mode: str = "neural_atom_k1_v4",
) -> tuple[dict, dict]:
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
        model = make_encoder(calibration_mode).to("cuda").train()
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
        "calibration_model_id": calibration_mode,
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
    from .k1_edge_memory import MODES as EDGE_MEMORY_MODES, PARAMETERS, check_mechanism
    from .k1_edge_slot_interaction import (
        MODES as EDGE_SLOT_MODES,
        PARAMETERS as EDGE_SLOT_PARAMETERS,
        check_mechanism as check_edge_slot,
    )
    from .k1_edge_conditioned_slot import (
        MODES as EDGE_CONDITIONED_MODES,
        PARAMETERS as EDGE_CONDITIONED_PARAMETERS,
        check_mechanism as check_edge_conditioned,
    )
    from .k1_gpspp_local import (
        MODES as GPSPP_LOCAL_MODES,
        PARAMETERS as GPSPP_LOCAL_PARAMETERS,
        check_mechanism as check_gpspp_local,
    )
    from .k1_pair_token import (
        MODES as PAIR_TOKEN_MODES,
        PARAMETERS as PAIR_TOKEN_PARAMETERS,
        check_mechanism as check_pair_token,
    )
    from .k1_functional_group_token import (
        MODES as FUNCTIONAL_GROUP_MODES,
        PARAMETERS as FUNCTIONAL_GROUP_PARAMETERS,
        check_mechanism as check_functional_group,
    )
    from .k1_chem_typed_pair_token import (
        MODES as CHEM_TYPED_PAIR_MODES,
        PARAMETERS as CHEM_TYPED_PAIR_PARAMETERS,
        check_mechanism as check_chem_typed_pair,
    )
    from .k1_multiplicative_pair_value import (
        MODES as MULTIPLICATIVE_PAIR_MODES,
        PARAMETERS as MULTIPLICATIVE_PAIR_PARAMETERS,
        check_mechanism as check_multiplicative_pair,
    )
    from .k1_sparse_triplet import (
        MODES as SPARSE_TRIPLET_MODES,
        PARAMETERS as SPARSE_TRIPLET_PARAMETERS,
        check_mechanism as check_sparse_triplet,
    )
    from .k1_spd_pair_token import (
        MODES as SPD_PAIR_TOKEN_MODES,
        PARAMETERS as SPD_PAIR_TOKEN_PARAMETERS,
        check_mechanism as check_spd_pair_token,
    )
    from .k1_oneshot_triplet_pair_token import (
        MODES as ONESHOT_TRIPLET_MODES,
        PARAMETERS as ONESHOT_TRIPLET_PARAMETERS,
        check_mechanism as check_oneshot_triplet,
    )
    from .k1_pair_token_mose import (
        MODES as PAIR_TOKEN_MOSE_MODES,
        PARAMETERS as PAIR_TOKEN_MOSE_PARAMETERS,
        check_mechanism as check_pair_token_mose,
    )
    active_edge_modes = EDGE_MEMORY_MODES + EDGE_SLOT_MODES
    recoverable_modes = (
        active_edge_modes
        + EDGE_CONDITIONED_MODES
        + GPSPP_LOCAL_MODES
        + FUNCTIONAL_GROUP_MODES
        + CHEM_TYPED_PAIR_MODES
        + MULTIPLICATIVE_PAIR_MODES
        + SPARSE_TRIPLET_MODES
        + SPD_PAIR_TOKEN_MODES
        + ONESHOT_TRIPLET_MODES
        + PAIR_TOKEN_MOSE_MODES
    )
    import torch

    configure_fp32_determinism(SEED)
    baseline = make_encoder("neural_atom_k1_v4").to("cuda").eval()
    baseline_sha = _shared_k1_state_sha256(baseline, mode)
    configure_fp32_determinism(SEED)
    model = make_encoder(mode).to("cuda").eval()
    shared_sha = _shared_k1_state_sha256(_base_state(model), mode)
    if shared_sha != baseline_sha:
        raise RuntimeError(f"K1 shared initialization changed for {mode}")
    batch = next(iter(_train_loader(roles["train"], 0))).to("cuda", non_blocking=True)
    if mode in MOSE_REPLACEMENT_MODES:
        with torch.no_grad():
            candidate_prediction = _forward(model, batch)
        exact_nested_initialization = False
        mechanism_checks = {
            "mose_input_shape": list(batch.random_walk_pe.shape),
            "mose_input_finite": bool(torch.isfinite(batch.random_walk_pe).all()),
            "mose_input_nonnegative": bool((batch.random_walk_pe >= 0).all()),
            "rwse_replaced_not_concatenated": model.rwse_dim == 31,
            "hidden_batchnorm": any(
                isinstance(module, torch.nn.BatchNorm1d)
                for module in model.rwse_encoder.modules()
            ),
            "candidate_output_finite": bool(torch.isfinite(candidate_prediction).all()),
        }
        expected_checks = {
            "mose_input_shape": [int(batch.num_nodes), 31],
            "mose_input_finite": True,
            "mose_input_nonnegative": True,
            "rwse_replaced_not_concatenated": True,
            "hidden_batchnorm": mode == "neural_atom_k1_mose_hidden_bn",
            "candidate_output_finite": True,
        }
        if mechanism_checks != expected_checks:
            raise RuntimeError(f"MoSE feature invariant failed: {mechanism_checks}")
    elif mode in PAIR_TOKEN_MOSE_MODES:
        with torch.no_grad():
            baseline_prediction = baseline(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.random_walk_pe[:, :16],
            ).view(-1)
            candidate_prediction = _forward(model, batch)
        exact_nested_initialization = bool(
            torch.equal(baseline_prediction, candidate_prediction)
        )
        mechanism_checks = check_pair_token_mose(model, batch)
        mechanism_checks["exact_k1_output"] = exact_nested_initialization
        if (
            mechanism_checks.get("combined_input_shape")
            != [int(batch.num_nodes), 47]
            or exact_nested_initialization is not True
            or sum(parameter.numel() for parameter in model.parameters())
            != PAIR_TOKEN_MOSE_PARAMETERS[mode]
        ):
            raise RuntimeError(
                f"PairToken+MoSE preflight identity failed: {mechanism_checks}"
            )
    elif mode in MOSE_GATED_DUAL_MODES:
        with torch.no_grad():
            baseline_prediction = baseline(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.random_walk_pe[:, :16],
            ).view(-1)
            candidate_prediction = _forward(model, batch)
        exact_nested_initialization = bool(
            torch.equal(baseline_prediction, candidate_prediction)
        )
        mechanism_checks = {
            "combined_input_shape": list(batch.random_walk_pe.shape),
            "rwse_channels": 16,
            "mose_channels": 31,
            "mose_input_finite": bool(
                torch.isfinite(batch.random_walk_pe[:, 16:]).all()
            ),
            "mose_input_nonnegative": bool(
                (batch.random_walk_pe[:, 16:] >= 0).all()
            ),
            "zero_initialized_residual_output": bool(
                torch.count_nonzero(model.mose_residual[-1].weight).item() == 0
                and torch.count_nonzero(model.mose_residual[-1].bias).item() == 0
            ),
            "gate_scope": model.gate_scope,
            "exact_k1_output": exact_nested_initialization,
            "candidate_output_finite": bool(torch.isfinite(candidate_prediction).all()),
        }
        if mechanism_checks != {
            "combined_input_shape": [int(batch.num_nodes), 47],
            "rwse_channels": 16,
            "mose_channels": 31,
            "mose_input_finite": True,
            "mose_input_nonnegative": True,
            "zero_initialized_residual_output": True,
            "gate_scope": (
                "molecule-context"
                if mode == "neural_atom_k1_rwse_mose_context_gate"
                else "node-mose-mean"
            ),
            "exact_k1_output": True,
            "candidate_output_finite": True,
        }:
            raise RuntimeError(
                f"RWSE/MoSE residual invariant failed: {mechanism_checks}"
            )
    else:
        with torch.no_grad():
            baseline_prediction = _forward(baseline, batch)
            candidate_prediction = _forward(model, batch)
        exact_nested_initialization = bool(torch.equal(baseline_prediction, candidate_prediction))
    if mode not in {"neural_atom_k1_v4", *MOSE_MODES} and mode not in active_edge_modes and not exact_nested_initialization:
        raise RuntimeError(f"Candidate is not functionally nested in K1: {mode}")
    if mode not in MOSE_MODES:
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
    elif mode == "neural_atom_k1_repset_readout":
        readout = model.repset_readout
        probe = torch.linspace(
            -1.0,
            1.0,
            steps=int(batch.num_nodes) * 192,
            device="cuda",
        ).reshape(int(batch.num_nodes), 192)
        readout.eval()
        with torch.no_grad():
            correction = readout(probe, batch.batch)
        mechanism_checks = {
            "hidden_sets": readout.n_hidden_sets,
            "elements_per_hidden_set": readout.n_elements,
            "prototype_shape": list(readout.prototype.shape),
            "graph_correction_shape": list(correction.shape),
            "zero_representation_return_exact": bool(
                torch.count_nonzero(correction).item() == 0
            ),
            "target_residual": False,
        }
        if mechanism_checks != {
            "hidden_sets": 8,
            "elements_per_hidden_set": 8,
            "prototype_shape": [192, 64],
            "graph_correction_shape": [int(batch.num_graphs), 192],
            "zero_representation_return_exact": True,
            "target_residual": False,
        }:
            raise RuntimeError(f"RepSet readout invariant failed: {mechanism_checks}")
    elif mode == "neural_atom_k1_tied_selector":
        probe = torch.linspace(
            -1.0,
            1.0,
            steps=int(batch.num_nodes) * 192,
            device="cuda",
        ).reshape(int(batch.num_nodes), 192)
        layer_checks = []
        for layer in MIXER_LAYERS:
            mixer = model.base.neural_atom_mixers[str(layer)]
            update, _, assignment, valid, diagnostics = (
                _tied_selector_single_slot_update(
                    mixer,
                    model.shared_selector,
                    probe,
                    batch.batch,
                )
            )
            layer_checks.append(
                {
                    "layer": layer,
                    "selector_scope": diagnostics["selector_scope"],
                    "value_scope": diagnostics["value_scope"],
                    "allocation_mass_one": bool(
                        torch.allclose(
                            assignment.sum(dim=-1),
                            torch.ones_like(assignment.sum(dim=-1)),
                            atol=1e-6,
                            rtol=0,
                        )
                    ),
                    "padding_mass_zero": bool(
                        assignment.masked_select(~valid.unsqueeze(1))
                        .abs()
                        .sum()
                        .item()
                        == 0.0
                    ),
                    "zero_return_exact": bool(
                        torch.count_nonzero(update).item() == 0
                    ),
                    "selector_modules_removed": bool(
                        not hasattr(mixer, "node_key")
                        and not hasattr(mixer, "slot_query")
                    ),
                }
            )
        independent_values = len(
            {
                id(model.base.neural_atom_mixers[str(layer)].node_value)
                for layer in MIXER_LAYERS
            }
        ) == len(MIXER_LAYERS)
        independent_returns = len(
            {
                id(model.base.neural_atom_mixers[str(layer)].return_projection)
                for layer in MIXER_LAYERS
            }
        ) == len(MIXER_LAYERS)
        mechanism_checks = {
            "shared_selector_count": 1,
            "independent_value_modules": independent_values,
            "independent_return_modules": independent_returns,
            "layers": layer_checks,
        }
        if not (
            independent_values
            and independent_returns
            and all(
                all(
                    value is True
                    for key, value in check.items()
                    if key
                    in {
                        "allocation_mass_one",
                        "padding_mass_zero",
                        "zero_return_exact",
                        "selector_modules_removed",
                    }
                )
                for check in layer_checks
            )
        ):
            raise RuntimeError(f"Tied-selector invariant failed: {mechanism_checks}")
    elif mode in {
        "neural_atom_k1_collapsed_mha",
        "neural_atom_k1_no_slot_attention",
    }:
        probe = torch.linspace(
            -1.0,
            1.0,
            steps=int(batch.num_nodes) * 192,
            device="cuda",
        ).reshape(int(batch.num_nodes), 192)
        layer_checks = []
        for layer in MIXER_LAYERS:
            mixer = model.base.neural_atom_mixers[str(layer)]
            processor = (
                model.collapsed_slot_projections[str(layer)]
                if mode == "neural_atom_k1_collapsed_mha"
                else None
            )
            update, slots, assignment, valid, diagnostics = (
                _single_slot_processor_update(
                    mixer,
                    processor,
                    probe,
                    batch.batch,
                )
            )
            check = {
                "layer": layer,
                "active_slots": diagnostics["active_slots"],
                "slot_processor": diagnostics["slot_processor"],
                "slot_attention_module_removed": diagnostics[
                    "slot_attention_module_removed"
                ],
                "slot_shape": list(slots.shape),
                "allocation_mass_one": bool(
                    torch.allclose(
                        assignment.sum(dim=-1),
                        torch.ones_like(assignment.sum(dim=-1)),
                        atol=1e-6,
                        rtol=0,
                    )
                ),
                "padding_mass_zero": bool(
                    assignment.masked_select(~valid.unsqueeze(1))
                    .abs()
                    .sum()
                    .item()
                    == 0.0
                ),
                "zero_return_exact": bool(
                    torch.count_nonzero(update).item() == 0
                ),
            }
            if mode == "neural_atom_k1_collapsed_mha":
                reference_mixer = baseline.neural_atom_mixers[str(layer)]
                raw_slots = torch.linspace(
                    -0.5,
                    0.5,
                    steps=int(batch.num_graphs) * 64,
                    device="cuda",
                ).reshape(int(batch.num_graphs), 1, 64)
                with torch.no_grad():
                    expected_attention, _ = reference_mixer.slot_attention(
                        raw_slots,
                        raw_slots,
                        raw_slots,
                        need_weights=False,
                    )
                    collapsed_attention = processor(raw_slots)
                check["matches_length_one_attention_in_eval"] = bool(
                    torch.allclose(
                        expected_attention,
                        collapsed_attention,
                        atol=1e-6,
                        rtol=1e-6,
                    )
                )
            layer_checks.append(check)
        expected_processor = (
            "collapsed-value-output"
            if mode == "neural_atom_k1_collapsed_mha"
            else "none"
        )
        mechanism_checks = {
            "attention_sequence_length": 1,
            "slot_processor": expected_processor,
            "layers": layer_checks,
        }
        required_flags = {
            "slot_attention_module_removed",
            "allocation_mass_one",
            "padding_mass_zero",
            "zero_return_exact",
        }
        if mode == "neural_atom_k1_collapsed_mha":
            required_flags.add("matches_length_one_attention_in_eval")
        if not all(
            check["active_slots"] == 1
            and check["slot_processor"] == expected_processor
            and check["slot_shape"] == [int(batch.num_graphs), 1, 64]
            and all(check[name] is True for name in required_flags)
            for check in layer_checks
        ):
            raise RuntimeError(
                f"Single-slot processor invariant failed: {mechanism_checks}"
            )
    elif mode in {
        "neural_atom_k1_uniform_return",
        "neural_atom_k1_inverse_return",
        "neural_atom_k1_no_attention_uniform_return",
    }:
        return_mode = (
            "uniform"
            if mode in {
                "neural_atom_k1_uniform_return",
                "neural_atom_k1_no_attention_uniform_return",
            }
            else "inverse-score"
        )
        remove_slot_attention = (
            mode == "neural_atom_k1_no_attention_uniform_return"
        )
        probe = torch.linspace(
            -1.0,
            1.0,
            steps=int(batch.num_nodes) * 192,
            device="cuda",
        ).reshape(int(batch.num_nodes), 192)
        layer_checks = []
        for layer in MIXER_LAYERS:
            mixer = model.base.neural_atom_mixers[str(layer)]
            (
                update,
                slots,
                source_assignment,
                return_assignment,
                valid,
                diagnostics,
            ) = _return_allocation_update(
                mixer,
                probe,
                batch.batch,
                return_mode,
                remove_slot_attention=remove_slot_attention,
            )
            source_padding = source_assignment.masked_select(
                ~valid.unsqueeze(1)
            ).abs().sum()
            return_padding = return_assignment.masked_select(
                ~valid.unsqueeze(1)
            ).abs().sum()
            check = {
                "layer": layer,
                "active_slots": diagnostics["active_slots"],
                "source_allocation": diagnostics["source_allocation"],
                "return_allocation": diagnostics["return_allocation"],
                "slot_shape": list(slots.shape),
                "source_mass_one": bool(
                    torch.allclose(
                        source_assignment.sum(dim=-1),
                        torch.ones_like(source_assignment.sum(dim=-1)),
                        atol=1e-6,
                        rtol=0,
                    )
                ),
                "return_mass_one": bool(
                    torch.allclose(
                        return_assignment.sum(dim=-1),
                        torch.ones_like(return_assignment.sum(dim=-1)),
                        atol=1e-6,
                        rtol=0,
                    )
                ),
                "source_padding_mass_zero": bool(source_padding.item() == 0.0),
                "return_padding_mass_zero": bool(return_padding.item() == 0.0),
                "zero_return_exact": bool(torch.count_nonzero(update).item() == 0),
                "slot_attention_module_removed": diagnostics[
                    "slot_attention_module_removed"
                ],
                "slot_processor": diagnostics["slot_processor"],
            }
            if return_mode == "uniform":
                expected = valid.unsqueeze(1).to(return_assignment.dtype)
                expected = expected / expected.sum(dim=-1, keepdim=True)
                check["uniform_return_exact"] = bool(
                    torch.equal(return_assignment, expected)
                )
            else:
                check["return_differs_from_source"] = bool(
                    not torch.equal(return_assignment, source_assignment)
                )
            layer_checks.append(check)
        required = {
            "source_mass_one",
            "return_mass_one",
            "source_padding_mass_zero",
            "return_padding_mass_zero",
            "zero_return_exact",
            (
                "uniform_return_exact"
                if return_mode == "uniform"
                else "return_differs_from_source"
            ),
        }
        if remove_slot_attention:
            required.add("slot_attention_module_removed")
        mechanism_checks = {
            "source_allocation": "learned-softmax-over-atoms",
            "return_allocation": return_mode,
            "added_parameters": 0,
            "slot_processor": (
                "none" if remove_slot_attention else "length-one-self-attention"
            ),
            "layers": layer_checks,
        }
        if not all(
            check["active_slots"] == 1
            and check["source_allocation"] == "learned-softmax-over-atoms"
            and check["return_allocation"] == return_mode
            and check["slot_shape"] == [int(batch.num_graphs), 1, 64]
            and (not remove_slot_attention or check["slot_processor"] == "none")
            and all(check[name] is True for name in required)
            for check in layer_checks
        ):
            raise RuntimeError(
                f"Return-allocation invariant failed: {mechanism_checks}"
            )
    elif mode in EDGE_CONDITIONED_MODES:
        mechanism_checks = check_edge_conditioned(model, batch)
        expected_parameters = EDGE_CONDITIONED_PARAMETERS[mode]
        if sum(parameter.numel() for parameter in model.parameters()) != expected_parameters:
            raise RuntimeError("Edge-conditioned-slot parameter identity changed")
    mean = torch.tensor(target_stats["mean_eV"], device="cuda")
    std = torch.tensor(target_stats["sample_std_eV"], device="cuda")
    if mode in active_edge_modes:
        mechanism_checks = (
            check_mechanism(model, batch)
            if mode in EDGE_MEMORY_MODES
            else check_edge_slot(model, batch)
        )
        expected_parameters = (
            PARAMETERS if mode in EDGE_MEMORY_MODES else EDGE_SLOT_PARAMETERS[mode]
        )
        if sum(p.numel() for p in model.parameters()) != expected_parameters:
            raise RuntimeError("Edge-memory parameter identity changed")
    elif mode in GPSPP_LOCAL_MODES:
        mechanism_checks = check_gpspp_local(model, batch)
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != GPSPP_LOCAL_PARAMETERS[mode]
        ):
            raise RuntimeError("GPSPP-local parameter identity changed")
    elif mode in PAIR_TOKEN_MODES:
        mechanism_checks = check_pair_token(model, batch)
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != PAIR_TOKEN_PARAMETERS[mode]
        ):
            raise RuntimeError("Pair-token parameter identity changed")
    elif mode in FUNCTIONAL_GROUP_MODES:
        mechanism_checks = check_functional_group(model, batch)
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != FUNCTIONAL_GROUP_PARAMETERS[mode]
        ):
            raise RuntimeError("Functional-group-token parameter identity changed")
    elif mode in CHEM_TYPED_PAIR_MODES:
        mechanism_checks = check_chem_typed_pair(model, batch)
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != CHEM_TYPED_PAIR_PARAMETERS[mode]
        ):
            raise RuntimeError("Chem-typed pair-token parameter identity changed")
    elif mode in MULTIPLICATIVE_PAIR_MODES:
        mechanism_checks = check_multiplicative_pair(model, batch)
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != MULTIPLICATIVE_PAIR_PARAMETERS[mode]
        ):
            raise RuntimeError("Multiplicative pair-value parameter identity changed")
    elif mode in SPARSE_TRIPLET_MODES:
        mechanism_checks = check_sparse_triplet(model, batch)
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != SPARSE_TRIPLET_PARAMETERS[mode]
        ):
            raise RuntimeError("Sparse-triplet parameter identity changed")
    elif mode in SPD_PAIR_TOKEN_MODES:
        mechanism_checks = check_spd_pair_token(model, batch)
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != SPD_PAIR_TOKEN_PARAMETERS[mode]
        ):
            raise RuntimeError("SPD PairToken parameter identity changed")
    elif mode in ONESHOT_TRIPLET_MODES:
        mechanism_checks = check_oneshot_triplet(model, batch)
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != ONESHOT_TRIPLET_PARAMETERS[mode]
        ):
            raise RuntimeError("One-shot triplet PairToken parameter identity changed")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    _optimizer_step(model, optimizer, batch, mean, std)
    if mode in recoverable_modes:
        from .k1_edge_memory import snapshot_step, check_resume_equivalence
        snapshot = snapshot_step(model, optimizer)
    _optimizer_step(model, optimizer, batch, mean, std)
    if mode in recoverable_modes:
        mechanism_checks["resume_two_step_bitwise_equal"] = check_resume_equivalence(
            model, optimizer, snapshot, batch, mean, std, _optimizer_step
        )
        del snapshot
    candidate_parameters = []
    if mode in active_edge_modes:
        candidate_parameters = list(model.base.edge_updates.parameters())
    elif mode in EDGE_CONDITIONED_MODES:
        candidate_parameters = list(model.edge_conditioned_keys.parameters())
    elif mode in GPSPP_LOCAL_MODES:
        candidate_parameters = list(model.local_adapters.parameters())
    elif mode in PAIR_TOKEN_MODES:
        candidate_parameters = list(model.relation_token.parameters())
    elif mode in FUNCTIONAL_GROUP_MODES:
        candidate_parameters = list(model.functional_group_token.parameters())
    elif mode in CHEM_TYPED_PAIR_MODES:
        candidate_parameters = list(model.relation_token.parameters())
    elif mode in MULTIPLICATIVE_PAIR_MODES:
        candidate_parameters = list(model.relation_token.parameters())
    elif mode in SPARSE_TRIPLET_MODES:
        candidate_parameters = (
            list(model.triplet_initial.parameters())
            + list(model.triplet_updates.parameters())
            + list(model.triplet_to_edge.parameters())
            + list(model.triplet_to_node.parameters())
        )
    elif mode in SPD_PAIR_TOKEN_MODES:
        candidate_parameters = list(model.relation_token.parameters())
    elif mode in ONESHOT_TRIPLET_MODES:
        candidate_parameters = (
            list(model.triplet_adapter.parameters())
            + list(model.relation_token.parameters())
        )
    elif mode in PAIR_TOKEN_MOSE_MODES:
        candidate_parameters = (
            list(model.mose_residual.parameters())
            + list(model.relation_token.parameters())
        )
    elif mode in MOSE_MODES:
        if mode in MOSE_REPLACEMENT_MODES:
            candidate_parameters = list(model.rwse_encoder.parameters())
        else:
            candidate_parameters = list(model.mose_residual.parameters()) + list(
                model.mose_gate.parameters()
            )
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
    elif mode == "neural_atom_k1_repset_readout":
        candidate_parameters = list(model.repset_readout.parameters())
    elif mode == "neural_atom_k1_tied_selector":
        candidate_parameters = list(model.shared_selector.parameters())
    elif mode == "neural_atom_k1_collapsed_mha":
        candidate_parameters = list(model.collapsed_slot_projections.parameters())
    elif mode in {
        "neural_atom_k1_no_slot_attention",
        "neural_atom_k1_no_attention_uniform_return",
    }:
        candidate_parameters = [
            parameter
            for mixer in model.base.neural_atom_mixers.values()
            for module in (mixer.slot_ffn, mixer.return_projection)
            for parameter in module.parameters()
        ]
    elif mode in {
        "neural_atom_k1_uniform_return",
        "neural_atom_k1_inverse_return",
        "neural_atom_k1_no_attention_uniform_return",
    }:
        candidate_parameters = [
            parameter
            for mixer in model.base.neural_atom_mixers.values()
            for module in (mixer.node_key, mixer.node_value, mixer.return_projection)
            for parameter in module.parameters()
        ]
    candidate_trainable = mode == "neural_atom_k1_v4" or any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0
        for parameter in candidate_parameters
    )
    if not candidate_trainable:
        raise RuntimeError(f"Candidate-only mechanism has no finite gradient: {mode}")
    if mode in MOSE_GATED_DUAL_MODES:
        residual_trainable = any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and float(parameter.grad.abs().sum()) > 0
            for parameter in model.mose_residual.parameters()
        )
        gate_trainable = any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and float(parameter.grad.abs().sum()) > 0
            for parameter in model.mose_gate.parameters()
        )
        mechanism_checks["residual_trainable_after_two_steps"] = residual_trainable
        mechanism_checks["gate_trainable_after_two_steps"] = gate_trainable
        if not residual_trainable or not gate_trainable:
            raise RuntimeError("Selective MoSE residual or gate has no finite gradient")
    if mode in PAIR_TOKEN_MOSE_MODES:
        mose_residual_trainable = any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and float(parameter.grad.abs().sum()) > 0
            for parameter in model.mose_residual.parameters()
        )
        pair_token_trainable = any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and float(parameter.grad.abs().sum()) > 0
            for parameter in model.relation_token.parameters()
        )
        mechanism_checks["mose_residual_trainable_after_two_steps"] = (
            mose_residual_trainable
        )
        mechanism_checks["pair_token_trainable_after_two_steps"] = (
            pair_token_trainable
        )
        if not mose_residual_trainable or not pair_token_trainable:
            raise RuntimeError(
                "PairToken or MoSE residual has no finite gradient after two steps"
            )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if (
        mode in MOSE_MODES
        and parameter_count
        != ARCHITECTURE_CONFIGS[mode]["expected_parameters"]
    ):
        raise RuntimeError("K1-MoSE parameter identity changed")
    peak_reserved_mib = torch.cuda.max_memory_reserved() / 1024**2
    total_memory_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
    memory_reserve_fraction = 1.0 - peak_reserved_mib / total_memory_mib
    if memory_reserve_fraction < 0.15:
        raise RuntimeError("Architecture preflight retained less than 15% memory")
    del baseline, model, optimizer
    torch.cuda.empty_cache()
    return {
        "mode": mode,
        "parameter_count": parameter_count,
        "shared_k1_initial_state_sha256": shared_sha,
        "exact_k1_function_at_initialization": exact_nested_initialization,
        "initialization_policy": (
            "feature-replacement-shared-k1-state"
            if mode in MOSE_REPLACEMENT_MODES
            else (
                "identical-tensors-altered-edge-dataflow"
                if mode in active_edge_modes or mode in EDGE_CONDITIONED_MODES
                else "nested-function"
            )
        ),
        "candidate_mechanism_trainable_after_two_steps": candidate_trainable,
        "preflight_peak_reserved_mib": peak_reserved_mib,
        "preflight_total_memory_mib": total_memory_mib,
        "preflight_memory_reserve_fraction": memory_reserve_fraction,
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
        "feature_fingerprint": (
            MOSE_FEATURE_FINGERPRINT
            if mode in MOSE_REPLACEMENT_MODES
            else (
                MOSE_RWSE_FEATURE_FINGERPRINT
                if mode in MOSE_DUAL_MODES
                else FEATURE_FINGERPRINT
            )
        ),
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
    resume_from: Path | None = None,
) -> dict:
    import torch
    import torch.nn.functional as functional
    from .training_reproducibility import capture_rng_state, restore_rng_state
    from .k1_edge_memory import MODES as EDGE_MEMORY_MODES
    from .k1_edge_slot_interaction import MODES as EDGE_SLOT_MODES
    from .k1_edge_conditioned_slot import MODES as EDGE_CONDITIONED_MODES
    from .k1_gpspp_local import MODES as GPSPP_LOCAL_MODES
    from .k1_pair_token import MODES as PAIR_TOKEN_MODES
    from .k1_functional_group_token import MODES as FUNCTIONAL_GROUP_MODES
    from .k1_chem_typed_pair_token import MODES as CHEM_TYPED_PAIR_MODES
    from .k1_multiplicative_pair_value import MODES as MULTIPLICATIVE_PAIR_MODES
    from .k1_sparse_triplet import MODES as SPARSE_TRIPLET_MODES
    from .k1_spd_pair_token import MODES as SPD_PAIR_TOKEN_MODES
    from .k1_oneshot_triplet_pair_token import MODES as ONESHOT_TRIPLET_MODES
    active_edge_modes = EDGE_MEMORY_MODES + EDGE_SLOT_MODES
    recovery_chunk_modes = (
        active_edge_modes
        + EDGE_CONDITIONED_MODES
        + GPSPP_LOCAL_MODES
        + PAIR_TOKEN_MODES
        + FUNCTIONAL_GROUP_MODES
        + CHEM_TYPED_PAIR_MODES
        + MULTIPLICATIVE_PAIR_MODES
        + SPARSE_TRIPLET_MODES
        + SPD_PAIR_TOKEN_MODES
        + ONESHOT_TRIPLET_MODES
        + MOSE_MODES
    )

    if mode not in ARCHITECTURE_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}")
    if len(source_commit) != 40 or len(source_archive_sha256) != 64:
        raise ValueError("Committed source and archive identities are required")
    validate_screen_arm(physical_batch_per_device=BATCH_SIZE)
    if compute_row_order_fingerprint() != ROW_ORDER_FINGERPRINT:
        raise RuntimeError("Frozen row order implementation changed")
    configure_fp32_determinism(SEED)
    root, manifest = find_fixed_cache()
    roles = load_roles(
        root,
        manifest,
        retain_wedge_topology=mode in SPARSE_TRIPLET_MODES + ONESHOT_TRIPLET_MODES,
    )
    functional_group_manifest = None
    if mode in FUNCTIONAL_GROUP_MODES + CHEM_TYPED_PAIR_MODES:
        from .pcqm_functional_group_sidecar import attach_functional_group_roles

        roles, functional_group_manifest = attach_functional_group_roles(
            roles, manifest
        )
    mose_manifest = None
    if mode in MOSE_MODES:
        from .pcqm_mose import attach_mose_roles

        roles, mose_manifest = attach_mose_roles(
            roles,
            manifest,
            retain_rwse=mode in MOSE_DUAL_MODES,
        )
    output.mkdir(parents=True, exist_ok=True)
    certificate, runtime = build_runtime_certificate(
        roles,
        output,
        calibration_mode=mode,
    )
    target_stats = runtime["target_stats"]
    preflight = _architecture_preflight(mode, roles, target_stats)
    if functional_group_manifest is not None:
        preflight["functional_group_sidecar"] = {
            "aggregate_sha256": functional_group_manifest["aggregate_sha256"],
            "contract_sha256": functional_group_manifest[
                "functional_group_contract_sha256"
            ],
            "derived_only_from_existing_ogb_graph_features": True,
            "gap_labels_read": False,
            "protected_roles_read": False,
        }
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
    start_epoch = 0
    if resume_from is not None:
        import shutil
        checkpoint = torch.load(resume_from / "last_checkpoint.pt", map_location="cpu", weights_only=False)
        for key, expected in {
            "mode": mode, "source_commit": source_commit,
            "source_archive_sha256": source_archive_sha256,
            "runtime_certificate_id": runtime["runtime_certificate_id"],
            "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
            "row_order_fingerprint": ROW_ORDER_FINGERPRINT,
        }.items():
            if checkpoint.get(key) != expected:
                raise RuntimeError(f"Resume identity mismatch: {key}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trace = checkpoint["trace"]
        start_epoch = checkpoint["epoch"] + 1
        if len(trace) != start_epoch or not 0 <= start_epoch <= EPOCHS:
            raise RuntimeError("Resume epoch/trace mismatch")
        best, best_epoch = checkpoint["best"], checkpoint["best_epoch"]
        for name, digest in checkpoint["best_artifact_sha256"].items():
            if sha256_file(resume_from / name) != digest:
                raise RuntimeError(f"Resume selected artifact corrupted: {name}")
            if (resume_from / name).resolve() != (output / name).resolve():
                shutil.copyfile(resume_from / name, output / name)
        # Match the uninterrupted persistent dev loader's already-started workers.
        # Their initial base-seed draw must not advance the restored model RNG.
        iter(development_loader)
        restore_rng_state(checkpoint["rng_state"])
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, EPOCHS):
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
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
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
        if not math.isfinite(validation_mae):
            raise RuntimeError("Nonfinite development MAE")
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
                "rng_state": capture_rng_state(),
                "best": best,
                "best_epoch": best_epoch,
                "best_artifact_sha256": {
                    name: sha256_file(output / name)
                    for name in ("best_model.pt", "best_development_payload.pt")
                },
                "fixed_manifest_sha256": FIXED_MANIFEST_SHA256,
                "row_order_fingerprint": ROW_ORDER_FINGERPRINT,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
                "test_challenge_role_read": False,
            },
        )
        atomic_json(output / "trace.json", {"epochs": trace})
        if mode in recovery_chunk_modes and (epoch + 1) % 10 == 0:
            import tarfile
            chunk = output / f"recovery_epoch_{epoch + 1:02d}.tar"
            temporary = chunk.with_suffix(".tmp")
            with tarfile.open(temporary, "w") as archive:
                for name in ("last_checkpoint.pt", "best_model.pt", "best_development_payload.pt", "trace.json", "preflight.json", "runtime_certificate.json"):
                    archive.add(output / name, arcname=name)
            os.replace(temporary, chunk)
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
    if mose_manifest is not None:
        record["mose_cache"] = {
            "format": mose_manifest["format"],
            "pattern_sha256": mose_manifest["pattern_sha256"],
            "aggregate_sha256": mose_manifest["aggregate_sha256"],
            "feature_dim": mose_manifest["feature_dim"],
            "feature_transform_at_training": mose_manifest[
                "feature_transform_at_training"
            ],
        }
    if functional_group_manifest is not None:
        record["functional_group_sidecar"] = {
            "format": functional_group_manifest["format"],
            "aggregate_sha256": functional_group_manifest["aggregate_sha256"],
            "functional_group_contract_sha256": functional_group_manifest[
                "functional_group_contract_sha256"
            ],
            "fixed_geometry_aggregate_sha256": functional_group_manifest[
                "fixed_geometry_aggregate_sha256"
            ],
            "gap_labels_read": False,
            "protected_roles_read": False,
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
