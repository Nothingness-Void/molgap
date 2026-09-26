"""Reusable EdgeState train/eval and resume primitives, without a science recipe."""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .edge_state_adapter import build_edge_state_model, edge_state_metadata
from .experiment_spec import ExperimentSpec
from .screen_policy import canonical_fingerprint
from .training_reproducibility import (
    assert_finite_state_dict, atomic_torch_save, capture_rng_state,
    restore_rng_state, sha256_file,
)
from .v4_runtime import model_state_sha256, torch_load_compat


CHECKPOINT_FORMAT = "molgap-edge-state-training-checkpoint-v1"
SAMPLER_FORMAT = "molgap-edge-state-epoch-randperm-v1"


def _digest(value: str, label: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _positive_number(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _nonnegative_integer(value, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")


def _finite_tensors(value, label: str) -> None:
    import torch

    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{label} contains a non-finite tensor")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tensors(item, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite_tensors(item, f"{label}[{index}]")


def _require_fp32(model, batch) -> None:
    import torch

    if batch.y.dtype != torch.float32 or batch.random_walk_pe.dtype != torch.float32:
        raise ValueError("EdgeState targets and RWSE must be FP32")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("EdgeState training core requires FP32 model parameters")


def training_target_statistics(chunks, *, expected_rows: int, minimum_std: float) -> tuple[float, float]:
    """Compute train-only mean and sample std from caller-authenticated targets."""
    import torch

    if type(expected_rows) is not int or expected_rows < 2:
        raise ValueError("expected_rows must be at least two")
    lower = _positive_number(minimum_std, "minimum_std")
    values = []
    for chunk in chunks:
        if (not torch.is_tensor(chunk) or not chunk.is_floating_point()
                or chunk.ndim not in (1, 2) or (chunk.ndim == 2 and chunk.shape[1] != 1)
                or not bool(torch.isfinite(chunk).all())):
            raise ValueError("Training targets must be finite floating-point vectors")
        values.append(chunk.reshape(-1).double().cpu())
    if not values or sum(value.numel() for value in values) != expected_rows:
        raise ValueError("Training target row count does not match the approved train role")
    joined = torch.cat(values)
    return float(joined.mean()), float(joined.std(unbiased=True).clamp_min(lower))


class EpochPermutationBatchSampler:
    """Stateless seed+epoch global randperm with full-batch resume coordinates."""

    def __init__(self, dataset_size: int, batch_size: int, *, seed: int,
                 epoch: int, start_batch: int = 0) -> None:
        import torch

        for name, value in (("dataset_size", dataset_size), ("batch_size", batch_size),
                            ("seed", seed), ("epoch", epoch), ("start_batch", start_batch)):
            _nonnegative_integer(value, name)
        if batch_size == 0 or dataset_size < batch_size:
            raise ValueError("Sampler requires at least one complete physical batch")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.full_batches = dataset_size // batch_size
        if start_batch > self.full_batches:
            raise ValueError("Sampler resume cursor exceeds the full-batch count")
        self.start_batch = start_batch
        generator = torch.Generator().manual_seed(seed + epoch)
        self.indices = torch.randperm(dataset_size, generator=generator)[:self.full_batches * batch_size]
        self.order_sha256 = hashlib.sha256(
            self.indices.numpy().astype("<i8", copy=False).tobytes()
        ).hexdigest()

    def __iter__(self):
        for batch_index in range(self.start_batch, self.full_batches):
            left = batch_index * self.batch_size
            yield self.indices[left:left + self.batch_size].tolist()

    def __len__(self) -> int:
        return self.full_batches - self.start_batch

    def state_for(self, next_batch: int) -> dict:
        """Use the processed-batch cursor, never DataLoader's prefetched position."""
        _nonnegative_integer(next_batch, "next_batch")
        if next_batch > self.full_batches:
            raise ValueError("Sampler resume cursor exceeds the full-batch count")
        return {
            "format": SAMPLER_FORMAT, "dataset_size": self.dataset_size,
            "batch_size": self.batch_size, "seed": self.seed, "epoch": self.epoch,
            "next_batch": next_batch, "order_sha256": self.order_sha256,
        }

    @classmethod
    def from_state(cls, state: Mapping, *, dataset_size: int, batch_size: int,
                   seed: int, epoch: int) -> "EpochPermutationBatchSampler":
        if not isinstance(state, Mapping) or set(state) != {
            "format", "dataset_size", "batch_size", "seed", "epoch", "next_batch", "order_sha256"
        } or state["format"] != SAMPLER_FORMAT:
            raise ValueError("Invalid EdgeState sampler state format")
        if (state["dataset_size"] != dataset_size or state["batch_size"] != batch_size
                or state["seed"] != seed or state["epoch"] != epoch):
            raise ValueError("EdgeState sampler contract differs on resume")
        _digest(state["order_sha256"], "order_sha256")
        sampler = cls(dataset_size, batch_size, seed=seed, epoch=epoch,
                      start_batch=state["next_batch"])
        if sampler.order_sha256 != state["order_sha256"]:
            raise ValueError("EdgeState sampler row order differs on resume")
        return sampler


def _verify_sampler_state(state: Mapping, binding: "EdgeStateTrainingBinding",
                          cursor: Mapping) -> None:
    if not isinstance(state, Mapping):
        raise ValueError("Missing EdgeState sampler state")
    EpochPermutationBatchSampler.from_state(
        state, dataset_size=binding.train_rows,
        batch_size=binding.physical_batch_size, seed=binding.initialization_seed,
        epoch=cursor["epoch"],
    )
    if state["next_batch"] != cursor["batch_offset"]:
        raise ValueError("Sampler next batch differs from checkpoint cursor")
    batches_per_epoch = binding.train_rows // binding.physical_batch_size
    if cursor["optimizer_steps"] != cursor["epoch"] * batches_per_epoch + cursor["batch_offset"]:
        raise ValueError("Optimizer steps differ from the sampler cursor")
    if cursor["sample_presentations"] != cursor["optimizer_steps"] * binding.physical_batch_size:
        raise ValueError("Sample presentations differ from the optimizer step count")


@dataclass(frozen=True)
class EdgeStateTrainingBinding:
    spec_identity: str
    arm_id: str
    arm_identity: str
    variant: str
    num_layers: int
    initialization_seed: int
    initialization_sha256: str
    source_commit: str
    source_package_sha256: str
    graph_manifest_sha256: str
    training_contract_sha256: str
    runtime_fingerprint: str
    train_rows: int
    physical_batch_size: int
    target_mean_eV: float
    target_std_eV: float

    @classmethod
    def from_spec(
        cls, spec: ExperimentSpec, arm_id: str, *, source_commit: str,
        source_package_sha256: str, graph_manifest_sha256: str,
        training_contract_sha256: str, runtime_fingerprint: str,
        train_rows: int, physical_batch_size: int,
        target_mean_eV: float, target_std_eV: float,
    ) -> "EdgeStateTrainingBinding":
        """Bind declarations to caller-supplied, independently verified artifacts."""
        metadata = edge_state_metadata(spec, arm_id)
        arm = next(item for item in spec.to_dict()["arms"] if item["arm_id"] == arm_id)
        if arm["initialization"]["kind"] != "random":
            raise ValueError("EdgeState training core requires random initialization")
        if type(source_commit) is not str or re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})", source_commit
        ) is None:
            raise ValueError("source_commit must be a full lowercase commit ID")
        for label, value in (
            ("source_package_sha256", source_package_sha256),
            ("graph_manifest_sha256", graph_manifest_sha256),
            ("training_contract_sha256", training_contract_sha256),
            ("runtime_fingerprint", runtime_fingerprint),
        ):
            _digest(value, label)
        if type(target_mean_eV) not in (int, float) or not math.isfinite(target_mean_eV):
            raise ValueError("target_mean_eV must be finite")
        std = _positive_number(target_std_eV, "target_std_eV")
        for name, value in (("train_rows", train_rows),
                            ("physical_batch_size", physical_batch_size)):
            _nonnegative_integer(value, name)
        if physical_batch_size == 0 or train_rows < physical_batch_size:
            raise ValueError("Training requires at least one complete physical batch")
        return cls(
            spec_identity=metadata.spec_identity, arm_id=arm_id,
            arm_identity=canonical_fingerprint(arm), variant=metadata.variant,
            num_layers=metadata.num_layers, initialization_seed=arm["initialization"]["seed"],
            initialization_sha256=arm["initialization"]["state_sha256"],
            source_commit=source_commit,
            source_package_sha256=source_package_sha256,
            graph_manifest_sha256=graph_manifest_sha256,
            training_contract_sha256=training_contract_sha256,
            runtime_fingerprint=runtime_fingerprint,
            train_rows=train_rows, physical_batch_size=physical_batch_size,
            target_mean_eV=float(target_mean_eV), target_std_eV=std,
        )


def construct_bound_model(spec: ExperimentSpec, binding: EdgeStateTrainingBinding):
    """Construct after caller seeding; reject a mismatched initial model state."""
    if type(binding) is not EdgeStateTrainingBinding:
        raise TypeError("Expected EdgeStateTrainingBinding")
    metadata = edge_state_metadata(spec, binding.arm_id)
    arm = next(item for item in spec.to_dict()["arms"] if item["arm_id"] == binding.arm_id)
    if (metadata.spec_identity != binding.spec_identity
            or metadata.arm_identity != binding.arm_identity
            or metadata.variant != binding.variant
            or metadata.num_layers != binding.num_layers
            or arm["initialization"]["seed"] != binding.initialization_seed
            or arm["initialization"]["state_sha256"] != binding.initialization_sha256):
        raise ValueError("Training binding does not match ExperimentSpec")
    model = build_edge_state_model(spec, binding.arm_id)
    observed = model_state_sha256(model)
    if observed != binding.initialization_sha256:
        raise RuntimeError("Initial EdgeState model state differs from the frozen Spec")
    return model


def validate_ogb_gap_batch(batch) -> int:
    """Validate one accepted train/development batch's model-facing contract."""
    import torch

    from .ogb_features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS

    fields = ("x", "edge_index", "edge_attr", "batch", "random_walk_pe", "y", "source_idx")
    if any(not torch.is_tensor(getattr(batch, name, None)) for name in fields):
        raise ValueError("EdgeState batch is missing required tensor fields")
    x, edges, bonds, groups = batch.x, batch.edge_index, batch.edge_attr, batch.batch
    pe, target, source = batch.random_walk_pe, batch.y.reshape(-1), batch.source_idx.reshape(-1)
    integers = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if (x.ndim != 2 or x.shape[1] != len(ATOM_FEATURE_DIMS) or x.shape[0] == 0
            or x.dtype not in integers):
        raise ValueError("Expected nonempty OGB atom9 categorical features")
    if (edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype not in integers
            or bonds.ndim != 2 or bonds.shape != (edges.shape[1], len(BOND_FEATURE_DIMS))
            or bonds.dtype not in integers):
        raise ValueError("Expected aligned OGB bond3 features and edge_index")
    if (groups.ndim != 1 or groups.numel() != x.shape[0] or groups.dtype not in integers
            or groups.numel() == 0):
        raise ValueError("Invalid graph batch vector")
    if len({value.device for value in (x, edges, bonds, groups, pe, target, source)}) != 1:
        raise ValueError("Batch tensors must be on one device")
    if bool(((groups < 0) | (groups >= x.shape[0])).any()):
        raise ValueError("Graph IDs must be contiguous from zero")
    count = int(groups.max()) + 1
    if count < 1 or not torch.equal(torch.unique(groups), torch.arange(count, device=groups.device)):
        raise ValueError("Graph IDs must be contiguous from zero")
    if (pe.shape != (x.shape[0], 16) or not pe.is_floating_point()
            or not bool(torch.isfinite(pe).all())):
        raise ValueError("Expected finite RWSE16 per atom")
    if (batch.y.shape not in ((count,), (count, 1)) or not target.is_floating_point()
            or not bool(torch.isfinite(target).all())):
        raise ValueError("Expected one finite Gap target in eV per graph")
    if (batch.source_idx.shape not in ((count,), (count, 1)) or source.dtype not in integers
            or bool((source < 0).any()) or torch.unique(source).numel() != count):
        raise ValueError("Expected unique nonnegative source_idx per graph")
    if edges.numel() and bool(((edges < 0) | (edges >= x.shape[0])).any()):
        raise ValueError("edge_index refers to a nonexistent atom")
    if edges.numel() and bool((groups[edges[0].long()] != groups[edges[1].long()]).any()):
        raise ValueError("edge_index crosses graph boundaries")
    if bool(((x < 0) | (x >= x.new_tensor(ATOM_FEATURE_DIMS))).any()):
        raise ValueError("OGB atom feature outside its categorical range")
    if bool(((bonds < 0) | (bonds >= bonds.new_tensor(BOND_FEATURE_DIMS))).any()):
        raise ValueError("OGB bond feature outside its categorical range")
    return count


def _predict(model, batch, rows: int):
    import torch

    result = model(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.random_walk_pe,
    )
    if (result.shape != (rows, 1) or result.dtype != torch.float32
            or not bool(torch.isfinite(result).all())):
        raise RuntimeError("EdgeState output is nonfinite or has the wrong shape")
    return result[:, 0]


def normalized_gap_step(model, optimizer, batch, binding: EdgeStateTrainingBinding, *, clip_norm: float) -> dict:
    """One FP32 normalized-L1 optimizer step; caller owns sampler and schedule."""
    import torch
    import torch.nn.functional as functional

    if type(binding) is not EdgeStateTrainingBinding:
        raise TypeError("Expected EdgeStateTrainingBinding")
    limit = _positive_number(clip_norm, "clip_norm")
    count = validate_ogb_gap_batch(batch)
    if count != binding.physical_batch_size:
        raise ValueError("Training batch differs from the bound physical batch size")
    _require_fp32(model, batch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=batch.x.device.type, enabled=False):
        prediction = _predict(model, batch, count)
        target = (batch.y.reshape(-1) - binding.target_mean_eV) / binding.target_std_eV
        loss = functional.l1_loss(prediction, target)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Normalized Gap loss is nonfinite")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), limit, error_if_nonfinite=True,
    )
    optimizer.step()
    return {"rows": count, "normalized_l1": float(loss.detach()),
            "gradient_norm": float(grad_norm.detach())}


def evaluate_development(model, batches, binding: EdgeStateTrainingBinding, *, expected_source_idx) -> dict:
    """Return source-aligned eV predictions for an explicitly approved dev role."""
    import torch

    if type(binding) is not EdgeStateTrainingBinding:
        raise TypeError("Expected EdgeStateTrainingBinding")
    expected = torch.as_tensor(expected_source_idx).reshape(-1).cpu()
    if (expected.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            or expected.numel() == 0 or bool((expected < 0).any())
            or torch.unique(expected).numel() != expected.numel()):
        raise ValueError("Expected distinct nonnegative integer development source indices")
    expected = expected.long()
    was_training = model.training
    model.eval()
    results = {"source_idx": [], "target_eV": [], "prediction_eV": []}
    try:
        with torch.no_grad():
            for batch in batches:
                count = validate_ogb_gap_batch(batch)
                _require_fp32(model, batch)
                prediction = _predict(model, batch, count)
                prediction_eV = prediction * binding.target_std_eV + binding.target_mean_eV
                if not bool(torch.isfinite(prediction_eV).all()):
                    raise RuntimeError("Denormalized EdgeState predictions are nonfinite")
                results["source_idx"].append(batch.source_idx.reshape(-1).long().cpu())
                results["target_eV"].append(batch.y.reshape(-1).cpu())
                results["prediction_eV"].append(prediction_eV.cpu())
    finally:
        model.train(was_training)
    if not results["source_idx"]:
        raise ValueError("Development loader produced no batches")
    joined = {key: torch.cat(values) for key, values in results.items()}
    if not torch.equal(torch.sort(joined["source_idx"]).values, torch.sort(expected).values):
        raise ValueError("Development source_idx does not match the approved role")
    positions = {int(source): index for index, source in enumerate(joined["source_idx"])}
    order = torch.tensor([positions[int(source)] for source in expected], dtype=torch.long)
    joined = {key: value[order] for key, value in joined.items()}
    joined["mae_eV"] = float((joined["prediction_eV"] - joined["target_eV"]).abs().mean())
    return joined


def save_checkpoint(
    path: Path, binding: EdgeStateTrainingBinding, model, optimizer, scheduler, *,
    epoch: int, batch_offset: int, optimizer_steps: int, sample_presentations: int,
    sampler_state: Mapping, loader_generator=None,
) -> dict:
    """Persist model/optimizer/RNG and caller-owned sampler state with a SHA."""
    if type(binding) is not EdgeStateTrainingBinding:
        raise TypeError("Expected EdgeStateTrainingBinding")
    for name, value in (("epoch", epoch), ("batch_offset", batch_offset),
                        ("optimizer_steps", optimizer_steps),
                        ("sample_presentations", sample_presentations)):
        _nonnegative_integer(value, name)
    cursor = {"epoch": epoch, "batch_offset": batch_offset,
              "optimizer_steps": optimizer_steps,
              "sample_presentations": sample_presentations}
    _verify_sampler_state(sampler_state, binding, cursor)
    assert_finite_state_dict(model.state_dict(), label="EdgeState model")
    optimizer_state = optimizer.state_dict()
    scheduler_state = None if scheduler is None else scheduler.state_dict()
    _finite_tensors(optimizer_state, "EdgeState optimizer")
    _finite_tensors(scheduler_state, "EdgeState scheduler")
    _finite_tensors(sampler_state, "EdgeState sampler")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "binding": asdict(binding),
        "cursor": cursor,
        "model": model.state_dict(),
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "sampler": dict(sampler_state),
        "rng": capture_rng_state(loader_generator=loader_generator),
    }
    path = Path(path)
    atomic_torch_save(path, payload)
    return {"sha256": sha256_file(path), "cursor": payload["cursor"].copy()}


def restore_checkpoint(
    path: Path, expected_sha256: str, binding: EdgeStateTrainingBinding,
    model, optimizer, scheduler, *, loader_generator=None,
) -> dict:
    """Reject changed bytes or contract before loading a trusted checkpoint."""
    if type(binding) is not EdgeStateTrainingBinding:
        raise TypeError("Expected EdgeStateTrainingBinding")
    _digest(expected_sha256, "expected_sha256")
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("EdgeState checkpoint SHA-256 mismatch")
    payload = torch_load_compat(path, map_location="cpu", weights_only=False)
    if (type(payload) is not dict
            or set(payload) != {"format", "binding", "cursor", "model", "optimizer", "scheduler", "sampler", "rng"}
            or payload["format"] != CHECKPOINT_FORMAT
            or payload["binding"] != asdict(binding)):
        raise ValueError("EdgeState checkpoint identity or format mismatch")
    cursor = payload["cursor"]
    if type(cursor) is not dict or set(cursor) != {
        "epoch", "batch_offset", "optimizer_steps", "sample_presentations"
    }:
        raise ValueError("Invalid EdgeState resume cursor")
    for name, value in cursor.items():
        _nonnegative_integer(value, name)
    _verify_sampler_state(payload["sampler"], binding, cursor)
    if (scheduler is None) != (payload["scheduler"] is None):
        raise ValueError("EdgeState scheduler presence changed on resume")
    if not isinstance(payload["model"], Mapping):
        raise ValueError("Invalid EdgeState checkpoint model state")
    if (not isinstance(payload["optimizer"], Mapping)
            or (scheduler is not None and not isinstance(payload["scheduler"], Mapping))
            or not isinstance(payload["rng"], Mapping)
            or (loader_generator is None) == ("loader_generator" in payload["rng"])):
        raise ValueError("EdgeState checkpoint resume state is incomplete")
    assert_finite_state_dict(payload["model"], label="EdgeState checkpoint model")
    _finite_tensors(payload["optimizer"], "EdgeState checkpoint optimizer")
    _finite_tensors(payload["scheduler"], "EdgeState checkpoint scheduler")
    _finite_tensors(payload["sampler"], "EdgeState checkpoint sampler")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"], loader_generator=loader_generator)
    return {"cursor": cursor.copy(), "sampler_state": dict(payload["sampler"])}
