"""Explicit, versioned training-objective parameters for GPTrans addons.

The frozen trainer defaults remain unchanged. This module supplies a loss
adapter, not a replacement training loop, data loader, or submission route.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Mapping

from .screen_policy import canonical_fingerprint


@dataclass(frozen=True)
class GPTransObjectiveConfig:
    auxiliary_hidden_dim: int = 32
    descriptor_weight: float = 0.0
    fingerprint_weight: float = 0.0
    auxiliary_seed: int = 42

    def __post_init__(self):
        if type(self.auxiliary_hidden_dim) is not int or self.auxiliary_hidden_dim <= 0:
            raise ValueError("auxiliary_hidden_dim must be a positive integer")
        if type(self.auxiliary_seed) is not int or not 0 <= self.auxiliary_seed < 2**63:
            raise ValueError("auxiliary_seed must be an integer in [0, 2**63)")
        for name in ("descriptor_weight", "fingerprint_weight"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))

    @property
    def enabled(self) -> bool:
        return self.descriptor_weight > 0 or self.fingerprint_weight > 0

    def to_dict(self) -> dict:
        return {"schema": "molgap-gptrans-objective-v1", **asdict(self)}

    @property
    def identity(self) -> str:
        return canonical_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping) -> "GPTransObjectiveConfig":
        data = dict(data)
        if data.pop("schema", "molgap-gptrans-objective-v1") != "molgap-gptrans-objective-v1":
            raise ValueError("unsupported objective schema")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown objective parameters: {sorted(unknown)}")
        return cls(**data)


def validate_objective_checkpoint(checkpoint: Mapping, config: GPTransObjectiveConfig) -> None:
    """Old checkpoints are baseline-only; never infer a missing candidate config."""
    stored = checkpoint.get("training_objective")
    if stored is None:
        if config != GPTransObjectiveConfig():
            raise RuntimeError("checkpoint has no explicit training objective")
        if any(name.startswith("_chemical_aux_head.") for name in checkpoint.get("model", {})):
            raise RuntimeError("auxiliary checkpoint is missing its objective identity")
    elif stored != config.to_dict() or checkpoint.get("training_objective_sha256") != config.identity:
        raise RuntimeError("checkpoint training objective changed")


def combine_losses(prediction, target, config, auxiliary_prediction=None,
                   descriptor_target=None, fingerprint_target=None,
                   descriptor_valid_mask=None):
    """Return a scalar optimized loss and detached, separately named metrics."""
    import torch
    import torch.nn.functional as F

    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError("Gap prediction/target must be aligned one-dimensional tensors")
    gap = F.l1_loss(prediction, target)
    total = gap
    metrics = {"gap_l1_normalized": gap.detach()}
    if config.enabled:
        if auxiliary_prediction is None or auxiliary_prediction.shape != (prediction.shape[0], 712):
            raise ValueError("auxiliary prediction must have shape (batch, 712)")
        for name, weight, expected, actual in (
            ("descriptor_mse", config.descriptor_weight, descriptor_target, auxiliary_prediction[:, :200]),
            ("fingerprint_bce", config.fingerprint_weight, fingerprint_target, auxiliary_prediction[:, 200:]),
        ):
            if weight == 0:
                continue
            if expected is None or expected.shape != actual.shape:
                raise ValueError(f"{name} target shape mismatch")
            if expected.device != actual.device:
                raise ValueError(f"{name} targets must be finite and on the prediction device")
            mask = None
            if name == "descriptor_mse" and descriptor_valid_mask is not None:
                mask = descriptor_valid_mask
                if mask.dtype != torch.bool or mask.shape != actual.shape or mask.device != actual.device:
                    raise ValueError("Descriptor mask must be aligned boolean targets on the same device")
            if not bool(torch.isfinite(expected if mask is None else expected[mask]).all()):
                raise ValueError(f"{name} targets must be finite and on the prediction device")
            if name == "fingerprint_bce" and not bool(((expected == 0) | (expected == 1)).all()):
                raise ValueError("fingerprint targets must be binary")
            if mask is not None:
                value = (F.mse_loss(actual[mask], expected[mask].float()) if bool(mask.any())
                         else actual.sum() * 0)
            else:
                value = F.mse_loss(actual, expected.float()) if name == "descriptor_mse" else F.binary_cross_entropy_with_logits(actual, expected.float())
            metrics[name] = value.detach()
            total = total + weight * value
    metrics["total_loss"] = total.detach()
    return total, metrics


class GPTransObjective:
    """Attach before optimizer/EMA creation; capture graph state only for loss.

    The owner joins labels to each training batch by source_index first and
    supplies chemical_descriptors / chemical_fingerprint tensors. This class
    never guesses row identity or reads a dataset.
    """

    def __init__(self, model, config: GPTransObjectiveConfig):
        import torch
        from torch import nn

        self.model, self.config = model, config
        if hasattr(model, "_chemical_aux_head"):
            raise ValueError("model already has an auxiliary head")
        if config.enabled:
            input_dim = model.readout[0].in_features
            parameter = next(model.parameters())
            # CPU-only RNG fork avoids changing backbone dropout or CUDA RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config.auxiliary_seed)
                head = nn.Sequential(nn.Linear(input_dim, config.auxiliary_hidden_dim),
                                     nn.GELU(), nn.Linear(config.auxiliary_hidden_dim, 712))
            model.add_module("_chemical_aux_head", head.to(device=parameter.device, dtype=parameter.dtype))

    def loss(self, model, batch, mean, std):
        from .pcqm_gptrans_v4 import _forward

        if model is not self.model:
            raise ValueError("objective belongs to a different model")
        captured = []
        handle = None
        if self.config.enabled:
            handle = model.readout.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
        try:
            prediction = _forward(model, batch)
        finally:
            if handle is not None:
                handle.remove()
        auxiliary = None
        if self.config.enabled:
            if len(captured) != 1:
                raise RuntimeError("expected one GPTrans readout invocation")
            auxiliary = model._chemical_aux_head(captured[0])
        return combine_losses(prediction, (batch.y.view(-1).float() - mean) / std,
                              self.config, auxiliary,
                              getattr(batch, "chemical_descriptors", None),
                              getattr(batch, "chemical_fingerprint", None),
                              getattr(batch, "chemical_descriptor_valid_mask", None))

    def checkpoint_metadata(self) -> dict:
        return {"training_objective": self.config.to_dict(),
                "training_objective_sha256": self.config.identity}

    def scientific_fields(self, base: Mapping) -> dict:
        return {**base, "loss_fingerprint": ("normalized-gap-l1+chemical-aux:" + self.config.identity
                                             if self.config.enabled else base["loss_fingerprint"])}


def export_gap_state_dict(state: Mapping) -> dict:
    """Load this into a fresh unmodified GPTrans model with strict=True."""
    return {name: value for name, value in state.items() if not name.startswith("_chemical_aux_head.")}
