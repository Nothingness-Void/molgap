"""Retention-aware objectives for controlled dataset scale-up."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RetentionLoss:
    total: torch.Tensor
    label: torch.Tensor
    distillation: torch.Tensor
    retained_rows: int


def replay_weight_for_fraction(
    old_rows: int,
    new_rows: int,
    target_old_fraction: float,
) -> float:
    """Return the old-row sampling weight required for a target draw fraction."""
    if old_rows <= 0 or new_rows <= 0:
        raise ValueError("old_rows and new_rows must both be positive")
    if not 0.0 < target_old_fraction < 1.0:
        raise ValueError("target_old_fraction must be in (0, 1)")
    return (
        target_old_fraction
        * float(new_rows)
        / ((1.0 - target_old_fraction) * float(old_rows))
    )


def retention_loss(
    prediction: torch.Tensor,
    label: torch.Tensor,
    source_idx: torch.Tensor,
    teacher_prediction: torch.Tensor,
    *,
    boundary: int,
    distillation_weight: float,
) -> RetentionLoss:
    """Add teacher retention only for rows in the preserved source prefix."""
    if prediction.shape != label.shape or prediction.shape != teacher_prediction.shape:
        raise ValueError("prediction, label, and teacher_prediction shapes must match")
    if source_idx.ndim != 1 or source_idx.shape[0] != prediction.shape[0]:
        raise ValueError("source_idx must contain one index per prediction row")
    if distillation_weight < 0.0:
        raise ValueError("distillation_weight must be non-negative")

    label_value = F.l1_loss(prediction, label)
    retained = source_idx < boundary
    retained_rows = int(retained.sum().item())
    if retained_rows:
        distillation_value = F.l1_loss(
            prediction[retained],
            teacher_prediction[retained],
        )
    else:
        distillation_value = prediction.sum() * 0.0
    total = label_value + distillation_weight * distillation_value
    return RetentionLoss(
        total=total,
        label=label_value,
        distillation=distillation_value,
        retained_rows=retained_rows,
    )
