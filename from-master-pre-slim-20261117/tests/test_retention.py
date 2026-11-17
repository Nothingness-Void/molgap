from __future__ import annotations

import pytest
import torch

from molgap.retention import replay_weight_for_fraction, retention_loss


def test_replay_weight_targets_half_old_draws() -> None:
    weight = replay_weight_for_fraction(500_000, 1_500_000, 0.5)
    assert weight == pytest.approx(3.0)


def test_retention_loss_only_distills_old_prefix() -> None:
    prediction = torch.tensor([[1.0], [2.0], [4.0]], requires_grad=True)
    label = torch.tensor([[0.0], [2.0], [3.0]])
    teacher = torch.tensor([[0.0], [0.0], [100.0]])
    source_idx = torch.tensor([10, 600_000, 700_000])

    result = retention_loss(
        prediction,
        label,
        source_idx,
        teacher,
        boundary=500_000,
        distillation_weight=0.25,
    )

    assert result.retained_rows == 1
    assert result.label.item() == pytest.approx(2.0 / 3.0)
    assert result.distillation.item() == pytest.approx(1.0)
    assert result.total.item() == pytest.approx(2.0 / 3.0 + 0.25)
    result.total.backward()
    assert prediction.grad is not None
