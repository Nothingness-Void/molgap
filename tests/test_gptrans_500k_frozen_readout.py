"""Small deterministic checks for the local 500K diagnostic helpers."""
from __future__ import annotations

import numpy as np
import torch

from experiments.pcqm_gptrans_500k_frozen_readout.analyze_predictions import _bins
from experiments.pcqm_gptrans_500k_frozen_readout.probe_readout import (
    _new_head,
    _paired_gain,
)


def test_paired_gain_sign_and_row_interval() -> None:
    target = torch.tensor([0.0, 0.0, 0.0, 0.0])
    reference = torch.tensor([1.0, 1.0, 1.0, 1.0])
    candidate = torch.tensor([0.5, 0.5, 0.5, 0.5])
    result = _paired_gain(target, reference, candidate)
    assert result["reference_minus_candidate_mae_eV"] == 0.5
    assert result["row_bootstrap_95pct_eV"] == [0.5, 0.5]
    assert result["row_win_fraction"] == 1.0


def test_quintile_bins_cover_all_rows() -> None:
    values = np.arange(100, dtype=np.float64)
    gains = np.ones(100, dtype=np.float64) * 0.002
    bins = _bins(values, gains)
    assert sum(item["rows"] for item in bins) == 100
    assert all(abs(item["mean_gain_eV"] - 0.002) < 1e-12 for item in bins)


def test_residual_head_is_identity_at_initialization() -> None:
    head = _new_head(288, 256)
    assert torch.count_nonzero(head(torch.randn(5, 288))).item() == 0
