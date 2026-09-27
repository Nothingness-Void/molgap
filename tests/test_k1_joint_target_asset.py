"""Synthetic label identity checks only; no model construction or inference."""
import hashlib

import pytest
import torch

from molgap.k1_joint_objective import verify_frozen_train_targets


def test_reduction_roundoff_never_changes_applied_frozen_constants():
    values = torch.arange(100000, dtype=torch.float32) / 10000
    asset = {"target_sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
             "mean": float(values.mean()) + 4.76837158203125e-7,
             "std": float(values.std())}
    result = verify_frozen_train_targets(values, torch.arange(100000), asset)
    assert result["computed_train_mean_eV"] != result["applied_mean_eV"]
    assert result["applied_mean_eV"] == asset["mean"]
    assert result["train_target_sha256"] == asset["target_sha256"]
    with pytest.raises(RuntimeError, match="bytes changed"):
        verify_frozen_train_targets(values + 1e-3, torch.arange(100000), asset)
    with pytest.raises(RuntimeError, match="membership changed"):
        verify_frozen_train_targets(values, torch.arange(100000).flip(0), asset)
