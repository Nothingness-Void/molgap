from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from molgap.pcqm_k1_gptrans_fusion import (
    calibration_mask,
    fit_convex_blend_weight,
    mae,
    _validate_fusion_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fusion_contract_freezes_official_validation_partition_and_roles():
    contract_path = ROOT / "experiments/pcqm_k1_gptrans_full_fusion/fusion_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert contract["format"] == "molgap-pcqm-k1-gptrans-fusion-contract-v1"
    assert contract["evaluation_role"]["rows"] == 73_545
    assert contract["evaluation_role"]["test_dev_read"] is False
    assert contract["evaluation_role"]["test_challenge_read"] is False
    assert contract["calibration"]["calibration_predicate"] == "source_idx % 5 == 0"
    assert contract["calibration"]["holdout_predicate"] == "source_idx % 5 != 0"
    assert contract["calibration"]["fitted_parameters"] == 1
    assert contract["resources"]["accelerator"] == "exactly one A100"
    accepted = _validate_fusion_contract(
        {"source_manifest_sha256": contract["evaluation_role"]["official_row_manifest_sha256"]},
        contract["evaluation_role"]["graph_acceptance_sha256"],
    )
    assert accepted == contract


def test_fusion_calibration_mask_is_source_aligned_and_deterministic():
    source_idx = np.arange(20, dtype=np.int64)
    expected = source_idx % 5 == 0

    assert np.array_equal(calibration_mask(source_idx), expected)
    assert calibration_mask(source_idx).sum() == 4


def test_fusion_weight_search_recovers_exact_convex_mix():
    target = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    k1 = target + 0.1
    gp = target - 0.3

    alpha, loss = fit_convex_blend_weight(k1, gp, target)

    assert alpha == 0.75
    assert loss == pytest.approx(0.0, abs=1e-12)


def test_fusion_weight_search_ties_prefer_equal_blend():
    prediction = np.array([-1.0, 0.0, 1.0])

    alpha, loss = fit_convex_blend_weight(prediction, prediction, np.zeros(3))

    assert alpha == 0.5
    assert loss == pytest.approx(2 / 3)


def test_fusion_metrics_reject_misaligned_or_nonfinite_inputs():
    with pytest.raises(ValueError):
        fit_convex_blend_weight(np.array([0.0]), np.array([0.0, 1.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        fit_convex_blend_weight(np.array([np.nan]), np.array([0.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        calibration_mask(np.array([1, 1], dtype=np.int64))
    with pytest.raises(ValueError):
        mae(np.array([0.0]), np.array([np.inf]))
