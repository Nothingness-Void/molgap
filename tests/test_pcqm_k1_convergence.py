from __future__ import annotations

import pytest
from pathlib import Path

from molgap.pcqm_k1_convergence import (
    BASE_CHECKPOINT_SHA256,
    BASE_CONTRACT_SHA256,
    BASE_MODEL_BUNDLE_SHA256,
    BASE_VALID_ACCEPTANCE_SHA256,
    BASE_VALID_MAE_EV,
    CONTRACT,
    MAX_ADDITIONAL_STEPS,
    MIN_LR,
    START_LR,
    continuation_learning_rate,
)


def test_continuation_contract_is_bounded_and_consumes_validation():
    assert CONTRACT["source_official_valid_mae_eV"] == BASE_VALID_MAE_EV
    assert CONTRACT["max_additional_passes"] == 6
    assert CONTRACT["patience_evaluations"] == 3
    assert CONTRACT["selection_role"] == "official-valid-consumed-for-convergence-selection"
    assert CONTRACT["test_dev_role"] == "sealed"
    assert CONTRACT["source_training_contract_sha256"] == BASE_CONTRACT_SHA256
    assert CONTRACT["source_checkpoint_sha256"] == BASE_CHECKPOINT_SHA256
    assert CONTRACT["source_model_bundle_sha256"] == BASE_MODEL_BUNDLE_SHA256
    assert CONTRACT["source_official_valid_acceptance_sha256"] == BASE_VALID_ACCEPTANCE_SHA256
    assert CONTRACT["architecture_source"] == "frozen Neural-Atom K1 source set"


def test_continuation_learning_rate_is_monotonic_and_bounded():
    values = [continuation_learning_rate(step) for step in (1, MAX_ADDITIONAL_STEPS // 2, MAX_ADDITIONAL_STEPS)]
    assert START_LR > values[0] > values[1] > values[2]
    assert values[-1] == pytest.approx(MIN_LR)
    with pytest.raises(ValueError):
        continuation_learning_rate(0)


def test_all_pbs_jobs_mount_the_frozen_ogb_archive():
    root = Path("experiments/pcqm_k1_full_convergence/jobs")
    for path in sorted(root.glob("*.pbs")):
        text = path.read_text(encoding="utf-8")
        assert "OGB_ARCHIVE=/lustre/home/users/sm2/chou/molgap-k1-gpttrans-full/inputs/ogb-1.3.6.zip" in text
        assert "$OGB_ARCHIVE:$ROOT/code/src" in text
        assert "MOLGAP_OGB_SOURCE_SHA256=a18d4cacc6a35ad24938f52cfe197a255a5f64bb197f8d0f056c204467ec1e33" in text
        assert "k1_convergence_20260915_r1" in text
        assert "pcqm_gptrans_full_convergence" not in text

    preflight = (root / "preflight.pbs").read_text(encoding="utf-8")
    train = (root / "train.pbs").read_text(encoding="utf-8")
    for text in (preflight, train):
        assert 'source-root "$BASE/outputs/k1/full"' in text
        assert 'source-evaluation "$BASE/outputs/fusion_study/metrics.json"' in text
