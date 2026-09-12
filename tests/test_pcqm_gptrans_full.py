from __future__ import annotations

import json
from pathlib import Path

from molgap.pcqm_gptrans_full_runner import (
    EXPECTED_INITIAL_SHA256,
    EXPECTED_PARAMETERS,
    SAMPLE_PRESENTATIONS,
    TOTAL_STEPS,
    TRAINING_CONTRACT,
    TRAINING_CONTRACT_SHA256,
    WARMUP_STEPS,
    GPTRANS_SOURCE_SHA256,
    _learning_rate,
)
from molgap.pcqm_k1_full_runner import _architecture_source_sha256
from molgap.training_reproducibility import canonical_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def test_gptrans_full_contract_matches_frozen_json():
    frozen = json.loads(
        (ROOT / "experiments/pcqm_k1_gptrans_full_fusion/training_contract.json")
        .read_text(encoding="utf-8")
    )
    assert frozen == TRAINING_CONTRACT
    assert canonical_fingerprint(frozen) == TRAINING_CONTRACT_SHA256
    assert frozen["architecture"]["seed42_initial_model_sha256"] == EXPECTED_INITIAL_SHA256
    assert frozen["architecture"]["parameter_count"] == EXPECTED_PARAMETERS


def test_gptrans_full_schedule_uses_frozen_sample_exposure():
    assert SAMPLE_PRESENTATIONS == 20_000_000
    assert TOTAL_STEPS == 156_250
    assert TOTAL_STEPS * 128 == SAMPLE_PRESENTATIONS
    assert _learning_rate(1) == 2.5e-4
    assert _learning_rate(WARMUP_STEPS) == 1.0e-3
    assert _learning_rate(TOTAL_STEPS) == 1.0e-6
    assert 0 < WARMUP_STEPS < TOTAL_STEPS


def test_gptrans_architecture_source_hash_is_frozen():
    assert _architecture_source_sha256(ROOT / "src/molgap/gptrans.py") == GPTRANS_SOURCE_SHA256
