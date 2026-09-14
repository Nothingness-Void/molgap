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
    _gptrans_source_path,
    _learning_rate,
)
from molgap.pcqm_gptrans_full_acceptance import _resolve_preflight_path
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
    expected_path = (ROOT / "src/molgap/gptrans.py").resolve()
    assert _gptrans_source_path() == expected_path
    assert _architecture_source_sha256(_gptrans_source_path()) == GPTRANS_SOURCE_SHA256


def test_recovery_jobs_keep_the_frozen_ogb_archive_hash():
    recovery_jobs = (
        ROOT
        / "experiments/pcqm_k1_gptrans_full_fusion/jobs/recovery_r1"
    )
    expected = (
        "export MOLGAP_OGB_SOURCE_SHA256="
        "a18d4cacc6a35ad24938f52cfe197a255a5f64bb197f8d0f056c204467ec1e33"
    )
    scripts = sorted(recovery_jobs.glob("*.pbs"))
    assert len(scripts) == 8
    assert all(expected in script.read_text(encoding="utf-8") for script in scripts)


def test_gptrans_acceptance_can_use_versioned_preflight_path(tmp_path):
    full = tmp_path / "full"
    preflight = tmp_path / "preflight_r1" / "preflight.json"
    full.mkdir()
    preflight.parent.mkdir()
    preflight.write_text("{}", encoding="utf-8")
    assert _resolve_preflight_path(full, preflight) == preflight.resolve()
