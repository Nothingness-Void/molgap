"""CPU-only checks for warm-cost and actual 40-epoch budget projection."""
from __future__ import annotations

import math

import pytest

from molgap.pcqm_metagin_profile import projected_40_epoch_seconds
from molgap.pcqm_metagin_screen import _warmed_step_seconds
from molgap.pcqm_metagin_sidecar import (
    ACCEPTED_AGGREGATE_SHA256, ACCEPTED_MANIFEST_SHA256,
    ACCEPTED_PRODUCER_COMMIT,
)
from molgap.constants import REPO_ROOT
from molgap.training_reproducibility import sha256_file


def test_cold_cuda_step_is_observed_but_not_extrapolated():
    result = _warmed_step_seconds([
        (0.1, "cold", 10.0),
        (0.1, "repeat", 0.4),
        (0.1, "repeat", 0.5),
    ])
    assert result == 0.5
    with pytest.raises(RuntimeError, match="not deterministic"):
        _warmed_step_seconds([(0.1, "cold", 10.0), (0.1, "a", 0.4), (0.1, "b", 0.5)])


def test_projection_includes_training_and_development_forward_work():
    assert projected_40_epoch_seconds(0.4, 0.1) == pytest.approx(
        40 * (781 * 0.4 + 391 * 0.1)
    )
    for value in (0, -1, math.nan, math.inf):
        with pytest.raises(ValueError):
            projected_40_epoch_seconds(value, 0.1)


def test_reused_cpu_sidecar_has_separate_frozen_producer_identity():
    import json

    receipt = json.loads((REPO_ROOT / (
        "experiments/pcqm_metagin_2d_100k/results/cpu_sidecar_acceptance.json"
    )).read_text(encoding="utf-8"))
    assert receipt["source_commit"] == ACCEPTED_PRODUCER_COMMIT
    assert receipt["manifest_sha256"] == ACCEPTED_MANIFEST_SHA256
    assert receipt["aggregate_sha256"] == ACCEPTED_AGGREGATE_SHA256
    root = REPO_ROOT / "platforms/_records/kaggle/training/metagin_2d_sidecar_v1"
    manifest = root / "molgap-metagin-2d-hop-cache-v1/manifest.json"
    if manifest.exists():
        assert sha256_file(manifest) == ACCEPTED_MANIFEST_SHA256
