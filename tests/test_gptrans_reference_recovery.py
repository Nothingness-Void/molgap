"""Compact GPTrans V4 reference recovery invariants (no remote artifacts)."""

import json
from pathlib import Path

from molgap.comparison_readiness import (
    validate_reference_bundle,
    validate_target_transform_asset,
    validate_trace_plan,
)


ROOT = Path(__file__).resolve().parents[1]
RECOVERED = ROOT / "experiments/pcqm_gptrans_author_alignment/recovered_reference"


def read(name: str) -> dict:
    return json.loads((RECOVERED / name).read_text(encoding="utf-8"))


def test_gptrans_reference_is_aligned_and_structurally_valid():
    bundle = validate_reference_bundle(read("reference_bundle.json"))
    transform = validate_target_transform_asset(read("target_transform.json"))
    acceptance = read("reference_acceptance.json")
    prediction = read("prediction_manifest.json")
    assert bundle["prediction_manifest"]["source_idx_sha256"] == prediction["source_idx_sha256"]
    assert bundle["prediction_manifest"]["target_sha256"] == prediction["target_sha256"]
    assert bundle["comparison_identity"]["target_transform_asset_sha256"] == transform["asset_sha256"]
    assert acceptance["prediction_mae_eV"] == prediction["development_gap_mae_eV"]
    assert acceptance["training_executed"] is False
    assert acceptance["model_inference_executed"] is False


def test_gptrans_historical_trace_cannot_claim_v5_strict_causal_readiness():
    trace = read("trace_manifest.json")
    acceptance = read("reference_acceptance.json")
    assert trace["backtest_eligibility"]["eligible"] is False
    assert acceptance["strict_causal_trace_ready"] is False
    observed_fields = {
        "optimizer_step": True,
        "sample_presentations": True,
        "epoch_or_pass": True,
        "learning_rate": True,
        "live_train_metric": True,
        "live_dev_metric": False,
        "ema_dev_metric": True,
        "checkpoint_identity": True,
    }
    try:
        validate_trace_plan(
            observed_fields,
            experiment_purpose="mechanism_comparison",
            ema_enabled=True,
        )
    except ValueError as error:
        assert "live_dev_metric" in str(error)
    else:
        raise AssertionError("Historical EMA-only trace must not pass strict gate")
