"""Saved metadata/array regression tests; no model construction or inference."""
import json

import pytest
import torch

from molgap.constants import REPO_ROOT
from molgap.k1_terminal_analysis import accepted_native_trace, paired_saved_errors
from molgap.k1_screen_trace import recorder, record_epoch
from molgap.research_memory.trace import file_digest

ROOT = REPO_ROOT / "experiments/pcqm_k1_linear_attention_100k"


def test_saved_pair_refuses_row_or_target_misalignment():
    reference = {"source_idx": torch.tensor([1, 2]), "target_eV": torch.tensor([1., 2.]),
                 "prediction_eV": torch.tensor([1.2, 1.9])}
    candidate = {**reference, "source_idx": torch.tensor([2, 1])}
    with pytest.raises(ValueError, match="alignment"):
        paired_saved_errors(reference, candidate)
    candidate = {**reference, "target_eV": torch.tensor([2., 1.])}
    with pytest.raises(ValueError, match="alignment"):
        paired_saved_errors(reference, candidate)


def test_saved_pair_delta_sign_and_posthoc_caveat():
    reference = {"source_idx": torch.arange(10), "target_eV": torch.zeros(10),
                 "prediction_eV": torch.ones(10)}
    candidate = {**reference, "prediction_eV": torch.full((10,), 1.25)}
    summary = paired_saved_errors(reference, candidate)
    assert summary["candidate_minus_reference_eV"] == .25
    assert summary["paired_row_bootstrap"]["ci95"] == [.25, .25]
    assert summary["candidate_win_fraction"] == 0
    assert "not deployable" in summary["subgroup_caveat"]


def test_native_trace_not_reconstructed_from_epoch_seconds(tmp_path):
    (tmp_path / "last_checkpoint.pt").write_bytes(b"synthetic-checkpoint-not-model")
    stream = recorder(tmp_path, "TC-static-native", "static:v1")
    raw = {"epoch": 39, "learning_rate": .000001, "train_normalized_mae": .1,
           "development_gap_mae_eV": .2, "optimizer_steps": 7,
           "sample_presentations": 896, "seconds": 1.0}
    record_epoch(stream, tmp_path, raw, observed_steps=7, observed_samples=896, elapsed=2.5)
    path = tmp_path / "canonical_trace.json"
    digest = file_digest(path)
    trace = accepted_native_trace(path, {"epochs": [raw]}, file_digest(tmp_path / "last_checkpoint.pt"))
    assert trace["observations"][-1]["cumulative_device_time_seconds"] == 2.5
    assert file_digest(path) == digest
    with pytest.raises(ValueError, match="differs"):
        accepted_native_trace(path, {"epochs": [{**raw, "optimizer_steps": 8}]}, "wrong")


def test_training_terminal_has_canonical_science_label_and_native_bytes():
    final = ROOT / "rml_plan/rml_finalized"
    trajectory = json.loads((final / "trajectory.json").read_text())
    evidence = json.loads((final / "v5_evidence.json").read_text())
    assert trajectory["comparison_class"] == "STRICT_CAUSAL"
    assert trajectory["decision"]["outcome"] == evidence["outcome"]["scientific_status"] == "NEGATIVE_UNDER_CONTRACT"
    assert file_digest(final / "trace.json") == file_digest(ROOT / "results/canonical_trace.json")
    costs = list((final / "costs").glob("*.json"))
    assert len(costs) == 1
    cost = json.loads(costs[0].read_text())
    assert cost["cost_event_id"] == "cost-TC-k1-linear-attention-100k-s42"
    trace = json.loads((final / "trace.json").read_text())
    assert cost["measurement"]["device_hours"]["value"] == trace["observations"][-1]["cumulative_device_time_seconds"] / 3600


def test_failed_audit_and_recovery_have_distinct_physical_run_identity():
    failed = json.loads((ROOT / "audit_plan/rml_finalized/trajectory.json").read_text())
    recovered = json.loads((ROOT / "audit_recovery/trajectory.json").read_text())
    assert failed["decision"]["outcome"] == "INFRASTRUCTURE_ONLY"
    assert recovered["record_mode"] == "retrospective_partial"
    assert recovered["decision"]["outcome"] == "NO_TRAIN"
    assert failed["actions"][0]["run_ids"] != recovered["actions"][0]["run_ids"]
    assert "decision_state" not in recovered
    cost = json.loads(next((ROOT / "audit_recovery/costs").glob("*.json")).read_text())
    assert cost["measurement"]["device_hours"] == {"status": "measurement_missing", "value": None}
