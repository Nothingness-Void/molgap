"""Synthetic retention tests; no model, data or remote execution."""
import math

from molgap.gptrans_scale_ema_acceptance import (
    _retained_artifacts, _matches_schedule_learning_rate, _transfer_terminal_decision,
)
from molgap.research_memory.schemas import TRAJECTORY_OUTCOMES
from molgap.research_memory.terminal_wiring import inspect_trace_retention_evidence, is_trace_artifact
from molgap.training_reproducibility import atomic_json


def test_shared_live_views_retain_one_unambiguous_canonical_trace(tmp_path):
    paths = [tmp_path / view / name for view in ("ema999", "ema9999")
        for name in ("canonical_trace.json", "trace.json")]
    for path in paths:
        atomic_json(path, {"synthetic": True})
    primary = tmp_path / "ema999/canonical_trace.json"
    artifacts = _retained_artifacts(paths, primary, lambda p: p.relative_to(tmp_path).as_posix())
    assert len(artifacts) == 4
    assert sum(is_trace_artifact(a) for a in artifacts) == 1
    declared, selected = inspect_trace_retention_evidence({"artifacts": artifacts}, arm_identifier="ema999")
    assert declared and selected["locator"] == "ema999/canonical_trace.json"
    assert all(a["sha256"] and a["availability"] == "local_verified" for a in artifacts)


def test_frozen_schedule_accepts_only_final_bit_platform_variation():
    expected = 0.0008486203155433191
    linux_observation = 0.0008486203155433192
    assert _matches_schedule_learning_rate(expected, expected)
    assert _matches_schedule_learning_rate(linux_observation, expected)
    assert not _matches_schedule_learning_rate(expected * 1.00000001, expected)
    three_ulps = expected
    for _ in range(3):
        three_ulps = math.nextafter(three_ulps, math.inf)
    assert not _matches_schedule_learning_rate(three_ulps, expected)
    assert not _matches_schedule_learning_rate(math.nan, expected)
    assert not _matches_schedule_learning_rate(math.inf, expected)


def test_transfer_terminal_uses_outcome_not_comparison_class():
    decision = _transfer_terminal_decision("experiments/study/decision.md")
    assert decision["outcome"] in TRAJECTORY_OUTCOMES
    assert decision["outcome"] == "INCONCLUSIVE"
    assert decision["final"] is True
    assert decision["decision_ref"] == "experiments/study/decision.md"
    assert decision["reopen_conditions"] == ["Separately authorized causal500K comparison"]
