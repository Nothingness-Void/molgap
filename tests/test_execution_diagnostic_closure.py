"""Standalone timing evidence retains traces without causal/replay eligibility."""
from copy import deepcopy

import pytest

from molgap.research_memory.finalize import _excluded_execution_trace


def records():
    frozen = {"state_at_start": {"reference_ids": []},
        "actions": [{"action_id": "A001", "type": "execution_diagnostic"}]}
    package = {"action_id": "A001", "evidence": {"evidence_id": "ev-timing",
        "outcome": {"execution_status": "complete", "comparison_status": "context_only",
            "scientific_status": "INCONCLUSIVE"}}, "decision": {"outcome": "INCONCLUSIVE"}}
    manifest = {"backtest_eligibility": {"eligible": False},
        "comparison_role": "reference", "reference_id": "ev-timing"}
    return frozen, package, manifest


def test_excluded_execution_trace_has_only_a_local_storage_anchor():
    assert _excluded_execution_trace(*records())


@pytest.mark.parametrize("where,path,value", [
    (0, ("state_at_start", "reference_ids"), ["ev-frozen"]),
    (0, ("actions", 0, "type"), "TRAIN"),
    (1, ("comparison_readiness_ref",), "causal.json"),
    (1, ("action_replay",), {}),
    (1, ("evidence", "outcome", "comparison_status"), "strict_causal"),
    (1, ("evidence", "outcome", "execution_status"), "interrupted"),
    (1, ("evidence", "outcome", "scientific_status"), "POSITIVE_UNDER_CONTRACT"),
    (1, ("decision", "outcome"), "POSITIVE_UNDER_CONTRACT"),
    (2, ("backtest_eligibility", "eligible"), True),
    (2, ("comparison_role",), "candidate"),
    (2, ("reference_id",), "ev-unplanned"),
])
def test_storage_exception_cannot_release_scientific_comparison_or_replay(where, path, value):
    values = deepcopy(records())
    target = values[where]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert not _excluded_execution_trace(*values)
