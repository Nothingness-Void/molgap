"""Scientific checks for the frozen saved-prediction transfer rules."""

import hashlib
import json

import numpy as np
import pytest

from molgap.constants import REPO_ROOT
from molgap.pcqm_specialist_combination_transfer import (
    _arrays,
    _fuse,
    apply_disagreement_bins,
    fit_disagreement_bins,
    fit_global_weight,
)
from molgap.research_memory.schemas import (
    validate_cost_event,
    validate_role_event,
    validate_trajectory,
)
from molgap.v5_common import validate_v5_evidence_envelope


def test_global_weight_prefers_better_prediction():
    target = np.zeros(10_000)
    base = np.full(10_000, 0.02)
    expert = np.full(10_000, 0.2)
    assert fit_global_weight(target, base, expert) == 1.0


def test_disagreement_rule_uses_predictions_without_target_at_application():
    target = np.zeros(50_000)
    base = np.linspace(0.01, 0.5, 50_000)
    expert = np.zeros(50_000)
    rule = fit_disagreement_bins(target, base, expert, 0.5)
    prediction, counts = apply_disagreement_bins(base, expert, rule)
    assert sum(counts) == 50_000
    assert len(rule["weights_on_k1"]) == 5
    assert np.allclose(prediction, _fuse(base, expert, np.asarray(rule["weights_on_k1"])[np.digitize(np.abs(base - expert), rule["thresholds_eV"], right=True)]))


def test_row_identity_fails_closed():
    rows = np.arange(100_000, 150_000)
    values = np.zeros(50_000)
    rows[-1] = 150_000
    with pytest.raises(ValueError, match="frozen source range"):
        _arrays(rows, values, values, values, 100_000)


def test_three_round_evidence_and_native_rml_records():
    root = REPO_ROOT / "experiments/pcqm_specialist_combination_transfer"
    result_path = root / "results/three_round_transfer.json"
    acceptance = json.loads((root / "results/acceptance.json").read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == acceptance["result_sha256"]
    assert set(result["500k"]["rounds"]) == {
        "R1_fixed_equal", "R2_global_weight", "R3_disagreement_bins"
    }
    assert result["official_validation_role_read"] is False
    assert result["test_dev_role_read"] is False
    assert result["test_challenge_role_read"] is False
    validate_trajectory(json.loads((root / "trajectory.json").read_text(encoding="utf-8")))
    for path in (root / "costs").glob("*.json"):
        validate_cost_event(json.loads(path.read_text(encoding="utf-8")))
    for path in (root / "roles").glob("*.json"):
        validate_role_event(json.loads(path.read_text(encoding="utf-8")))
    validate_v5_evidence_envelope(
        json.loads((root / "v5_evidence.json").read_text(encoding="utf-8")),
        repo_root=REPO_ROOT,
    )
