"""Offline postmortem invariants over accepted, source-only PCQM records."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/pcqm_scale_transfer_reassessment/analyze.py"


def _analyze():
    spec = importlib.util.spec_from_file_location("pcqm_scale_transfer_reassessment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze(ROOT)


def test_joint_gain_contract_and_horizon() -> None:
    result = _analyze()
    matched = result["matched_500k"]
    joint = matched["arms"]["joint"]
    assert matched["reference_best_development_eV"] == 0.10686753690242767
    assert 0.0020 < joint["best_selected_gain_eV"] < 0.0021
    assert joint["snapshots"][0]["same_step_development_gain_eV"] > 0.018
    assert joint["last_10_mean_gain_eV"] < 0.003
    assert joint["terminal_online_train_difference_eV"] > 0.008
    assert result["cross_scale_context"]["joint_projection_shortfall_eV"] > 0.005


def test_full_budget_and_generated_result_are_reproducible() -> None:
    result = _analyze()
    full = result["full_context"]
    assert full["gptrans_selected_total_presentations"] == 40271360
    assert full["k1_completed_total_presentations"] == 30135680
    assert full["edge_state_nominal_selected_presentations_if_complete_passes"] > 100000000
    saved = json.loads((ROOT / "experiments/pcqm_scale_transfer_reassessment/analysis.json").read_text(encoding="utf-8"))
    assert saved == result
