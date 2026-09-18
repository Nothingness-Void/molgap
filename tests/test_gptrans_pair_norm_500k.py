import json
from pathlib import Path

from molgap.futility_gate import MatchedPrefixGate, evaluate_matched_prefix_futility
from molgap.pcqm_500k_v4_evidence import (
    PAIR_UPDATE_NORM_FUTILITY_GATES,
    PARAMETERS,
    scientific_contract,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/pcqm_gptrans_pair_norm_500k"


def test_contract_and_documented_contract_agree() -> None:
    documented = json.loads((EXPERIMENT / "training_contract.json").read_text())
    executable = scientific_contract("gptrans_pair_update_norm")
    assert documented["parameters"] == PARAMETERS["gptrans_pair_update_norm"]
    assert documented["futility_gates"] == executable["futility_gates"]
    assert documented["reference_trace_sha256"] == (
        executable["futility_reference_trace_sha256"]
    )


def test_preregistered_backtest_keeps_positive_and_stops_negative() -> None:
    evidence = json.loads((EXPERIMENT / "backtest.json").read_text())
    positive = evidence["accepted_100k_pair_update_norm"]
    negative = evidence["rejected_100k_pair_post_norm"]
    for completed, key in ((30, "candidate_minus_reference_best_eV_at_epoch30"),
                           (40, "candidate_minus_reference_best_eV_at_epoch40")):
        gate = next(
            item for item in PAIR_UPDATE_NORM_FUTILITY_GATES
            if item.completed_epochs == completed
        )
        positive_decision = evaluate_matched_prefix_futility(
            completed_epochs=completed,
            candidate_best_mae_eV=gate.reference_best_mae_eV + positive[key],
            gates=PAIR_UPDATE_NORM_FUTILITY_GATES,
        )
        negative_decision = evaluate_matched_prefix_futility(
            completed_epochs=completed,
            candidate_best_mae_eV=gate.reference_best_mae_eV + negative[key],
            gates=PAIR_UPDATE_NORM_FUTILITY_GATES,
        )
        assert positive_decision["futility_stopped"] is False
        assert negative_decision["futility_stopped"] is True


def test_gate_objects_remain_immutable_value_objects() -> None:
    assert all(isinstance(item, MatchedPrefixGate) for item in PAIR_UPDATE_NORM_FUTILITY_GATES)

