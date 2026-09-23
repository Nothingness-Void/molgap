from molgap.futility_gate import MatchedPrefixGate, evaluate_matched_prefix_futility
from molgap.pcqm_500k_v4_evidence import scientific_contract


GATES = (
    MatchedPrefixGate(30, 0.112521231174469, 0.006),
    MatchedPrefixGate(40, 0.10822822153568268, 0.003),
)


def test_gate_only_evaluates_exact_registered_boundaries() -> None:
    assert evaluate_matched_prefix_futility(
        completed_epochs=29,
        candidate_best_mae_eV=0.2,
        gates=GATES,
    ) is None


def test_clear_loser_stops_at_epoch_30() -> None:
    decision = evaluate_matched_prefix_futility(
        completed_epochs=30,
        candidate_best_mae_eV=0.119,
        gates=GATES,
    )
    assert decision is not None
    assert decision["futility_stopped"] is True


def test_competitive_candidate_continues() -> None:
    decision = evaluate_matched_prefix_futility(
        completed_epochs=40,
        candidate_best_mae_eV=0.109,
        gates=GATES,
    )
    assert decision is not None
    assert decision["futility_stopped"] is False


def test_candidate_contract_does_not_mutate_frozen_reference_contract() -> None:
    reference = scientific_contract()
    candidate = scientific_contract("gptrans_pair_update_norm")
    assert reference["selection_fingerprint"] == (
        "best-development-raw-model-60epochs"
    )
    assert "futility_gates" not in reference
    assert candidate["futility_reference_trace_sha256"].startswith("22cb2bea")
    assert len(candidate["futility_gates"]) == 2
