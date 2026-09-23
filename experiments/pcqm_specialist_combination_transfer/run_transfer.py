"""Thin CLI for the frozen K1/PairToken saved-prediction transfer audit."""

from molgap.constants import REPO_ROOT
from molgap.pcqm_specialist_combination_transfer import run_transfer


def main() -> None:
    experiment = REPO_ROOT / "experiments/pcqm_specialist_combination_transfer"
    result = run_transfer(
        REPO_ROOT / "platforms/_records/local/pcqm_molecular_router_audit/aligned_prediction_matrix.pt",
        REPO_ROOT / "platforms/_records/scnet/k1_matched60_reference_122743291/output/neural_atom_k1/best_predictions.pt",
        REPO_ROOT / "platforms/_records/scnet/training/pcqm_k1_pair_token_500k_122312462/best_predictions.pt",
        experiment / "results/three_round_transfer.json",
    )
    for name, round_result in result["500k"]["rounds"].items():
        print(name, round_result["mae_eV"], round_result["exploratory_gate_pass"])


if __name__ == "__main__":
    main()
