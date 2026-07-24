import numpy as np

from molgap.hierarchical_oracle import hierarchical_oracle_analysis


def test_budget_is_molecule_level_and_residual_is_bounded() -> None:
    truth = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    base = np.asarray([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
    expert = np.asarray([[-1.0, 4.0], [1.0, 3.0], [2.0, 2.0]])

    report, arrays = hierarchical_oracle_analysis(
        truth,
        base,
        expert,
        target_names=("a", "b"),
        budgets=(1 / 3,),
    )

    assert arrays["switch_33pct"].tolist() == [True, False, False]
    assert report["budgets"]["33pct"]["called_molecules"] == 1
    assert np.isclose(report["methods"]["switch_33pct"]["average_mae_eV"], 10 / 6)
    assert np.all((arrays["optimal_alpha"] >= 0.0) & (arrays["optimal_alpha"] <= 1.0))
    assert (
        report["methods"]["unconstrained_residual"]["average_mae_eV"]
        <= report["methods"]["unconstrained_switch"]["average_mae_eV"]
    )


def test_invalid_shapes_are_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        hierarchical_oracle_analysis(
            np.zeros((2, 1)),
            np.zeros((3, 1)),
            np.zeros((2, 1)),
            target_names=("gap",),
        )
