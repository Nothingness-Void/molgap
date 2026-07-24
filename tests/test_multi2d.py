import numpy as np

from molgap.multi2d import add_mean_ensembles, delta_block, targetwise_oracle


def test_equal_mean_and_oracle() -> None:
    truth = np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    predictions = {
        "a": truth + np.asarray([0.2, -0.1, 0.4]),
        "b": truth + np.asarray([-0.2, 0.3, -0.1]),
    }
    combined = add_mean_ensembles(predictions, {"mean_ab": ["a", "b"]})
    np.testing.assert_allclose(
        combined["mean_ab"], (predictions["a"] + predictions["b"]) / 2
    )
    oracle, selected = targetwise_oracle(truth, predictions)
    np.testing.assert_allclose(oracle, truth + np.asarray([0.2, -0.1, -0.1]))
    assert selected.shape == truth.shape


def test_delta_block_without_bootstrap() -> None:
    truth = np.zeros((2, 3))
    baseline = np.ones((2, 3))
    candidate = np.full((2, 3), 0.5)
    result = delta_block(truth, baseline, candidate, n_bootstrap=0)
    assert result["gap"]["delta"] == -0.5
    assert result["average"]["delta"] == -0.5
