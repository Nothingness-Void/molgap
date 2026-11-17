import torch

from molgap.architecture_screen_acceptance import _ensemble_metrics, _summary


def test_summary_and_equal_seed_ensemble():
    summary = _summary([-0.003, -0.002, -0.001])
    assert summary["mean"] == -0.002
    payloads = [
        {
            "targets": torch.tensor([[0.0, 0.0, 0.0]]),
            "predictions": torch.tensor([[0.2, 0.1, 0.3]]),
        },
        {
            "targets": torch.tensor([[0.0, 0.0, 0.0]]),
            "predictions": torch.tensor([[0.0, 0.1, 0.1]]),
        },
    ]
    metrics = _ensemble_metrics(payloads)
    assert metrics["HOMO"] == torch.tensor(0.1).item()
    assert metrics["LUMO"] == torch.tensor(0.1).item()
    assert metrics["Gap"] == torch.tensor(0.2).item()


def test_gap_only_equal_seed_ensemble():
    payloads = [
        {
            "targets": torch.tensor([[1.0], [2.0]]),
            "predictions": torch.tensor([[0.8], [2.4]]),
        },
        {
            "targets": torch.tensor([[1.0], [2.0]]),
            "predictions": torch.tensor([[1.0], [2.0]]),
        },
    ]
    metrics = _ensemble_metrics(payloads, "gap")
    assert set(metrics) == {"Gap", "average"}
    assert abs(metrics["Gap"] - 0.15) < 1e-7
    assert metrics["average"] == metrics["Gap"]
