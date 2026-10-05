import pytest
import torch

from molgap import pcqm_500k_v4_evidence as runner


def test_composed_factory_requires_pinned_checkpoint():
    for arm in runner.COMPOSED_ARMS:
        with pytest.raises(ValueError, match="pinned initial-state"):
            runner.make_model(arm)


@pytest.mark.parametrize("fail", [False, True])
def test_ema_evaluation_restores_live_parameters(monkeypatch, fail):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)

    class EMA:
        def state_dict(self):
            return {"weight": torch.tensor([[2.0]])}

    def evaluate(observed, *args):
        assert float(observed.weight) == 2.0
        if fail:
            raise RuntimeError("evaluation failed")
        return {"mae_eV": 0.5}

    monkeypatch.setattr(runner, "evaluate", evaluate)
    if fail:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            runner._evaluate_ema(model, EMA(), [], 0.0, 1.0)
    else:
        assert runner._evaluate_ema(model, EMA(), [], 0.0, 1.0) == {"mae_eV": 0.5}
    assert float(model.weight) == 1.0
