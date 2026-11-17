"""CPU toy dropout/BN checks; no molecular data or real K1 forward."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional

from molgap.k1_clean_second import clean_second_view


class LocalGPSBlock(nn.Module):
    """Toy of the existing K1 functional-dropout owner with a BN child."""
    def __init__(self):
        super().__init__()
        self.dropout = 0.4
        self.linear = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x):
        return self.bn(x + functional.dropout(self.linear(x), p=self.dropout,
                                              training=self.training))


class ToyK1(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = LocalGPSBlock()
        self.attention = nn.MultiheadAttention(4, 2, dropout=0.4, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        hidden = self.local(x).unsqueeze(0)
        hidden, _ = self.attention(hidden, hidden, hidden, need_weights=False)
        return self.head(self.dropout(hidden.squeeze(0))).view(-1)


def test_clean_view_all_three_dropout_paths_repeatable_with_bn_training():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        model = ToyK1().train()
        x = torch.randn(8, 4, requires_grad=True)
        modes = {module: module.training for module in model.modules()}
        before_rng = torch.random.get_rng_state().clone()
        with clean_second_view(model):
            assert model.training and model.local.bn.training
            assert model.local.bn.track_running_stats
            assert not model.local.training
            assert not model.attention.training
            assert not model.dropout.training
            first, second = model(x), model(x)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            assert torch.equal(before_rng, torch.random.get_rng_state())
            (first.square().mean() + second.square().mean()).backward()
        assert model.local.bn.num_batches_tracked.item() == 2
        assert all(module.training == mode for module, mode in modes.items())
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_exception_and_nested_context_restore_mixed_modes():
    model = ToyK1().train()
    model.attention.training = False
    modes = {module: module.training for module in model.modules()}
    with pytest.raises(RuntimeError, match="toy"):
        with clean_second_view(model):
            with clean_second_view(model):
                assert not model.local.training and model.local.bn.training
            assert not model.local.training and model.local.bn.training
            raise RuntimeError("toy")
    assert all(module.training == mode for module, mode in modes.items())
    model.eval()
    with clean_second_view(model):
        assert not model.local.bn.training
    assert not any(module.training for module in model.modules())


@pytest.mark.parametrize("dropout_type", [nn.Dropout, nn.Dropout1d, nn.Dropout2d,
                                        nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout])
def test_dropout_nd_owners(dropout_type):
    model = nn.Sequential(dropout_type(0.4), nn.BatchNorm1d(4)).train()
    with clean_second_view(model):
        assert not model[0].training
        assert model[1].training
    assert model[0].training and model[1].training


def test_trainer_two_equal_loss_gradients_one_update_and_changed_learning(monkeypatch):
    import molgap.k1_screen_training as trainer
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        reference = ToyK1().train()
        candidate = copy.deepcopy(reference)
        batch = SimpleNamespace(x=torch.randn(8, 4), y=torch.randn(8))
        predictions, gradients, modes = [], [], []
        def forward(model, payload):
            modes.append((model.local.training, model.attention.training,
                          model.dropout.training, model.local.bn.training))
            prediction = model(payload.x)
            prediction.register_hook(lambda gradient: gradients.append(gradient.clone()))
            predictions.append(prediction)
            return prediction
        monkeypatch.setattr(trainer, "_forward", forward)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01)
        steps = []
        original_step = optimizer.step
        def counted_step(*args, **kwargs):
            steps.append(1)
            return original_step(*args, **kwargs)
        monkeypatch.setattr(optimizer, "step", counted_step)
        rng = torch.random.get_rng_state()
        trainer._optimizer_step(reference, ref_optimizer, batch, 0.0, 1.0, "mean2")
        ref_rng = torch.random.get_rng_state()
        ref_first = predictions[0].detach().clone()
        predictions.clear()
        gradients.clear()
        modes.clear()
        torch.random.set_rng_state(rng)
        loss, absolute, count = trainer._optimizer_step(
            candidate, optimizer, batch, 0.0, 1.0, "mean2_clean_second")
        assert steps == [1] and count == 8
        assert modes == [(True, True, True, True), (False, False, False, True)]
        assert len(predictions) == len(gradients) == 2
        torch.testing.assert_close(predictions[0].detach(), ref_first, rtol=0, atol=0)
        errors = [(prediction.detach() - batch.y).abs() for prediction in predictions]
        torch.testing.assert_close(loss, 0.5 * (errors[0].mean() + errors[1].mean()))
        torch.testing.assert_close(absolute, 0.5 * (errors[0].sum() + errors[1].sum()))
        # Both losses reach their own predictions with equal normalized L1 weight.
        for gradient in gradients:
            assert torch.isfinite(gradient).all()
            torch.testing.assert_close(gradient.abs(), torch.full_like(gradient, 0.5 / count))
        for parameter in candidate.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert candidate.local.bn.num_batches_tracked.item() == 2
        assert reference.local.bn.num_batches_tracked.item() == 2
        assert not torch.equal(torch.random.get_rng_state(), ref_rng)
        assert any(not torch.equal(a, b) for a, b in zip(candidate.parameters(), reference.parameters()))
        assert all(module.training for module in candidate.modules())
