"""Synthetic cost-quality mode checks; no molecular roles or remote execution."""
from types import SimpleNamespace

import pytest
import torch

from molgap import k1_screen_training as training
from molgap.experiment_execution import build_family_recipe, training_adapter
from molgap.experiment_spec import ADDONS


def test_recipe_reuses_frozen_exposure_and_registers_mode():
    recipe = build_family_recipe(("neural_atom_k1", "2"), addon="k1_two_pass_mean",
                                 source_idx_sha256="a" * 64, target_sha256="b" * 64)
    training.validate_recipe(recipe, mode="mean2")
    assert recipe["training_recipe"]["ema"] is False
    assert recipe["acceptance_requirements"]["optimizer_steps"] == 31240
    arm = {"family": {"name": "neural_atom_k1", "version": "2"},
           "addons": [{"name": "k1_two_pass_mean", "version": "1"}]}
    assert training_adapter(arm).mode(arm) == "mean2"
    assert ADDONS[("k1_two_pass_mean", "1")].source_module == "molgap.k1_screen_training"


@pytest.mark.parametrize("mode,calls", [("reference", 1), ("mean2", 2)])
def test_optimizer_averages_losses_not_predictions(monkeypatch, mode, calls):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    observed = []

    def forward(model, batch):
        sign = 1.0 if not observed else -1.0
        observed.append(sign)
        return model.weight.view(-1) * sign

    monkeypatch.setattr(training, "_forward", forward)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss, absolute, rows = training._optimizer_step(
        model, optimizer, SimpleNamespace(y=torch.zeros(1)), 0.0, 1.0, mode)
    assert len(observed) == calls
    assert float(loss) == pytest.approx(1.0)
    assert float(absolute) == pytest.approx(1.0)
    assert rows == 1
    assert float(model.weight.detach()) == pytest.approx(0.9)


def test_cost_guard_is_mode_specific_not_a_global_relaxation():
    assert training._maximum_overhead("mean2") == 1.0
    assert training._maximum_overhead("ssma") == 0.25
    assert training._maximum_overhead("reference") == 0.25
