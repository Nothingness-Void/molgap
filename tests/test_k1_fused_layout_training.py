"""CPU-only addon binding and default-preservation checks; no training roles."""
import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from molgap import k1_screen_training as owner
from molgap.experiment_execution import build_family_recipe, training_adapter


def recipe():
    return build_family_recipe(("neural_atom_k1", "2"), addon="k1_fused_layout",
                              source_idx_sha256="1" * 64, target_sha256="2" * 64)


def test_frozen_addon_and_single_forward_contract():
    value = recipe()
    owner.validate_recipe(value, mode="fused_layout")
    assert value["training_recipe"] == owner.build_screen_recipe(
        "reference", source_idx_sha256="1" * 64, target_sha256="2" * 64)["training_recipe"]
    arm = {"family": {"name": "neural_atom_k1", "version": "2"},
           "addons": [{"name": "k1_fused_layout", "version": "1"}]}
    assert training_adapter(arm).mode(arm) == "fused_layout"


@pytest.mark.parametrize("key,value", [("adamw_fused", False), ("adamw_foreach", True),
                                      ("cpu_layout", False), ("loader", "persistent")])
def test_reject_execution_policy_drift(key, value):
    altered = copy.deepcopy(recipe())
    altered["execution_policy"][key] = value
    with pytest.raises(ValueError, match="execution policy"):
        owner.validate_recipe(altered, mode="fused_layout")


def test_reference_rejects_silent_optimizer_override():
    value = owner.build_screen_recipe("reference", source_idx_sha256="1" * 64, target_sha256="2" * 64)
    value["execution_policy"] = recipe()["execution_policy"]
    with pytest.raises(ValueError, match="overrides"):
        owner.validate_recipe(value, mode="reference")


def test_optimizer_native_defaults_and_fused_are_distinct(monkeypatch):
    import torch
    from molgap import v4_runtime
    calls = []
    monkeypatch.setattr(torch.optim, "AdamW", lambda parameters, **kw: calls.append(kw))
    monkeypatch.setattr(v4_runtime, "make_adamw_compat", lambda parameters, **kw: calls.append(kw))
    model = SimpleNamespace(parameters=lambda: ())
    owner._screen_optimizer(model, "reference")
    owner._screen_optimizer(model, "fused_layout")
    assert calls[0] == {"lr": 4e-4, "weight_decay": 1e-5}
    assert calls[1] == {"lr": 4e-4, "weight_decay": 1e-5, "fused": True, "foreach": False}


@pytest.mark.parametrize("mode", ["reference", "fused_layout"])
def test_layout_wraps_transfer_and_owner_step_then_cleans_up(monkeypatch, mode):
    from molgap import k1_execution_layout
    events = []
    @contextmanager
    def layout(model, batch, device):
        events.append("layout-enter")
        try:
            yield
        finally:
            events.append("layout-exit")
    class Batch:
        def to(self, device, *, non_blocking):
            events.append("transfer")
            return self
    monkeypatch.setattr(k1_execution_layout, "cpu_layout_context", layout)
    monkeypatch.setattr(owner, "_optimizer_step", lambda *args: events.append("owner-step"))
    owner._screen_step(None, None, Batch(), 0, 1, mode)
    assert events == (["layout-enter", "transfer", "owner-step", "layout-exit"]
                      if mode == "fused_layout" else ["transfer", "owner-step"])
