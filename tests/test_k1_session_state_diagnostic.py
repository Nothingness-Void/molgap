"""Synthetic translation checks; no real checkpoint or molecule access."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from molgap.k1_frozen_inference import read_session_k1_states
from molgap.family_training_state import _state_identity


def fixture(monkeypatch, kind="selected"):
    from molgap import family_training_runner, training_reproducibility
    context = {"family_name": "neural_atom_k1", "family_version": "3", "training_recipe_sha256": "a" * 64}
    recipe = SimpleNamespace(family="neural_atom_k1", mode="reference", clean_bn=SimpleNamespace(identity="policy"),
        ema_decay=.9999, loop_plan=SimpleNamespace(steps_per_pass=10), optimizer_steps=20, batch_size=128,
        selection="best-development-ema")
    live, raw, clean = ({"weight": torch.tensor([v])} for v in (1., 2., 2.))
    calibration = {"raw_revision": 20, "policy_identity": "policy", "raw_state_identity": _state_identity(raw),
                   "clean_state_identity": _state_identity(clean)}
    owner = {"live": live, "raw_ema": raw, "raw_revision": 20,
             "clean": {"model": clean, "calibration": calibration, "stale": False}}
    payload = {"context": context, "optimizer_step": 20, "owner_state": owner,
               "weights": "ema", "selection": recipe.selection, "epoch_or_pass": 2.,
               "calibration": deepcopy(calibration), "model": clean if kind == "selected" else live,
               "ema": raw, "sample_presentations": 2560, "cursor": {"status": "complete"}}
    monkeypatch.setattr(training_reproducibility, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(family_training_runner, "read_recipe", lambda path: ({}, recipe))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: payload)
    return context, recipe, payload


def call(context, kind="selected"):
    return read_session_k1_states("synthetic.pt", expected_sha256="a" * 64,
        expected_context=context, recipe_path="recipe.json", expected_step=20, checkpoint_kind=kind)


@pytest.mark.parametrize("kind", ["selected", "final"])
def test_translate_exact_same_step_views(monkeypatch, kind):
    context, recipe, payload = fixture(monkeypatch, kind)
    owner, actual = call(context, kind)
    assert owner is payload["owner_state"] and actual is recipe


@pytest.mark.parametrize("fault", ["context", "step", "stale", "revision", "raw", "clean", "selected", "policy"])
def test_reject_unbound_selected_state(monkeypatch, fault):
    context, _, payload = fixture(monkeypatch)
    if fault == "context":
        payload["context"] = {**context, "arm_id": "other"}
    elif fault == "step":
        payload["optimizer_step"] = 10
    elif fault == "stale":
        payload["owner_state"]["clean"]["stale"] = True
    elif fault == "revision":
        payload["owner_state"]["raw_revision"] = 10
    elif fault == "raw":
        payload["owner_state"]["raw_ema"]["weight"] = torch.tensor([3.])
    elif fault == "clean":
        payload["owner_state"]["clean"]["model"]["weight"] = torch.tensor([3.])
    elif fault == "selected":
        payload["model"] = {"weight": torch.tensor([3.])}
    else:
        payload["owner_state"]["clean"]["calibration"]["policy_identity"] = "other"
    with pytest.raises(ValueError):
        call(context)


@pytest.mark.parametrize("fault", ["samples", "cursor", "live", "ema"])
def test_reject_final_binding_fault(monkeypatch, fault):
    context, _, payload = fixture(monkeypatch, "final")
    if fault == "samples":
        payload["sample_presentations"] = 1280
    elif fault == "cursor":
        payload["cursor"]["status"] = "paused"
    else:
        payload["model" if fault == "live" else "ema"] = {"weight": torch.tensor([4.])}
    with pytest.raises(ValueError):
        call(context, "final")


def test_sha_rejection_precedes_deserialization(monkeypatch):
    from molgap import training_reproducibility
    monkeypatch.setattr(training_reproducibility, "sha256_file", lambda path: "b" * 64)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Unbound bytes loaded"))
    with pytest.raises(ValueError, match="bytes differ"):
        call({})
