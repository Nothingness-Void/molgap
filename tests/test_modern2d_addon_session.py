"""Actual tiny shared Sessions; no remote release or real molecular data."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from molgap.experiment_execution import build_addon_declaration
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.model_substrate import SlotBinding, SlotName
from molgap.modern2d_training import Modern2DFormalTrainingProfile
from test_family_training_session import assert_equal
from test_modern2d_formal_training import case, actual_session, native_metadata


def composed(case):
    prefix = "ppgt" if case.recipe.family == "ppgt_small" else "esa"
    addon = build_addon_declaration(case.recipe.interface, prefix + "_readout_scale",
        repo_root=Path(__file__).resolve().parents[1])
    return replace(case.recipe, mode="composed", addons=(addon,),
        formal_training=Modern2DFormalTrainingProfile("configured-screen", horizon_passes=4, horizon_steps=4),
        schedule=ScheduleSpec("cosine_step", 4, reference_epochs=4))


def test_composed_session_updates_and_resumes_addon_and_optimizer(case, monkeypatch):
    case.recipe = composed(case)
    assert FamilyRecipe.from_dict(case.recipe.to_dict()).identity == case.recipe.identity
    full = actual_session(case, [])
    before = {k: v.detach().clone() for k, v in full.kernel.model.state_dict().items() if k.endswith("gain")}
    assert len(before) == 1
    endpoint = full.run()
    expected = deepcopy(full.state_dict())
    assert any(not torch.equal(value, full.kernel.model.state_dict()[key]) for key, value in before.items())
    checkpoints = []
    first = actual_session(case, checkpoints)
    assert first.run(should_pause=lambda: True).optimizer_step == 1
    resumed = actual_session(case, [])
    resumed.load_state_dict(deepcopy(checkpoints[-1]))
    assert resumed.run() == endpoint
    assert_equal(expected, resumed.state_dict())
    full.close()
    first.close()
    resumed.close()


def test_legacy_contract_does_not_accept_new_addon(case):
    configured = composed(case)
    with pytest.raises(ValueError):
        replace(configured, formal_training=Modern2DFormalTrainingProfile())
    with pytest.raises(ValueError, match="readout slot"):
        SlotBinding(case.recipe.family, SlotName.PRE_MESSAGE, 1)


def test_configured_recipe_cannot_reuse_historical_native_authority(native_metadata):
    original = native_metadata.recipe
    recipe = replace(original, formal_training=Modern2DFormalTrainingProfile("configured-screen"))
    context = replace(native_metadata.context, recipe_identity=recipe.identity)
    with pytest.raises(ValueError, match="CPU synthetic admission only"):
        context.validate_recipe(recipe)
