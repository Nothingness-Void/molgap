"""Fresh research releases preserve history and explicitly bind the fast path."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from molgap.family_training_recipe import FamilyRecipe
from molgap.modern2d_training import Modern2DFormalTrainingProfile
from molgap.experiment_input import _read_scientific_recipe, resolve_experiment_input
from test_modern2d_formal_training import case
from test_modern2d_scientific_input import configured
from test_experiment_workflow import _write_json, _git


def test_original_execution_default_is_wire_compatible(case):
    wire = case.recipe.to_dict()
    assert "execution_mode" not in wire["formal_training"]
    assert FamilyRecipe.from_dict(wire).identity == case.recipe.identity


def test_batched_execution_is_explicit_ppgt_only(case):
    base = case.recipe
    profile = Modern2DFormalTrainingProfile(purpose="configured-screen",
        horizon_passes=60, horizon_steps=base.optimizer_steps, execution_mode="batched-v1")
    schedule = replace(base.schedule, total_units=base.optimizer_steps)
    if base.family != "ppgt_small":
        with pytest.raises(ValueError, match="unrelated family"):
            replace(base, formal_training=profile, schedule=schedule)
    else:
        recipe = replace(base, formal_training=profile, schedule=schedule)
        assert recipe.to_dict()["formal_training"]["execution_mode"] == "batched-v1"
        assert FamilyRecipe.from_dict(recipe.to_dict()).identity == recipe.identity
    with pytest.raises(ValueError, match="explicit new configured"):
        Modern2DFormalTrainingProfile(execution_mode="batched-v1")


def test_scientific_source_reissue_needs_new_authority(configured):
    c = configured
    path = c.repo / c.value["arms"][0]["recipe_ref"]
    record = json.loads(path.read_bytes())
    family = (record["family"], record["family_version"])
    record["execution_policy"]["feature_transform_source_sha256"] = "0" * 64
    _write_json(path, record)
    authority = dict(purpose="CONFIGURED_SCREEN", scientific_training_authorized=True,
                     source_reissue_authorized=True)
    retained, current = _read_scientific_recipe(path, family, authority)
    assert retained == record
    assert current.execution_policy["feature_transform_source_sha256"] != "0" * 64
    for delta in (dict(source_reissue_authorized=False), dict(scientific_training_authorized=False)):
        with pytest.raises(ValueError, match="Execution policy differs"):
            _read_scientific_recipe(path, family, {**authority, **delta})


def test_batched_scientific_input_preserves_mode(configured):
    c = configured
    if c.value["arms"][0]["model"]["family"] != "ppgt_small":
        return
    value, authority = deepcopy(c.value), deepcopy(c.authority)
    value["arms"][0]["training"]["execution_mode"] = "batched-v1"
    from molgap.research_memory.trace import file_digest
    qualification = "experiments/question/native_qualification.json"
    _write_json(c.repo / qualification, dict(native_certificate_status="accepted",
        execution_repair=dict(family="ppgt_small", update_equivalent=True, scientific_training_released=False)))
    authority.update(execution_mode="batched-v1", execution_qualification_ref=dict(
        path=qualification, sha256=file_digest(c.repo / qualification)))
    _write_json(c.path, value)
    _write_json(c.repo / value["authority_ref"], authority)
    _git(c.repo, "add", ".")
    _git(c.repo, "-c", "commit.gpgsign=false", "commit", "-m", "Explicit new batched screen")
    _, _, workflow, _ = resolve_experiment_input(c.path, c.repo)
    record = json.loads((c.repo / workflow["arms"][0]["recipe"]).read_bytes())
    assert record["formal_training"]["execution_mode"] == "batched-v1"
    assert qualification in workflow["source_records"]
