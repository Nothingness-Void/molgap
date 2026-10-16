"""Versioned ESA correction reuses the actual Session, not a second trainer."""
from copy import deepcopy
from dataclasses import replace
import json
import pytest
import torch

from molgap.esa_invariant import ESAInvariant, ESAInvariantConfig
from molgap.experiment_execution import TRAINING_ADAPTERS,validate_training_registry,validate_execution_plan
from molgap.experiment_spec import ExperimentSpec
from molgap.family_training_recipe import FamilyRecipe
from test_experiment_spec import arm,payload
from test_modern2d_session import make_session
from test_family_training_session import assert_equal


def test_version_two_actual_interruption_resume_matches_uninterrupted(tmp_path):
    complete,recipe,_,_=make_session(tmp_path,"esa_small",[],version="2")
    assert recipe.interface == ("esa_small","2") and recipe.cpu_synthetic_only
    assert FamilyRecipe.from_dict(json.loads(json.dumps(recipe.to_dict()))).identity==recipe.identity
    complete.run();expected=deepcopy(complete.state_dict())
    readonly,_,_,_=make_session(tmp_path,"esa_small",[],version="2")
    readonly.load_state_dict(deepcopy(expected))
    evaluation=readonly.kernel.evaluate(readonly.loaders)
    assert evaluation["source_idx"].numel()==3
    assert_equal(expected,readonly.state_dict());readonly.close()
    events=[];first,_,_,_=make_session(tmp_path,"esa_small",events,version="2")
    first.run(should_pause=lambda:True)
    from molgap.training_reproducibility import atomic_torch_save
    checkpoint=tmp_path/"esa_v2_resume.pt"
    atomic_torch_save(checkpoint,events[-1])
    serialized=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert_equal(events[-1],serialized)
    resumed,_,_,_=make_session(tmp_path,"esa_small",[],version="2")
    resumed.load_state_dict(serialized);resumed.run()
    assert_equal(expected,resumed.state_dict())


def test_same_shaped_v1_checkpoint_cannot_resume_under_v2_mathematics(tmp_path):
    events=[];old,recipe,_,_=make_session(tmp_path,"esa_small",events)
    assert "family_version" not in recipe.to_dict()
    old.run(should_pause=lambda:True)
    fixed,updated,_,_=make_session(tmp_path,"esa_small",[],version="2")
    assert updated.identity != recipe.identity
    before=deepcopy(fixed.state_dict())
    with pytest.raises(ValueError):fixed.load_state_dict(events[-1])
    assert_equal(before,fixed.state_dict());fixed.close()


def test_v2_static_declaration_remains_denied_for_platform_execution(payload):
    validate_training_registry(); declaration=arm("esa_small")
    declaration["family"]["version"]="2";payload["arms"]=[declaration]
    assert TRAINING_ADAPTERS[("esa_small","2")].model_factory=="molgap.esa_invariant:make_model"
    with pytest.raises(ValueError,match="CPU synthetic"):
        validate_execution_plan(ExperimentSpec(payload),[])
    m=ESAInvariant();assert sum(p.numel() for p in m.parameters())==5014849


@pytest.mark.parametrize("version", ["unknown", 2, True, ["2"]])
def test_unknown_or_untyped_versions_fail_closed(tmp_path,version):
    session,recipe,_,_=make_session(tmp_path,"esa_small",[],version="2")
    with pytest.raises(ValueError):replace(recipe,family_version=version)
    session.close()
