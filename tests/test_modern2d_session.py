"""Actual shared Session/Kernel/optimizer/sampler on generated CPU graphs only."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from molgap.family_training_data import DatasetSelection, TrainingData
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_kernel import FamilyKernel
from molgap.family_training_session import FamilyTrainingSession
from molgap.gptrans_execution_tuning import ExecutionTuning
from molgap.training_reproducibility import configure_fp32_determinism
from molgap.v4_runtime import model_state_sha256
from modern2d_fixtures import factory, model_config, graph
from test_family_training_session import assert_equal


def make_session(tmp_path, family, events):
    configure_fp32_determinism(42)
    torch.set_num_threads(1)
    cfg = model_config(family)
    if family == "ppgt_small": cfg.update(attention_dropout=.1, stochastic_depth=.1)
    initial = factory(family, cfg)
    sha = model_state_sha256(initial)
    path = tmp_path / (family + ".pt")
    torch.save(initial.state_dict(), path)
    selection = DatasetSelection("synthetic/modern2d.json", "a" * 64, True)
    recipe = FamilyRecipe(family, "reference", 42, sha, selection, 256, 3, 4,
        .001, .01, ScheduleSpec("cosine_step", 4), "best-development-live",
        "torch-pass-randperm", checkpoint_every=1, target_mean_eV=3., target_std_eV=2.,
        finite_check_every=1, model_config=cfg,
        torch_version=str(torch.__version__).split("+", 1)[0])
    train = [graph(2, i) for i in range(256)]
    dev = [graph(3, 256 + i) for i in range(3)]
    data = TrainingData(train, dev, selection.manifest_sha256, 256, 3, 0, 256, 3., 2., synthetic=True)
    loaders = FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0))
    kernel = FamilyKernel(recipe, path, device="cpu")
    session = FamilyTrainingSession(kernel, loaders,
        checkpoint=lambda p, s: events.append(deepcopy(s)),
        selected=lambda *a: None, pass_finished=lambda *a: None)
    return session, recipe, data, path


@pytest.mark.parametrize("family", ["ppgt_small", "esa_small"])
def test_exact_actual_session_midpass_resume_and_readonly_eval(tmp_path, family):
    complete, recipe, _, _ = make_session(tmp_path, family, [])
    assert FamilyRecipe.from_dict(json.loads(json.dumps(recipe.to_dict()))).identity == recipe.identity
    result = complete.run()
    expected = deepcopy(complete.state_dict())
    before_eval = deepcopy(complete.state_dict())
    evaluation = complete.kernel.evaluate(complete.loaders)
    assert_equal(before_eval, complete.state_dict())
    assert evaluation.rows == 3
    log = []
    first, _, _, _ = make_session(tmp_path, family, log)
    assert first.run(should_pause=lambda: True).optimizer_step == 1
    resumed, _, _, _ = make_session(tmp_path, family, [])
    resumed.load_state_dict(log[-1])
    assert resumed.run() == result
    assert_equal(expected, resumed.state_dict())
    assert result.optimizer_step == 4 and result.sample_presentations == 512
    # The selected development evaluator was the shared owner, never a new loop.
    assert expected["cursor"]["selected_step"] in {2, 4}


@pytest.mark.parametrize("family", ["ppgt_small", "esa_small"])
def test_cpu_profile_rejects_real_data_gpu_and_contract_changes(tmp_path, family):
    session, recipe, data, initial = make_session(tmp_path, family, [])
    with pytest.raises(ValueError, match="CPU family"):
        FamilyLoaders(recipe, replace(data, synthetic=False), ExecutionTuning(training_workers=0))
    with pytest.raises(ValueError, match="accelerator"):
        FamilyKernel(recipe, initial, device="cuda:0")
    for field, value in (("train_rows", 500000), ("optimizer_steps", 60),
                         ("dataset", DatasetSelection("roles/common_dev.json", "a" * 64, True))):
        with pytest.raises(ValueError, match="CPU synthetic"):
            replace(recipe, **{field: value})
    session.close()


@pytest.mark.parametrize("family", ["ppgt_small", "esa_small"])
def test_changed_model_configuration_cannot_resume_under_old_identity(tmp_path, family):
    log = []
    first, _, _, _ = make_session(tmp_path, family, log)
    first.run(should_pause=lambda: True)
    target, recipe, _, _ = make_session(tmp_path, family, [])
    log[-1]["cursor"]["recipe_identity"] = "f" * 64
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError): target.load_state_dict(log[-1])
    assert_equal(before, target.state_dict())
    target.close()
