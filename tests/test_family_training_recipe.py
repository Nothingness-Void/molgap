"""Configuration only; no data roles, model execution or platform calls."""
from dataclasses import replace

import pytest

from molgap.family_training_data import DatasetSelection
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec


def recipe(family="neural_atom_k1", rows=100000, *, development=True):
    steps = rows // 128
    return FamilyRecipe(family, "reference", 42, "1" * 64,
        DatasetSelection("fixed.manifest.json", "2" * 64, development),
        rows, 50000 if development else 0, steps * 40,
        4e-4 if family == "neural_atom_k1" else 1e-3, 1e-5,
        ScheduleSpec("cosine_epoch", 40),
        "best-development-live" if development else "final-live",
        "python-epoch-shuffle" if family == "neural_atom_k1" else "torch-pass-randperm",
        steps, target_mean_eV=5.3383002281188965, target_std_eV=1.275090217590332)


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
@pytest.mark.parametrize("rows,development", [(100000, True), (500000, True), (3378606, False)])
def test_same_recipe_interface_for_all_scales_and_families(family, rows, development):
    plan = recipe(family, rows, development=development)
    assert FamilyRecipe.from_dict(plan.to_dict()) == plan
    assert plan.loop_plan.steps_per_pass == rows // 128
    assert plan.identity == FamilyRecipe.from_dict(plan.to_dict()).identity
    assert "platform" not in plan.to_dict()
    assert plan.execution_policy["cpu_layout"] is True


def test_resume_chunk_retention_is_explicit_and_omitted_by_default(tmp_path):
    from types import SimpleNamespace
    from molgap.family_training_runner import _resume_trajectory
    plan = recipe()
    assert "resume_chunks" not in plan.to_dict()
    assert _resume_trajectory(plan, SimpleNamespace(arm_id="reference"), {}) is None
    chunked = replace(plan, resume_chunks=True)
    assert FamilyRecipe.from_dict(chunked.to_dict()) == chunked
    assert chunked.identity != plan.identity
    with pytest.raises(ValueError, match="prospective root"):
        _resume_trajectory(chunked, SimpleNamespace(arm_id="reference"), {})
    target = tmp_path / "prospective/reference/trajectory.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b'{"synthetic":true}')
    assert _resume_trajectory(chunked, SimpleNamespace(arm_id="reference"),
                              {"prospective_root": tmp_path}) == target.read_bytes()
    with pytest.raises(ValueError, match="exact boolean"):
        replace(plan, resume_chunks=1)


@pytest.mark.parametrize("change", ["unknown", "callback", "slow", "fp16", "missing", "schedule_unknown"])
def test_cannot_inject_execution_or_silently_select_another_backend(change):
    record = recipe().to_dict()
    if change == "unknown":
        record["extra"] = True
    elif change == "callback":
        record["worker"] = "module:run"
    elif change == "slow":
        record["execution_policy"]["cpu_layout"] = False
    elif change == "fp16":
        record["execution_policy"]["precision"] = "fp16"
    elif change == "missing":
        record.pop("initialization_sha256")
    else:
        record["schedule"]["callable"] = "module:schedule"
    with pytest.raises((ValueError, TypeError)):
        FamilyRecipe.from_dict(record)


@pytest.mark.parametrize("changes", [
    {"seed": True}, {"batch_size": 64}, {"learning_rate": float("nan")},
    {"learning_rate": 0.}, {"weight_decay": -1.}, {"train_rows": True},
    {"drop_last": 1}, {"ema_decay": .95}, {"mode": "not_registered"},
    {"family": "old_family"}, {"torch_version": "unqualified"},
    {"selection": "best-development-ema"}, {"optimizer_steps": 40 * 781 + 1},
    {"selection": "final-live"}, {"development_rows": 0},
    {"target_mean_eV": float("nan")}, {"target_std_eV": 0.}, {"target_std_eV": True},
])
def test_invalid_frozen_dimensions_fail(changes):
    with pytest.raises((ValueError, TypeError)):
        replace(recipe(), **changes)


def test_midpass_full_budget_is_allowed_without_fake_development():
    full = recipe(rows=3378606, development=False)
    full = replace(full, optimizer_steps=156250, checkpoint_every=500,
        schedule=ScheduleSpec("cosine_step", 156250))
    assert full.loop_plan.steps_per_pass == 26395
    assert full.optimizer_steps % full.loop_plan.steps_per_pass == 24275
    assert full.development_rows == 0


def test_warmup_schedules_match_existing_frozen_formula_exactly():
    from molgap import pcqm_gptrans_v4 as screen, pcqm_gptrans_full_runner as full
    epoch = ScheduleSpec("warmup_cosine_epoch", screen.EPOCHS,
        minimum_lr=screen.MIN_LEARNING_RATE, warmup_units=screen.WARMUP_EPOCHS)
    assert all(epoch.pre_update_lr(i, screen.LEARNING_RATE) == screen.FrozenEpochScheduler.learning_rate(i)
               for i in range(screen.EPOCHS))
    step = ScheduleSpec("four_stage_warmup_cosine_step", full.TOTAL_STEPS,
        minimum_lr=full.MINIMUM_LEARNING_RATE, warmup_units=4)
    for i in (1, 2604, 2605, 5208, 5209, full.WARMUP_STEPS, full.WARMUP_STEPS + 1, full.TOTAL_STEPS):
        assert step.pre_update_lr(i, full.LEARNING_RATE) == full._learning_rate(i)


def test_freezing_recipe_does_not_share_nested_mutable_input():
    record = recipe().to_dict()
    frozen = FamilyRecipe.from_dict(record)
    before = frozen.identity
    record["dataset"]["manifest_sha256"] = "f" * 64
    assert frozen.identity == before
    emitted = frozen.to_dict()
    emitted["schedule"]["total_units"] = 1
    assert frozen.identity == before


def test_normalization_is_frozen_identity_not_recomputed_by_execution_resources():
    frozen = recipe()
    changed = replace(frozen, target_mean_eV=frozen.target_mean_eV + 1e-7)
    assert changed.identity != frozen.identity
    assert FamilyRecipe.from_dict(frozen.to_dict()).target_std_eV == frozen.target_std_eV
