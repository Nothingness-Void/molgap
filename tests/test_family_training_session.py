"""Synthetic CPU session replay; native model factories/admission are not exercised."""
from copy import deepcopy

import pytest
import torch
from torch_geometric.data import Data

from molgap.family_training_core import optimizer_update
from molgap.family_training_data import DatasetSelection, TrainingData
from molgap.family_training_kernel import FamilyKernel, make_family_optimizer
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_session import FamilyTrainingSession
from molgap.gptrans_execution_hotpath import BoundForeachEMA, ReusedWeightView
from molgap.gptrans_execution_tuning import ExecutionTuning
from molgap.training_reproducibility import capture_rng_state, configure_fp32_determinism


class SyntheticKernel(FamilyKernel):
    """Use the real session/loader/scheduler with an elementary stochastic kernel."""
    def __init__(self, recipe):
        self.recipe, self.device = recipe, torch.device("cpu")
        self.model = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Dropout(.2), torch.nn.Linear(2, 1))
        self.optimizer = make_family_optimizer(recipe, self.model.parameters())
        self.ema = BoundForeachEMA(self.model, recipe.ema_decay) if recipe.ema_decay else None
        self.weight_view = ReusedWeightView()

    def forward(self, batch):
        return self.model(batch.source_idx.float().reshape(-1, 1) / 257.).reshape(-1)

    def step(self, batch, *, check_finite):
        mean, std = self.recipe.target_mean_eV, self.recipe.target_std_eV
        def loss_builder():
            absolute = (self.forward(batch) - (batch.y.view(-1) - mean) / std).abs()
            return absolute.mean(), (absolute.detach().sum(), batch.num_graphs)
        loss, (absolute, rows) = optimizer_update(self.model, self.optimizer, loss_builder,
            clip_norm=self.recipe.gradient_clip, check_finite=check_finite, ema=self.ema)
        return loss, absolute, rows


def make_session(family, *, development, schedule_kind, events, drop_last=True):
    configure_fp32_determinism(42)
    rows = 257
    train = [Data(x=torch.zeros(1, 9, dtype=torch.long),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        edge_attr=torch.empty(0, 3, dtype=torch.long), random_walk_pe=torch.zeros(1, 16),
        source_idx=torch.tensor([i]), y=torch.tensor([float(i % 7)])) for i in range(rows)]
    dev = [train[i].clone() for i in range(3)] if development else None
    if dev:
        for i, graph in enumerate(dev):
            graph.source_idx = torch.tensor([rows + i])
    steps_per_pass = 2 if drop_last else 3
    budget = steps_per_pass * 2 if development else steps_per_pass * 2 + 1
    units = 4 if schedule_kind.endswith("epoch") else budget
    if schedule_kind == "warmup_cosine_epoch":
        schedule = ScheduleSpec(schedule_kind, units, warmup_units=1)
    elif schedule_kind == "four_stage_warmup_cosine_step":
        schedule = ScheduleSpec(schedule_kind, units, warmup_units=4)
    else:
        schedule = ScheduleSpec(schedule_kind, units)
    selection = DatasetSelection("manifest.json", "1" * 64, development)
    recipe = FamilyRecipe(family, "reference", 42, "2" * 64, selection, rows,
        3 if development else 0, budget, .001, .01, schedule,
        "best-development-ema" if development else "final-ema",
        "python-epoch-shuffle" if family == "neural_atom_k1" else "torch-pass-randperm",
        checkpoint_every=2, target_mean_eV=3., target_std_eV=2., ema_decay=.999,
        finite_check_every=1, drop_last=drop_last)
    data = TrainingData(train, dev, selection.manifest_sha256, rows, 3 if development else 0,
        0, rows if development else None, 3., 2.)
    pool = FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0))
    def checkpoint(progress, state):
        events["checkpoint"].append(deepcopy(state))
    def selected(progress, weights, state, evaluation):
        events["selected"].append((progress, weights, deepcopy(state), deepcopy(evaluation)))
    def report(progress, metrics):
        events["passes"].append((progress, deepcopy(metrics)))
    return FamilyTrainingSession(SyntheticKernel(recipe), pool,
        checkpoint=checkpoint, selected=selected, pass_finished=report)


def events():
    return {"checkpoint": [], "selected": [], "passes": []}


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
@pytest.mark.parametrize("development", [True, False])
@pytest.mark.parametrize("schedule_kind", ["cosine_epoch", "cosine_step",
    "warmup_cosine_epoch", "four_stage_warmup_cosine_step"])
@pytest.mark.parametrize("drop_last", [True, False])
def test_exact_midpass_resume_including_partial_metric_schedule_and_ema(
        family, development, schedule_kind, drop_last):
    baseline_events = events()
    complete = make_session(family, development=development, schedule_kind=schedule_kind,
        events=baseline_events, drop_last=drop_last)
    result = complete.run()
    assert result.status == "complete"
    final = deepcopy(complete.state_dict())
    split_events = events()
    first = make_session(family, development=development, schedule_kind=schedule_kind,
        events=split_events, drop_last=drop_last)
    paused = first.run(should_pause=lambda: True)
    assert paused.optimizer_step == 1 and paused.status == "paused"
    retained = split_events["checkpoint"][-1]
    resumed = make_session(family, development=development, schedule_kind=schedule_kind,
        events=split_events, drop_last=drop_last)
    resumed.load_state_dict(retained)
    assert resumed.run() == result
    assert_equal(final, resumed.state_dict())
    assert_equal(baseline_events["passes"], split_events["passes"])
    assert_equal(baseline_events["selected"], split_events["selected"])
    assert all(item["cursor"]["best_development_mae_eV"] is None
               for item in split_events["checkpoint"]) == (not development)


@pytest.mark.parametrize("corruption", ["recipe", "samples", "cursor", "schedule", "metrics",
    "best", "model", "optimizer", "rng"])
def test_bad_resume_is_rejected_without_mutating_model_optimizer_or_global_rng(corruption):
    log = events()
    original = make_session("neural_atom_k1", development=True, schedule_kind="cosine_step", events=log)
    original.run(should_pause=lambda: True)
    state = deepcopy(log["checkpoint"][-1])
    target = make_session("neural_atom_k1", development=True, schedule_kind="cosine_step", events=events())
    before = deepcopy(target.state_dict())
    if corruption == "recipe":
        state["cursor"]["recipe_identity"] = "f" * 64
    elif corruption == "samples":
        state["sample_presentations"] += 1
    elif corruption == "cursor":
        state["cursor"]["next_batch"] = 0
    elif corruption == "schedule":
        state["scheduler"]["completed_units"] = 0
    elif corruption == "metrics":
        state["cursor"]["pending_absolute"] = []
    elif corruption == "best":
        state["cursor"]["best_development_mae_eV"] = .1
    elif corruption == "model":
        state["model"]["0.weight"] = torch.ones(99)
    elif corruption == "optimizer":
        state["optimizer"]["param_groups"] = []
    else:
        state["rng_state"]["torch"] = torch.ones(3, dtype=torch.uint8)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            target.load_state_dict(state)
        assert_equal(before, target.state_dict())
    finally:
        target.close()


def test_complete_resume_does_not_reconsume_input_or_republish_selection():
    log = events()
    session = make_session("gptrans_t", development=False, schedule_kind="cosine_step", events=log)
    completed = session.run()
    followup = events()
    resumed = make_session("gptrans_t", development=False, schedule_kind="cosine_step", events=followup)
    resumed.load_state_dict(log["checkpoint"][-1])
    before = capture_rng_state()["torch"]
    assert resumed.run() == completed
    assert followup == events()
    assert torch.equal(before, capture_rng_state()["torch"])


@pytest.mark.parametrize("terminal", [False, True])
def test_atomic_publication_failure_cleans_resources_and_never_claims_complete(terminal):
    session = make_session("neural_atom_k1", development=False, schedule_kind="cosine_step", events=events())
    def failed(progress, state):
        if not terminal or progress.status == "complete":
            raise OSError("durable destination unavailable")
    session.checkpoint = failed
    with pytest.raises(OSError, match="durable destination"):
        session.run(should_pause=None if terminal else lambda: True)
    assert session.closed and session.loaders.closed
    assert session.progress.status == "persistence_failed"


@pytest.mark.parametrize("corruption", ["lr", "weight_decay", "betas", "eps", "foreach", "fused",
    "moment_shape", "moment_step", "scheduler_period", "scheduler_base", "scheduler_min", "scheduler_epoch"])
def test_resume_cannot_change_update_contract_even_with_same_recipe_identity(corruption):
    log = events()
    first = make_session("gptrans_t", development=False, schedule_kind="cosine_step", events=log)
    first.run(should_pause=lambda: True)
    state = deepcopy(log["checkpoint"][-1])
    fresh = make_session("gptrans_t", development=False, schedule_kind="cosine_step", events=events())
    before = deepcopy(fresh.state_dict())
    group, native = state["optimizer"]["param_groups"][0], state["scheduler"]["native"]
    if corruption == "lr":
        group["lr"] *= .5
        native["_last_lr"] = [group["lr"]]
    elif corruption == "weight_decay":
        group["weight_decay"] = 0.
    elif corruption == "betas":
        group["betas"] = (.8, .9)
    elif corruption == "eps":
        group["eps"] = 1e-4
    elif corruption == "foreach":
        group["foreach"] = False
    elif corruption == "fused":
        group["fused"] = True
    elif corruption == "moment_shape":
        moments = state["optimizer"]["state"][0]
        moments["exp_avg"] = torch.zeros(99)
        moments["exp_avg_sq"] = torch.zeros(99)
    elif corruption == "moment_step":
        state["optimizer"]["state"][0]["step"] = torch.tensor(99.)
    elif corruption == "scheduler_period":
        native["T_max"] += 1
    elif corruption == "scheduler_base":
        native["base_lrs"] = [.02]
    elif corruption == "scheduler_min":
        native["eta_min"] = .01
    else:
        native["last_epoch"] = True
    try:
        with pytest.raises(ValueError):
            fresh.load_state_dict(state)
        assert_equal(before, fresh.state_dict())
    finally:
        fresh.close()
