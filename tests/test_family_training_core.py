"""Synthetic CPU verification of the shared execution core; no data roles."""
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

from molgap.family_training_core import (
    BatchPlan, LoopPlan, normalized_gap_update, optimizer_update,
    run_training_loop, train_batches,
)


@pytest.mark.parametrize("kwargs", [
    {"physical_batch_size": True}, {"physical_batch_size": 0},
    {"physical_batch_size": 2, "start_step": -1},
    {"physical_batch_size": 2, "stop_step": True},
    {"physical_batch_size": 2, "start_step": 3, "stop_step": 2},
    {"physical_batch_size": 2, "finite_check_every": 0},
    {"physical_batch_size": 2, "require_full_batch": 1},
])
def test_bad_execution_plan_fails(kwargs):
    with pytest.raises(ValueError):
        BatchPlan(**kwargs)


def test_completed_resume_does_not_consume_loader_or_switch_model_mode():
    class Forbidden:
        def __iter__(self):
            raise AssertionError("Completed resume consumed the input")
    model = torch.nn.Linear(1, 1).eval()
    result = list(train_batches(model, Forbidden(), BatchPlan(2, 3, 3), None))
    assert result == [] and not model.training


def test_one_core_obeys_step_boundary_and_schedule_hook_order():
    model = torch.nn.Linear(1, 1)
    events = []
    consumed = []
    def batches():
        try:
            for row in range(5):
                consumed.append(row)
                yield SimpleNamespace(num_graphs=2)
        finally:
            events.append("closed")
    def step(batch, *, check_finite):
        events.append(("step", check_finite))
        return torch.tensor(1.)
    rows = list(train_batches(model, batches(), BatchPlan(2, 3, 5, 2), step,
        before_step=lambda index: events.append(("before", index)),
        after_step=lambda index: events.append(("after", index))))
    assert consumed == [0, 1]
    assert [row.optimizer_step for row in rows] == [4, 5]
    assert events == [("before", 4), ("step", True), ("after", 4),
                      ("before", 5), ("step", False), ("after", 5), "closed"]


def test_interrupted_consumer_closes_input():
    closed = []
    def batches():
        try:
            while True:
                yield SimpleNamespace(num_graphs=2)
        finally:
            closed.append(True)
    stream = train_batches(torch.nn.Linear(1, 1), batches(), BatchPlan(2),
        lambda batch, **kwargs: None)
    next(stream)
    stream.close()
    assert closed == [True]


def test_reject_partial_before_optimizer_or_schedule():
    def forbidden(*args, **kwargs):
        raise AssertionError("Partial batch reached optimizer")
    with pytest.raises(RuntimeError, match="physical batch"):
        list(train_batches(torch.nn.Linear(1, 1), [SimpleNamespace(num_graphs=1)],
            BatchPlan(2), forbidden, before_step=forbidden))


def _equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            _equal(x, y)
    else:
        assert left == right


@pytest.mark.parametrize("family", ["k1", "gptrans_t"])
@pytest.mark.parametrize("steps", [1, 3, 7])
def test_kernel_preserves_original_optimizer_weights_gradients_and_rng(family, steps):
    torch.manual_seed(17)
    original = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(.2),
                                   torch.nn.Linear(4, 1))
    migrated = copy.deepcopy(original)
    x, y = torch.randn(2, 2), torch.randn(2)
    batch = SimpleNamespace(x=x, y=y, num_graphs=2)
    optimizer = torch.optim.AdamW(original.parameters(), lr=4e-4)
    migrated_optimizer = torch.optim.AdamW(migrated.parameters(), lr=4e-4)
    rng = torch.get_rng_state()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.l1_loss(original(x).view(-1), (y - .3) / 1.7)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(original.parameters(), 1.)
        optimizer.step()
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    for _ in range(steps):
        actual = normalized_gap_update(migrated, migrated_optimizer, batch, .3, 1.7,
            forward=lambda model, batch: model(batch.x).view(-1),
            clip_norm=1., check_finite=False, cast_target_float=family == "gptrans_t")
    assert torch.equal(actual, loss.detach())
    _equal(original.state_dict(), migrated.state_dict())
    _equal(optimizer.state_dict(), migrated_optimizer.state_dict())
    for a, b in zip(original.parameters(), migrated.parameters()):
        assert torch.equal(a.grad, b.grad)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_ema_is_updated_once_after_optimizer_and_not_before():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    observed = []
    initial = model.weight.detach().clone()
    ema = SimpleNamespace(update=lambda model: observed.append(model.weight.detach().clone()))
    loss, _ = optimizer_update(model, optimizer, lambda: (model.weight.sum(), None),
        clip_norm=1., check_finite=True, ema=ema)
    assert loss.grad_fn is None
    assert len(observed) == 1
    assert torch.equal(observed[0], model.weight)
    assert not torch.equal(observed[0], initial)


def test_nonfinite_loss_does_not_update_weights():
    model = torch.nn.Linear(1, 1)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    with pytest.raises(RuntimeError, match="loss became non-finite"):
        optimizer_update(model, optimizer,
            lambda: (model.weight.sum() * float("nan"), None),
            clip_norm=1., check_finite=True)
    _equal(before, model.state_dict())


@pytest.mark.parametrize("rows", [6, 14, 30])
def test_same_outer_loop_resumes_exactly_across_data_sizes(rows):
    from molgap.training_reproducibility import capture_rng_state, restore_rng_state
    torch.manual_seed(90)
    base = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.Dropout(.2), torch.nn.Linear(3, 1))
    rng = capture_rng_state()
    plan = LoopPlan(rows, 2, 12, 3)

    def run(interrupt):
        model = copy.deepcopy(base)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        restore_rng_state(rng)
        checkpoints, finished_passes = [], []
        def loader(pass_index, offset):
            return [SimpleNamespace(num_graphs=2,
                x=torch.tensor([[float(i)], [float(i + 1)]]), y=torch.zeros(2))
                for i in range(offset * 2, rows // 2 * 2, 2)]
        def step(batch, *, check_finite):
            return normalized_gap_update(model, optimizer, batch, 0., 1.,
                forward=lambda model, batch: model(batch.x).view(-1),
                clip_norm=1., check_finite=check_finite, cast_target_float=False)
        def checkpoint(progress):
            checkpoints.append((progress, copy.deepcopy(model.state_dict()),
                copy.deepcopy(optimizer.state_dict()), capture_rng_state()))
        result = run_training_loop(model, plan, loader, step, checkpoint,
            after_pass=lambda index, progress: finished_passes.append(index),
            should_pause=(lambda: len(checkpoints) == 1) if interrupt else None)
        if interrupt:
            assert result.status == "paused"
            progress, state, moments, saved_rng = checkpoints[-1]
            model.load_state_dict(state)
            optimizer.load_state_dict(moments)
            restore_rng_state(saved_rng)
            result = run_training_loop(model, plan, loader, step, checkpoint,
                start_step=progress.optimizer_step,
                after_pass=lambda index, progress: finished_passes.append(index))
        assert result.status == "complete"
        assert result.optimizer_step == 12 and result.sample_presentations == 24
        assert result.next_batch == 12 % plan.steps_per_pass
        return model.state_dict(), optimizer.state_dict(), torch.get_rng_state(), finished_passes
    _equal(run(False), run(True))


def test_outer_loop_checkpoint_failure_is_not_acknowledged_completion():
    def failure(progress):
        assert progress.status == "complete"
        raise OSError("atomic publication failed")
    with pytest.raises(OSError, match="atomic publication"):
        run_training_loop(torch.nn.Linear(1, 1), LoopPlan(2, 2, 1, 1),
            lambda pass_index, offset: [SimpleNamespace(num_graphs=2)],
            lambda batch, **kwargs: None, failure)


def test_outer_loop_rejects_short_pass_and_completed_resume_does_not_load():
    model = torch.nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="stream ended"):
        run_training_loop(model, LoopPlan(4, 2, 2, 1),
            lambda pass_index, offset: iter(()), lambda *args, **kwargs: None, lambda progress: None)
    def forbidden(*args):
        raise AssertionError("Completed resume loaded a role")
    result = run_training_loop(model, LoopPlan(4, 2, 2, 1), forbidden, forbidden,
                               forbidden, start_step=2)
    assert result.status == "complete"


def test_outer_loop_counts_nondropped_tail_exactly_before_and_after_resume():
    plan = LoopPlan(5, 2, 6, 2, drop_last=False)
    def loader(pass_index, offset):
        return [SimpleNamespace(num_graphs=rows) for rows in [2, 2, 1][offset:]]
    checkpoints = []
    result = run_training_loop(torch.nn.Linear(1, 1), plan, loader,
        lambda batch, **kwargs: None, checkpoints.append, start_step=2)
    assert result.sample_presentations == 10
    assert result.optimizer_step == 6
    assert result.pass_index == 2 and result.next_batch == 0
    assert [p.sample_presentations for p in checkpoints] == [5, 7, 10]


def test_early_partial_tail_fails_before_optimizer_update():
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected partial batch reached optimizer")
    with pytest.raises(RuntimeError, match="tail/cursor"):
        run_training_loop(torch.nn.Linear(1, 1), LoopPlan(5, 2, 3, 3, drop_last=False),
            lambda pass_index, offset: iter([SimpleNamespace(num_graphs=1)]),
            forbidden, forbidden)


def test_extra_pass_batches_are_not_silently_truncated():
    def forbidden(*args, **kwargs):
        raise AssertionError("Changed loader reached optimizer")
    with pytest.raises(RuntimeError, match="loader length"):
        run_training_loop(torch.nn.Linear(1, 1), LoopPlan(4, 2, 2, 2),
            lambda pass_index, offset: [SimpleNamespace(num_graphs=2)] * 3,
            forbidden, forbidden)
