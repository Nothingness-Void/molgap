"""Platform- and dataset-independent execution for the K1/GPTrans families.

Owning runners validate source, data, roles and native qualification before
calling these primitives. This module neither submits jobs nor grants authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class LoopPlan:
    """Only execution dimensions, not a data role or resource authorization."""
    train_rows: int
    physical_batch_size: int
    optimizer_steps: int
    checkpoint_every: int
    finite_check_every: int = 50
    checkpoint_at_pass: bool = True
    drop_last: bool = True

    def __post_init__(self):
        for name in ("train_rows", "physical_batch_size", "optimizer_steps",
                     "checkpoint_every", "finite_check_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("Positive integer required: " + name)
        if self.train_rows < self.physical_batch_size:
            raise ValueError("Training view must contain at least one physical batch")
        if type(self.checkpoint_at_pass) is not bool or type(self.drop_last) is not bool:
            raise ValueError("Checkpoint/tail policies must be exact booleans")

    @property
    def steps_per_pass(self):
        if self.drop_last:
            return self.train_rows // self.physical_batch_size
        return (self.train_rows + self.physical_batch_size - 1) // self.physical_batch_size


@dataclass(frozen=True)
class LoopProgress:
    optimizer_step: int
    sample_presentations: int
    pass_index: int
    next_batch: int
    status: str


def run_training_loop(
    model, plan: LoopPlan, loader: Callable, step: Callable, checkpoint: Callable,
    *, start_step: int = 0, before_pass: Callable | None = None,
    before_step: Callable | None = None, after_step: Callable | None = None,
    after_pass: Callable | None = None, observe_batch: Callable | None = None,
    should_pause: Callable | None = None, report_progress: Callable | None = None,
) -> LoopProgress:
    """One bounded, resumable outer loop for any family, scale or platform.

    Static family adapters bind the trusted callables; JSON never names a
    module/callback. ``loader(pass_index, start_batch)`` returns the authorized
    CPU stream. ``after_pass`` performs the frozen schedule/evaluation/selection
    only at a complete pass. ``checkpoint(progress)`` publishes actual state
    atomically via the existing artifact owner, before a pause/completion is
    acknowledged. Its exception propagates: persistence failure is not success.
    Data/role/source admission and native qualification remain external gates.
    """
    from contextlib import closing

    if type(plan) is not LoopPlan:
        raise TypeError("Expected LoopPlan")
    if type(start_step) is not int or not 0 <= start_step <= plan.optimizer_steps:
        raise ValueError("Resume step is outside the frozen execution budget")
    if not callable(checkpoint):
        raise TypeError("An atomic checkpoint owner is required")

    def progress(current, status):
        pass_index, offset = divmod(current, plan.steps_per_pass)
        rows_per_pass = (plan.steps_per_pass * plan.physical_batch_size
                         if plan.drop_last else plan.train_rows)
        samples = pass_index * rows_per_pass + min(offset * plan.physical_batch_size, rows_per_pass)
        return LoopProgress(current, samples,
                            pass_index, offset, status)

    current = start_step
    if current == plan.optimizer_steps:
        return progress(current, "complete")
    while current < plan.optimizer_steps:
        pass_index, offset = divmod(current, plan.steps_per_pass)
        if before_pass is not None:
            before_pass(pass_index, offset)
        stop = min(plan.optimizer_steps, (pass_index + 1) * plan.steps_per_pass)
        batches = loader(pass_index, offset)
        if hasattr(batches, "__len__") and len(batches) != plan.steps_per_pass - offset:
            raise RuntimeError("Training loader length differs from its frozen pass/cursor")
        batch_plan = BatchPlan(plan.physical_batch_size, current, stop, plan.finite_check_every,
                               require_full_batch=plan.drop_last)
        def bound_step(batch, *, check_finite):
            expected_rows = plan.physical_batch_size if plan.drop_last else min(
                plan.physical_batch_size, plan.train_rows - (current % plan.steps_per_pass) * plan.physical_batch_size)
            if int(batch.num_graphs) != expected_rows:
                raise RuntimeError("Batch rows differ from the frozen tail/cursor policy")
            return step(batch, check_finite=check_finite)
        with closing(train_batches(model, batches, batch_plan, bound_step,
                                   before_step=before_step, after_step=after_step)) as stream:
            for processed in stream:
                current = processed.optimizer_step
                if observe_batch is not None:
                    observe_batch(processed)
                pass_finished = current % plan.steps_per_pass == 0
                if pass_finished and after_pass is not None:
                    after_pass(pass_index, progress(current, "running"))
                complete = current == plan.optimizer_steps
                paused = not complete and should_pause is not None and should_pause()
                status = "complete" if complete else "paused" if paused else "running"
                if (current % plan.checkpoint_every == 0 or
                        (pass_finished and plan.checkpoint_at_pass) or complete or paused):
                    checkpoint(progress(current, status))
                if report_progress is not None:
                    report_progress(progress(current, status))
                if complete or paused:
                    return progress(current, status)
        if current != stop:
            raise RuntimeError("Training stream ended before its frozen pass/budget endpoint")
    return progress(current, "complete")


@dataclass(frozen=True)
class BatchPlan:
    physical_batch_size: int
    start_step: int = 0
    stop_step: int | None = None
    finite_check_every: int = 50
    require_full_batch: bool = True

    def __post_init__(self):
        for name in ("physical_batch_size", "finite_check_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("Positive integer required: " + name)
        if type(self.start_step) is not int or self.start_step < 0:
            raise ValueError("Nonnegative integer start_step required")
        if (self.stop_step is not None and
                (type(self.stop_step) is not int or self.stop_step < self.start_step)):
            raise ValueError("stop_step must not precede start_step")
        if type(self.require_full_batch) is not bool:
            raise ValueError("require_full_batch must be an exact bool")


@dataclass(frozen=True)
class BatchResult:
    optimizer_step: int
    rows: int
    value: object


def train_batches(
    model, batches: Iterable, plan: BatchPlan, step: Callable,
    *, before_step: Callable | None = None, after_step: Callable | None = None,
    before_fetch: Callable | None = None,
):
    """Execute one stream; the caller owns sampling, schedule and persistence.

    ``step(batch, check_finite=...)`` is an owning family's bound kernel, not a
    dynamic module selected by a config. Results stay on-device. The stop check
    precedes iterator creation/consumption, so a completed resume consumes no
    loader RNG or additional input. Closing the generator closes its stream.
    """
    if type(plan) is not BatchPlan:
        raise TypeError("Expected BatchPlan")
    if plan.stop_step == plan.start_step:
        return
    model.train()
    iterator = iter(batches)
    current = plan.start_step
    try:
        while plan.stop_step is None or current < plan.stop_step:
            if before_fetch is not None:
                before_fetch(current + 1)
            try:
                batch = next(iterator)
            except StopIteration:
                break
            rows = int(batch.num_graphs)
            if (rows < 1 or rows > plan.physical_batch_size or
                    (plan.require_full_batch and rows != plan.physical_batch_size)):
                raise RuntimeError("Training batch differs from the frozen physical batch")
            next_step = current + 1
            if before_step is not None:
                before_step(next_step)
            value = step(batch, check_finite=next_step % plan.finite_check_every == 0)
            if after_step is not None:
                after_step(next_step)
            current = next_step
            yield BatchResult(current, rows, value)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


@dataclass(frozen=True)
class UpdateObserver:
    """Trusted worker-local observation; never selected by configuration imports."""
    after_backward: Callable | None = None
    after_clip: Callable | None = None
    scope: Callable | None = None

    def __post_init__(self):
        if any(value is not None and not callable(value) for value in
               (self.after_backward, self.after_clip, self.scope)):
            raise TypeError("Update observation hooks must be trusted callables")


def update_scope(observer, name):
    from contextlib import nullcontext
    if observer is not None and type(observer) is not UpdateObserver:
        raise TypeError("Expected UpdateObserver")
    return observer.scope(name) if observer is not None and observer.scope is not None else nullcontext()


class _ObservedContext:
    def __init__(self, context, observer, label):
        self.context, self.observer, self.label = context, observer, label

    def __enter__(self):
        with update_scope(self.observer, self.label + "_enter"):
            return self.context.__enter__()

    def __exit__(self, *exception):
        with update_scope(self.observer, self.label + "_exit"):
            return self.context.__exit__(*exception)


def observed_context(context, observer, label):
    return context if observer is None else _ObservedContext(context, observer, label)


def optimizer_update(
    model, optimizer, loss_builder: Callable, *, clip_norm: float,
    check_finite: bool, ema=None, label: str = "Training", clip_parameters=None, observer=None,
):
    """Shared zero-grad/backward/clip/update order; loss stays family-owned.

    Builders return (scalar loss, auxiliary detached metrics). No scalar metric
    is copied to the CPU here. Optimizer mode and EMA policy remain explicit in
    the owning frozen contract rather than inferred from GPU type or data size.
    """
    import math
    import torch

    if (type(clip_norm) not in (int, float) or not math.isfinite(clip_norm)
            or clip_norm <= 0 or type(check_finite) is not bool):
        raise ValueError("Invalid gradient clipping or finite-check policy")
    with update_scope(observer, "zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    loss, metrics = loss_builder()
    if check_finite and not bool(torch.isfinite(loss)):
        raise RuntimeError(label + " loss became non-finite")
    with update_scope(observer, "backward"):
        loss.backward()
    if observer is not None and observer.after_backward is not None:
        observer.after_backward(model)
    with update_scope(observer, "gradient_clip"):
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters() if clip_parameters is None else clip_parameters, clip_norm)
    if observer is not None and observer.after_clip is not None:
        observer.after_clip(model, gradient_norm)
    if check_finite and not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError(label + " gradient became non-finite")
    with update_scope(observer, "optimizer"):
        optimizer.step()
    if ema is not None:
        with update_scope(observer, "ema"):
            ema.update(model)
    return loss.detach(), metrics


def normalized_gap_update(
    model, optimizer, batch, mean, std, *, forward: Callable,
    clip_norm: float, check_finite: bool, cast_target_float: bool,
    ema=None, label: str = "Training", clip_parameters=None, observer=None,
):
    """Single-forward Gap L1 kernel shared by both families and all scales."""
    import torch.nn.functional as functional

    def loss_builder():
        prediction = forward(model, batch)
        target = batch.y.view(-1)
        if cast_target_float:
            target = target.float()
        target = (target - mean) / std
        return functional.l1_loss(prediction, target), None

    return optimizer_update(model, optimizer, loss_builder, clip_norm=clip_norm,
        check_finite=check_finite, ema=ema, label=label, clip_parameters=clip_parameters, observer=observer)[0]


def evaluate_development(model, batches, *, forward, rows, source_start, mean, std):
    """One ordered FP32/eV evaluator; the state owner installs the selected view."""
    import torch
    predictions, targets, sources = [], [], []
    with torch.no_grad():
        for batch in batches:
            targets.append(batch.y.reshape(-1).float().cpu())
            index = getattr(batch, "source_idx", getattr(batch, "row_index", None))
            if index is None:
                raise ValueError("Development source identity is missing")
            sources.append(index.reshape(-1).long().cpu())
            predictions.append(forward(model, batch).reshape(-1) * std + mean)
    prediction, target, source = torch.cat(predictions).cpu(), torch.cat(targets), torch.cat(sources)
    if (prediction.shape != target.shape or prediction.numel() != rows or
            not torch.equal(source, torch.arange(source_start, source_start + rows)) or
            not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all())):
        raise ValueError("Development count/order/finite predictions differ")
    return {"mae_eV": float((prediction - target).abs().mean()), "prediction_eV": prediction,
            "target_eV": target, "source_idx": source}
