"""Common resumable family session; admission and artifact publication stay external.

Trusted owning adapters bind the event callbacks after source, role, runtime and
resource admission. This is not a CLI, release certificate or RML validator.
"""
from __future__ import annotations

from copy import deepcopy
import math

from ..family_training_core import LoopProgress, run_training_loop
from ..family_training_kernel import ContractSchedule, FamilyKernel
from ..family_training_loaders import FamilyLoaders


def validate_optimizer_schedule(recipe, optimizer, scheduler, *, step, pass_index, offset):
    """Validate frozen update semantics without constructing a model or optimizer."""
    import torch
    from ..family_training_recipe import ADAMW_BETAS, ADAMW_EPSILON
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1 or type(groups[0]) is not dict:
        raise ValueError("Expected the single frozen AdamW parameter group")
    group = groups[0]
    frozen = {"betas": ADAMW_BETAS, "eps": ADAMW_EPSILON, "weight_decay": recipe.weight_decay,
        "amsgrad": False, "maximize": False, "capturable": False, "differentiable": False,
        "foreach": None if recipe.family == "neural_atom_k1" else True,
        "fused": None if recipe.family == "neural_atom_k1" else False}
    for name, value in frozen.items():
        if name not in group or type(group[name]) is not type(value) or group[name] != value:
            raise ValueError("Resume changed frozen AdamW field: " + name)
    if "decoupled_weight_decay" in group and group["decoupled_weight_decay"] is not True:
        raise ValueError("Resume changed AdamW decoupled weight decay")
    ids = group.get("params")
    if not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids) or ids != list(range(len(ids))):
        raise ValueError("Resume parameter mapping is not the frozen single-group ordering")
    moments = optimizer.get("state")
    if not isinstance(moments, dict) or not moments or not set(moments) <= set(ids):
        raise ValueError("Missing or unbound optimizer moments")
    for record in moments.values():
        if type(record) is not dict or set(record) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Incomplete AdamW moments")
        count = record["step"]
        if (not isinstance(count, torch.Tensor) or count.ndim != 0 or not count.is_floating_point() or
                not bool(torch.isfinite(count)) or not 1 <= float(count) <= step or float(count) != int(float(count))):
            raise ValueError("Optimizer update exposure is outside acknowledged work")
        first, second = record["exp_avg"], record["exp_avg_sq"]
        if (not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor) or
                not first.is_floating_point() or first.shape != second.shape or first.dtype != second.dtype):
            raise ValueError("AdamW moment tensor schema differs")
    schedule = recipe.schedule
    units = pass_index if schedule.epoch_based else step
    if (type(scheduler) is not dict or set(scheduler) != {"completed_units", "native"}
            or type(scheduler["completed_units"]) is not int or scheduler["completed_units"] != units):
        raise ValueError("Scheduler exposure differs from the frozen update axis")
    if schedule.kind.startswith("cosine_"):
        native = scheduler["native"]
        if (type(native) is not dict or type(native.get("T_max")) is not int or native.get("T_max") != schedule.total_units or
                native.get("eta_min") != schedule.minimum_lr or native.get("base_lrs") != [recipe.learning_rate] or
                type(native.get("last_epoch")) is not int or native.get("last_epoch") != units or
                type(native.get("_step_count")) is not int or native.get("_step_count") != units + 1 or
                native.get("_last_lr") != [group.get("lr")] or group.get("initial_lr") != recipe.learning_rate):
            raise ValueError("Resume changed frozen native cosine schedule")
        expected_lr = schedule.minimum_lr + (recipe.learning_rate - schedule.minimum_lr) * .5 * (
            1 + math.cos(math.pi * units / schedule.total_units))
    else:
        if scheduler["native"] is not None:
            raise ValueError("Warmup schedule cannot restore a native cosine scheduler")
        epoch = pass_index if offset else pass_index - 1
        expected_lr = schedule.pre_update_lr(epoch if schedule.epoch_based else step, recipe.learning_rate)
    lr = group.get("lr")
    # Native cosine uses an iterative recurrence, not its closed-form expression.
    # This check permits double-precision roundoff only; it never rewrites LR.
    if type(lr) not in (float, int) or not math.isfinite(lr) or not math.isclose(
            lr, expected_lr, rel_tol=1e-10, abs_tol=1e-15):
        raise ValueError("Resume learning rate differs from the frozen trajectory")


def validate_session_resume(recipe, state):
    """One metadata/finite-state check shared by replay loading and CPU inspection."""
    import torch
    from ..experiment_family_workflow import _finite_tree, _validate_rng_state
    required = {"model", "optimizer", "scheduler", "ema", "rng_state", "cursor",
                "optimizer_step", "sample_presentations"}
    if type(state) is not dict or set(state) != required:
        raise ValueError("Incomplete family resume state")
    cursor, plan, step = state["cursor"], recipe.loop_plan, state["optimizer_step"]
    keys = {"recipe_identity", "pass_index", "next_batch", "status", "pending_absolute",
            "pending_rows", "best_development_mae_eV", "selected_step", "optimizer_state_ids"}
    if recipe.clean_bn is not None:
        keys.add("weight_state")
    if (type(cursor) is not dict or set(cursor) != keys or cursor["recipe_identity"] != recipe.identity
            or type(step) is not int or not 1 <= step <= plan.optimizer_steps):
        raise ValueError("Resume identity or step differs from the frozen recipe")
    pass_index, offset = divmod(step, plan.steps_per_pass)
    rows_per_pass = plan.steps_per_pass * plan.physical_batch_size if plan.drop_last else plan.train_rows
    pending_rows = min(offset * plan.physical_batch_size, rows_per_pass)
    if (type(cursor["pass_index"]) is not int or type(cursor["next_batch"]) is not int or
            (cursor["pass_index"], cursor["next_batch"]) != (pass_index, offset) or
            type(state["sample_presentations"]) is not int or
            state["sample_presentations"] != pass_index * rows_per_pass + pending_rows or
            type(cursor["pending_rows"]) is not int or cursor["pending_rows"] != pending_rows or
            cursor["status"] not in {"running", "paused", "complete"} or
            (cursor["status"] == "complete") != (step == plan.optimizer_steps)):
        raise ValueError("Resume cursor/exposure differs from acknowledged work")
    values = cursor["pending_absolute"]
    if (type(values) is not list or len(values) != offset or any(
            not isinstance(value, torch.Tensor) or value.ndim != 0 or not value.is_floating_point()
            or not bool(torch.isfinite(value)) or bool(value < 0) for value in values)):
        raise ValueError("Resume partial-pass metrics are incomplete")
    if values and any(value.dtype != values[0].dtype for value in values):
        raise ValueError("Resume partial-pass metric dtype changed")
    best, selected = cursor["best_development_mae_eV"], cursor["selected_step"]
    if best is not None and (type(best) not in (float, int) or not math.isfinite(best) or best < 0):
        raise ValueError("Invalid retained development metric")
    if selected is not None and (type(selected) is not int or not 1 <= selected <= step):
        raise ValueError("Invalid selected optimizer step")
    if recipe.dataset.development:
        if (best is None) != (selected is None) or ((pass_index > 0) != (best is not None)):
            raise ValueError("Resume has missing development selection history")
        if selected is not None and selected % plan.steps_per_pass:
            raise ValueError("Development selection did not occur at a complete pass")
    elif best is not None or selected != (step if step == plan.optimizer_steps else None):
        raise ValueError("Final-only recipe contains development selection")
    units = pass_index if recipe.schedule.epoch_based else step
    if type(state["scheduler"]) is not dict or state["scheduler"].get("completed_units") != units:
        raise ValueError("Resume schedule cursor differs from optimizer exposure")
    if (state["ema"] is None) != (recipe.ema_decay is None):
        raise ValueError("Resume EMA policy changed")
    for name in ("model", "optimizer", "scheduler"):
        if not isinstance(state[name], dict) or not state[name]:
            raise ValueError("Incomplete family resume " + name)
    optimizer = state["optimizer"]
    if not isinstance(optimizer.get("state"), dict) or not optimizer.get("param_groups"):
        raise ValueError("Incomplete optimizer resume state")
    validate_optimizer_schedule(recipe, optimizer, state["scheduler"], step=step, pass_index=pass_index, offset=offset)
    initialized = cursor["optimizer_state_ids"]
    if (type(initialized) is not list or any(type(index) is not int for index in initialized)
            or any(type(index) is not int for index in optimizer["state"])
            or initialized != sorted(optimizer["state"])):
        raise ValueError("Resume lost acknowledged optimizer moments")
    # SSMA can legitimately skip its conditional residual on an edgeless batch.
    # Other registered modes update every parameter; do not permit lazy reset.
    if recipe.mode != "ssma" and set(initialized) != set(optimizer["param_groups"][0].get("params", [])):
        raise ValueError("Resume is missing required optimizer moments")
    if any(not isinstance(value, torch.Tensor) for value in state["model"].values()):
        raise ValueError("Invalid resume model tensors")
    if state["ema"] is not None and (not isinstance(state["ema"], dict) or
            state["ema"].keys() != state["model"].keys() or any(
                not isinstance(value, torch.Tensor) or value.shape != state["model"][key].shape
                or value.dtype != state["model"][key].dtype for key, value in state["ema"].items())):
        raise ValueError("Resume EMA tensor schema differs from model")
    _finite_tree(state, "family.resume")
    _validate_rng_state(state["rng_state"])
    if recipe.clean_bn is not None:
        from ..family_training_state import validate_training_state
        component = cursor["weight_state"]
        if (type(component) is not dict or set(component) != {"format", "raw_revision", "clean"}
                or component["raw_revision"] != step):
            raise ValueError("Calibration revision differs from EMA update exposure")
        validate_training_state({**component, "live": dict(state["model"]), "raw_ema": state["ema"]},
            state["model"], raw_available=True, calibration_policy=recipe.clean_bn, allow_missing_clean=True)
        if step == plan.optimizer_steps and (component["clean"] is None or component["clean"]["stale"]):
            raise ValueError("Clean endpoint lacks fresh calibrated state")
    return LoopProgress(step, state["sample_presentations"], pass_index, offset, cursor["status"])


class FamilyTrainingSession:
    """One algorithm at all scales; scale-specific contracts select explicit policies."""
    def __init__(self, kernel: FamilyKernel, loaders: FamilyLoaders, *, checkpoint,
                 selected, pass_finished, batch_finished=None, checkpoint_builder=None):
        if not isinstance(kernel, FamilyKernel) or not isinstance(loaders, FamilyLoaders):
            raise TypeError("Expected family kernel and persistent input pool")
        if kernel.recipe.identity != loaders.recipe.identity:
            raise ValueError("Kernel/input recipe mismatch")
        if not all(callable(hook) for hook in (checkpoint, selected, pass_finished)):
            raise TypeError("Atomic checkpoint, selection and reporting owners are required")
        self.kernel, self.loaders, self.recipe = kernel, loaders, kernel.recipe
        self.schedule = ContractSchedule(kernel)
        self.checkpoint, self.selected, self.pass_finished = checkpoint, selected, pass_finished
        if batch_finished is not None and not callable(batch_finished):
            raise TypeError("Batch reporting must be callable")
        if checkpoint_builder is not None and not callable(checkpoint_builder):
            raise TypeError("Checkpoint builder sink must be a trusted callable")
        self.batch_finished, self.checkpoint_builder = batch_finished, checkpoint_builder
        self.progress = LoopProgress(0, 0, 0, 0, "running")
        self.pending_absolute, self.pending_rows = [], 0
        self._metric_host, self._metric_dtype = [], None
        self.best, self.selected_step = None, None
        self.closed = False

    def _flush_metrics(self):
        import torch
        if self.pending_absolute:
            first = self.pending_absolute[0]
            if any(value.ndim != 0 or value.dtype != first.dtype or value.device != first.device
                    or not value.is_floating_point() for value in self.pending_absolute):
                raise ValueError("Expected matching floating-point batch metrics")
            if self._metric_dtype is not None and self._metric_dtype != first.dtype:
                raise ValueError("Partial-pass metric dtype changed")
            self._metric_dtype = first.dtype
            # Only new scalars cross to CPU. Python floats preserve each FP32/64
            # value exactly; reporting still adds them in original batch order.
            self._metric_host.extend(torch.stack(self.pending_absolute).detach().cpu().tolist())
            self.pending_absolute = []

    def _cursor(self, optimizer_state):
        import torch
        self._flush_metrics()
        values = list(torch.tensor(self._metric_host, dtype=self._metric_dtype).unbind()) if self._metric_host else []
        cursor = {"recipe_identity": self.recipe.identity, "pass_index": self.progress.pass_index,
            "next_batch": self.progress.next_batch, "status": self.progress.status,
            "pending_absolute": values, "pending_rows": self.pending_rows,
            "best_development_mae_eV": self.best, "selected_step": self.selected_step,
            "optimizer_state_ids": sorted(optimizer_state["state"])}
        if self.recipe.clean_bn is not None:
            cursor["weight_state"] = self.kernel.training_state.calibration_state_dict()
        return cursor

    def _checkpoint_state(self):
        from ..experiment_family_workflow import tensor_safe_rng_state
        from ..training_reproducibility import capture_rng_state
        optimizer_state = self.kernel.optimizer.state_dict()
        state = {"model": self.kernel.model.state_dict(), "optimizer": optimizer_state,
            "scheduler": self.schedule.state_dict(),
            "ema": self.kernel.ema.state_dict() if self.kernel.ema is not None else None,
            "rng_state": tensor_safe_rng_state(capture_rng_state()), "cursor": self._cursor(optimizer_state),
            "optimizer_step": self.progress.optimizer_step,
            "sample_presentations": self.progress.sample_presentations}
        return state

    def state_dict(self):
        from ..experiment_family_workflow import _finite_tree
        state = self._checkpoint_state()
        _finite_tree(state, "family.checkpoint")
        return state

    def load_state_dict(self, state):
        """Restore same-recipe work, never resume a different budget or data identity."""
        import torch
        from ..experiment_family_workflow import _finite_tree, _validate_rng_state
        from ..training_reproducibility import restore_rng_state
        if self.closed or self.progress.optimizer_step:
            raise ValueError("Resume requires a fresh session")
        progress = validate_session_resume(self.recipe, state)
        cursor = state["cursor"]
        current = self.kernel.model.state_dict()
        if (not isinstance(state["model"], dict) or current.keys() != state["model"].keys() or any(
                not isinstance(state["model"][key], torch.Tensor) or
                value.shape != state["model"][key].shape or value.dtype != state["model"][key].dtype
                for key, value in current.items())):
            raise ValueError("Resume model tensor schema changed")
        expected_groups = self.kernel.optimizer.state_dict()["param_groups"]
        observed_groups = state["optimizer"]["param_groups"]
        for expected, observed in zip(expected_groups, observed_groups):
            keys = set(expected) - {"lr", "initial_lr"}
            if any(observed.get(key) != expected[key] for key in keys):
                raise ValueError("Resume optimizer defaults or parameter ordering changed")
        names = {id(value): name for name, value in self.kernel.model.named_parameters()}
        for parameter_id, parameter in enumerate(self.kernel.optimizer.param_groups[0]["params"]):
            moments = state["optimizer"]["state"].get(parameter_id)
            if (moments is None and parameter.requires_grad and
                    not (self.recipe.mode == "ssma" and names[id(parameter)].startswith("k1_joint_aggregation."))):
                raise ValueError("Resume lost required parameter moments")
            if moments is not None and any(moments[key].shape != parameter.shape or moments[key].dtype != parameter.dtype
                    for key in ("exp_avg", "exp_avg_sq")):
                raise ValueError("Resume optimizer moments do not match the live parameter schema")
        _finite_tree(state, "family.resume")
        _validate_rng_state(state["rng_state"])
        previous = deepcopy(self.state_dict())
        def install(value):
            if self.recipe.clean_bn is not None:
                component = value["cursor"]["weight_state"]
                self.kernel.training_state.load_state_dict({**component, "live": dict(value["model"]),
                    "raw_ema": value["ema"]},
                    calibration_policy=self.recipe.clean_bn if component["clean"] is not None else None)
            else:
                self.kernel.model.load_state_dict(value["model"], strict=True)
            self.kernel.optimizer.load_state_dict(value["optimizer"])
            self.schedule.load_state_dict(value["scheduler"])
            if self.kernel.ema is not None:
                self.kernel.ema.load_state_dict(value["ema"])
            restore_rng_state(value["rng_state"])
        try:
            install(state)
        except Exception:
            install(previous)
            raise
        self.pending_absolute = []
        values = cursor["pending_absolute"]
        self._metric_host = torch.stack(values).detach().cpu().tolist() if values else []
        self._metric_dtype = values[0].dtype if values else None
        self.pending_rows = cursor["pending_rows"]
        self.best, self.selected_step = cursor["best_development_mae_eV"], cursor["selected_step"]
        self.progress = progress

    def _select(self, progress, evaluation=None):
        weights = self.recipe.selection_weights
        if weights == "clean-ema":
            self.kernel.calibrate(self.loaders)
        self.kernel.training_state.evaluate("raw-ema" if weights == "ema" else weights,
            lambda model: self.selected(progress, weights, model.state_dict(), evaluation),
            calibration_policy=self.recipe.clean_bn if weights == "clean-ema" else None)
        self.selected_step = progress.optimizer_step

    def _finish_pass(self, pass_index, progress):
        from ..training_reproducibility import sum_scalar_metrics
        self.progress = progress
        self.schedule.after_pass(pass_index, progress)
        self._flush_metrics()
        metric = sum_scalar_metrics(self._metric_host) / self.pending_rows
        self.pending_absolute, self.pending_rows = [], 0
        self._metric_host, self._metric_dtype = [], None
        evaluation = None
        live_evaluation = None
        if self.recipe.dataset.development:
            if self.recipe.selection_weights == "clean-ema":
                self.kernel.calibrate(self.loaders)
            observed = []
            def select_evaluation(result):
                if observed:
                    raise RuntimeError("Evaluation observer requires one active weight view")
                observed.append(result)
                if self.best is None or result["mae_eV"] < self.best:
                    weights = self.recipe.selection_weights
                    self.selected(progress, weights, self.kernel.evaluation_model.state_dict(), result)
                    self.selected_step = progress.optimizer_step
                    self.best = result["mae_eV"]
            evaluation = self.kernel.evaluate(self.loaders, on_evaluation=select_evaluation)
            if len(observed) != 1 or observed[0] is not evaluation:
                raise RuntimeError("Evaluation adapter did not retain its observed result")
            from ..family_training_policy import mode_policy
            policy = mode_policy(self.recipe.family, self.recipe.mode)
            for weights in policy.extra_development_weights:
                if weights != self.recipe.selection_weights:
                    if weights != "live":
                        raise ValueError("No reporting field for this extra development state")
                    live_evaluation = self.kernel.evaluate(self.loaders, weights=weights)
        metrics = {"normalized_train_gap_mae": metric, "development": evaluation}
        if live_evaluation is not None:
            metrics["development_live"] = live_evaluation
        self.pass_finished(progress, metrics)

    def _persist(self, progress):
        from ..training_reproducibility import finite_validation_scope
        self.progress = progress
        try:
            with finite_validation_scope():
                if progress.status == "complete" and not self.recipe.dataset.development:
                    self._select(progress)
                if self.checkpoint_builder is None:
                    self.checkpoint(progress, self.state_dict())
                else:
                    # Registered publication validates the full snapshot once;
                    # generic callbacks still receive independently checked state.
                    self.checkpoint_builder(progress, self._checkpoint_state)
        except Exception:
            self.progress = LoopProgress(progress.optimizer_step, progress.sample_presentations,
                progress.pass_index, progress.next_batch, "persistence_failed")
            raise

    def run(self, *, should_pause=None):
        if self.closed:
            raise RuntimeError("Family session is closed")
        def step(batch, *, check_finite):
            return self.kernel.step(batch, check_finite=check_finite)
        def observe(processed):
            _, absolute, rows = processed.value
            if rows != processed.rows:
                raise RuntimeError("Family kernel metric exposure differs from batch rows")
            self.pending_absolute.append(absolute)
            self.pending_rows += rows
        def report(progress):
            self.batch_finished(progress, learning_rates=[group["lr"] for group in self.kernel.optimizer.param_groups])
        try:
            if self.recipe.dataset.development:
                from inspect import Parameter, signature
                parameters = signature(self.kernel.evaluate).parameters
                observer = parameters.get("on_evaluation")
                if not (observer is not None and observer.kind in {
                        Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}) and not any(
                        item.kind == Parameter.VAR_KEYWORD for item in parameters.values()):
                    raise TypeError("Evaluation adapter must accept the on_evaluation keyword")
            return run_training_loop(self.kernel.model, self.recipe.loop_plan, self.loaders.train,
                step, self._persist, start_step=self.progress.optimizer_step,
                before_pass=self.schedule.before_pass, before_step=self.schedule.before_step,
                after_step=self.schedule.after_step, after_pass=self._finish_pass,
                observe_batch=observe, should_pause=should_pause,
                report_progress=report if self.batch_finished is not None else None)
        finally:
            self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.loaders.close()
            finally:
                self.kernel.close()


# Retained pickles continue to name the historical owner.
for _compat_object in (validate_optimizer_schedule, validate_session_resume, FamilyTrainingSession,):
    _compat_object.__module__ = "molgap.family_training_session"
del _compat_object
