"""Static family/addon hooks on the shared accelerated execution core.

Only model-facing differences belong here. No data scale, platform, job, source
package or terminal evidence is selected by this module.
"""
from __future__ import annotations

from copy import copy
from pathlib import Path

from .family_training_recipe import ADAMW_BETAS, ADAMW_EPSILON, FamilyRecipe


def load_family_model(recipe, initial_state):
    """One CPU factory/transport path for release checks and native execution."""
    from .v4_runtime import read_frozen_state_artifact
    import torch
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Shared family construction requires the frozen FP32 default dtype")
    state, initialization = read_frozen_state_artifact(Path(initial_state),
        expected_state_sha256=recipe.initialization_sha256)
    def exact_load(model):
        expected = model.state_dict()
        if expected.keys() != state.keys() or any(
                expected[key].shape != value.shape or expected[key].dtype != value.dtype
                for key, value in state.items()):
            raise ValueError("Frozen initialization tensor keys/shape/dtype differ from the base model")
        model.load_state_dict(state, strict=True)
    if recipe.family == "neural_atom_k1":
        from .qm9_neural_atom import make_encoder
        model = make_encoder("neural_atom_k1")
        exact_load(model)
        if recipe.mode == "ssma":
            from .k1_joint_aggregation import attach_k1_joint_aggregation
            attach_k1_joint_aggregation(model, mode="ssma", layer=6, seed=recipe.seed)
    else:
        from .pcqm_gptrans_v4 import _make_model
        from .gptrans_execution_backend import bind_model_layout
        if recipe.mode == "degree_bond_local_ema999":
            from .gptrans_bond_local import make_bond_local_model
            model = make_bond_local_model()
        else:
            model = _make_model(variant=recipe.mode)
        exact_load(model)
        bind_model_layout(model)
    return model, initialization


def make_family_optimizer(recipe, parameters):
    """One optimizer owner; explicit constants equal the existing native defaults."""
    import torch
    from .v4_runtime import make_adamw_compat
    if recipe.family == "neural_atom_k1":
        return torch.optim.AdamW(parameters, lr=recipe.learning_rate, weight_decay=recipe.weight_decay,
            betas=ADAMW_BETAS, eps=ADAMW_EPSILON)
    return make_adamw_compat(parameters, lr=recipe.learning_rate, weight_decay=recipe.weight_decay,
        betas=ADAMW_BETAS, eps=ADAMW_EPSILON, fused=False, foreach=True)


class FamilyKernel:
    """Exclusively owned by one worker; never shared by concurrent threads."""
    def __init__(self, recipe: FamilyRecipe, initial_state: Path, *, device):
        import torch

        if type(recipe) is not FamilyRecipe:
            raise TypeError("Expected a frozen FamilyRecipe")
        if str(torch.__version__).split("+", 1)[0] != recipe.torch_version:
            raise ValueError("Installed Torch differs from the frozen execution pin")
        self.recipe, self.device = recipe, torch.device(device)
        self.model, self.initialization = load_family_model(recipe, initial_state)
        self.model.to(self.device)
        self.optimizer = make_family_optimizer(recipe, self.model.parameters())
        from .gptrans_execution_hotpath import BoundForeachEMA, ReusedWeightView
        self.ema = BoundForeachEMA(self.model, recipe.ema_decay) if recipe.ema_decay is not None else None
        self.weight_view = ReusedWeightView()

    def _gptrans_batch(self, batch, *, training):
        from .gptrans_execution_backend import prepare_layout
        from .gptrans_execution_hotpath import MODEL_FIELDS
        # Borrow immutable CPU data through a fresh PyG storage wrapper.
        prepared = copy(batch) if hasattr(batch, "execution_layout") else prepare_layout(copy(batch))
        fields = (*MODEL_FIELDS, "y") if training else MODEL_FIELDS
        return prepared.to(self.device, *fields, non_blocking=True)

    def step(self, cpu_batch, *, check_finite):
        mean, std = self.recipe.target_mean_eV, self.recipe.target_std_eV
        if self.recipe.family == "neural_atom_k1":
            from .k1_execution_layout import cpu_layout_context
            from .k1_screen_training import _model_batch_to_device, _optimizer_step
            from .training_reproducibility import deterministic_memory_fill_context
            borrowed = copy(cpu_batch)
            with deterministic_memory_fill_context(enabled=False), cpu_layout_context(
                    self.model, borrowed, self.device):
                batch = _model_batch_to_device(borrowed, self.device, training=True)
                loss, absolute, rows = _optimizer_step(self.model, self.optimizer,
                    batch, mean, std, self.recipe.mode,
                    clip_norm=self.recipe.gradient_clip, check_finite=check_finite)
        else:
            from .family_training_core import normalized_gap_update
            from .gptrans_execution_backend import prepared_forward
            batch = self._gptrans_batch(cpu_batch, training=True)
            loss = normalized_gap_update(self.model, self.optimizer, batch, mean, std,
                forward=prepared_forward, clip_norm=self.recipe.gradient_clip,
                check_finite=check_finite, cast_target_float=True)
            absolute, rows = loss * int(batch.num_graphs), int(batch.num_graphs)
        if self.ema is not None:
            self.ema.update(self.model)
        return loss, absolute, rows

    def forward(self, cpu_batch, *, model=None):
        """One owning forward for Live/Raw views or an independent Clean model."""
        model = self.model if model is None else model
        if self.recipe.family == "neural_atom_k1":
            from .k1_execution_layout import cpu_layout_context
            from .k1_screen_training import _model_batch_to_device, _forward
            borrowed = copy(cpu_batch)
            with cpu_layout_context(model, borrowed, self.device):
                batch = _model_batch_to_device(borrowed, self.device, training=False)
                return _forward(model, batch)
        from .gptrans_execution_backend import prepared_forward
        return prepared_forward(model, self._gptrans_batch(cpu_batch, training=False))

    def close(self):
        self.weight_view.clear()

    def evaluate(self, loaders):
        """Only the explicitly selected internal-development view is consumed."""
        import math
        from types import SimpleNamespace
        from .gptrans_execution_backend import buffered_evaluate
        data = loaders.data
        if self.recipe.identity != loaders.recipe.identity or data.development is None:
            raise ValueError("No matching selected development role")
        weights = self.recipe.selection.rsplit("-", 1)[1]
        with self.weight_view.use(self.model, self.ema, weights):
            # The existing buffered evaluator owns ordered prediction/target
            # accumulation. Keep CPU layout preparation before selective transfer.
            result = buffered_evaluate(SimpleNamespace(DEVELOPMENT_ROWS=data.development_rows),
                lambda model, batch: self.forward(batch), lambda graphs: loaders.dev(),
                self.model, data.development, self.recipe.target_mean_eV, self.recipe.target_std_eV,
                device="cpu", source_idx_start=data.development_source_start)
        if not math.isfinite(result["mae_eV"]):
            raise RuntimeError("Development predictions became nonfinite")
        return result


class ContractSchedule:
    """Native post-update cosine or the existing pre-update GPTrans schedule."""
    def __init__(self, kernel: FamilyKernel):
        import torch
        self.kernel, self.completed_units = kernel, 0
        schedule = kernel.recipe.schedule
        self.native = (torch.optim.lr_scheduler.CosineAnnealingLR(kernel.optimizer,
            T_max=schedule.total_units, eta_min=schedule.minimum_lr)
            if schedule.kind.startswith("cosine_") else None)

    def _set_lr(self, value):
        for group in self.kernel.optimizer.param_groups:
            group["lr"] = value

    def before_pass(self, pass_index, offset):
        schedule = self.kernel.recipe.schedule
        if schedule.kind == "warmup_cosine_epoch":
            self._set_lr(schedule.pre_update_lr(pass_index, self.kernel.recipe.learning_rate))

    def before_step(self, step):
        schedule = self.kernel.recipe.schedule
        if schedule.kind == "four_stage_warmup_cosine_step":
            self._set_lr(schedule.pre_update_lr(step, self.kernel.recipe.learning_rate))

    def after_step(self, step):
        if not self.kernel.recipe.schedule.epoch_based:
            if self.native is not None:
                self.native.step()
            self.completed_units = step

    def after_pass(self, pass_index, progress):
        if self.kernel.recipe.schedule.epoch_based:
            if self.native is not None:
                self.native.step()
            self.completed_units = pass_index + 1

    def state_dict(self):
        return {"completed_units": self.completed_units,
            "native": self.native.state_dict() if self.native is not None else None}

    def load_state_dict(self, state):
        if (type(state) is not dict or set(state) != {"completed_units", "native"}
                or type(state["completed_units"]) is not int or state["completed_units"] < 0
                or state["completed_units"] > self.kernel.recipe.schedule.total_units
                or (state["native"] is None) != (self.native is None)):
            raise ValueError("Schedule resume state differs from frozen policy")
        if self.native is not None:
            if not isinstance(state["native"], dict) or set(state["native"]) != set(self.native.state_dict()):
                raise ValueError("Native scheduler schema differs from the pinned runtime")
            self.native.load_state_dict(state["native"])
        self.completed_units = state["completed_units"]
