"""Static family/addon hooks on the shared accelerated execution core.

Only model-facing differences belong here. No data scale, platform, job, source
package or terminal evidence is selected by this module.
"""
from __future__ import annotations

from copy import copy
from pathlib import Path

from .family_training_recipe import ADAMW_BETAS, ADAMW_EPSILON, FamilyRecipe
from .family_training_policy import mode_policy


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
        construct = mode_policy(recipe.family, recipe.mode).construct
        model = construct(recipe, state) if construct is not None else make_encoder("neural_atom_k1")
        exact_load(model)
        if recipe.mode == "ssma":
            from .k1_joint_aggregation import attach_k1_joint_aggregation
            attach_k1_joint_aggregation(model, mode="ssma", layer=6, seed=recipe.seed)
    else:
        from .family_model_ops import make_gptrans_backbone
        from .gptrans_execution_backend import bind_model_layout
        construct = mode_policy(recipe.family, recipe.mode).construct
        model = (construct(recipe, state) if construct is not None else
                 make_gptrans_backbone(variant="reference" if recipe.mode == "composed" else recipe.mode))
        exact_load(model)
        bind_model_layout(model)
    if recipe.addons:
        from .experiment_execution import TRAINING_ADAPTERS
        from .shared_model_adapter import apply_reviewed_addons
        interface = (recipe.family, "3" if recipe.family == "neural_atom_k1" else "2")
        model = apply_reviewed_addons(TRAINING_ADAPTERS[interface],
            {"addons": list(recipe.addons)}, model)
        substrate = getattr(model, "_molgap_substrate", None)
        if substrate is None or not substrate.bindings:
            raise ValueError("Accelerated addons must attach to the reviewed substrate slots")
        substrate.seal()
    if recipe.clean_bn is not None:
        from torch.nn.modules.batchnorm import _BatchNorm
        if not any(isinstance(module, _BatchNorm) and module.track_running_stats for module in model.modules()):
            raise ValueError("Clean-BN requires a reviewed model with tracked BatchNorm buffers")
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
        self.initial_state = Path(initial_state)
        self.update_observer = None
        # Only the native reference enables safety fill for the qualification run.
        self._qualification_memory_fill = False
        self._qualification_execution_observer = None
        self.model, self.initialization = load_family_model(recipe, initial_state)
        self.model.to(self.device)
        self.optimizer = make_family_optimizer(recipe, self.model.parameters())
        self._clip_parameters = tuple(self.model.parameters())
        expected = [id(p) for p in self.model.parameters() if p.requires_grad]
        actual = [id(p) for g in self.optimizer.param_groups for p in g["params"] if p.requires_grad]
        if len(set(actual)) != len(actual) or set(actual) != set(expected):
            raise ValueError("Optimizer does not own every trainable backbone/addon parameter exactly once")
        from .gptrans_execution_hotpath import BoundForeachEMA, ReusedWeightView
        self.ema = BoundForeachEMA(self.model, recipe.ema_decay) if recipe.ema_decay is not None else None
        self.weight_view = ReusedWeightView()

    @property
    def training_state(self):
        from .family_training_state import FamilyTrainingState
        if getattr(self, "_training_state", None) is None:
            self._training_state = FamilyTrainingState(self.model, self.ema,
                clean_factory=self.make_clean_model, weight_view=self.weight_view)
        if self._training_state.live_model is not self.model or self._training_state.raw_ema is not self.ema:
            raise ValueError("Live/EMA owner changed after state binding")
        return self._training_state

    def make_clean_model(self):
        return load_family_model(self.recipe, self.initial_state)[0]

    @property
    def evaluation_model(self):
        if getattr(self, "_evaluation_model", None) is None:
            if self.weight_view.busy:
                return self.model
            raise RuntimeError("No active evaluation state")
        return self._evaluation_model

    def calibrate(self, loaders):
        policy = self.recipe.clean_bn
        if policy is None:
            raise ValueError("Clean calibration was not declared")
        return self.training_state.calibrate(policy, loaders.calibration(policy),
            forward=lambda model, batch: self.forward(batch, model=model),
            source_indices=lambda batch: batch.source_idx.reshape(-1))

    def _gptrans_batch(self, batch, *, training):
        from .gptrans_execution_backend import prepare_layout
        from .gptrans_execution_hotpath import MODEL_FIELDS
        # Borrow immutable CPU data through a fresh PyG storage wrapper.
        prepared = copy(batch) if hasattr(batch, "execution_layout") else prepare_layout(copy(batch))
        prepare = mode_policy(self.recipe.family, self.recipe.mode).prepare_batch
        if prepare is not None:
            prepared = prepare(prepared, training=training)
        fields = (*MODEL_FIELDS, "y") if training else MODEL_FIELDS
        return prepared.to(self.device, *fields, non_blocking=True)

    def step(self, cpu_batch, *, check_finite, observer=None):
        observer = observer if observer is not None else getattr(self, "update_observer", None)
        mean, std = self.recipe.target_mean_eV, self.recipe.target_std_eV
        if self.recipe.family == "neural_atom_k1":
            from .k1_execution_layout import cpu_layout_context
            from .k1_execution_backend import execution_context
            from .family_model_ops import k1_batch_to_device, k1_optimizer_step
            from .family_training_core import observed_context, update_scope
            from .training_reproducibility import deterministic_memory_fill_context
            borrowed = copy(cpu_batch)
            with deterministic_memory_fill_context(enabled=False), observed_context(cpu_layout_context(
                    self.model, borrowed, self.device), observer, "layout"), observed_context(execution_context(
                    self.model, borrowed, self.device, self.recipe.k1_execution), observer, "execution_provider"):
                with update_scope(observer, "transfer"):
                    batch = k1_batch_to_device(borrowed, self.device, training=True)
                loss, absolute, rows = k1_optimizer_step(self.model, self.optimizer,
                    batch, mean, std, self.recipe.mode,
                    clip_norm=self.recipe.gradient_clip, check_finite=check_finite,
                    clip_parameters=self._clip_parameters, observer=observer)
        else:
            from .family_training_core import normalized_gap_update
            from .gptrans_execution_backend import prepared_forward
            from .training_reproducibility import deterministic_memory_fill_context
            with deterministic_memory_fill_context(enabled=self._qualification_memory_fill):
                if self._qualification_execution_observer is not None:
                    self._qualification_execution_observer("before_update")
                batch = self._gptrans_batch(cpu_batch, training=True)
                objective = mode_policy(self.recipe.family, self.recipe.mode).objective
                if objective is not None:
                    from .family_training_core import optimizer_update
                    loss, (absolute, rows) = optimizer_update(self.model, self.optimizer,
                        lambda: objective(self.model, batch, prepared_forward, self.recipe),
                        clip_norm=self.recipe.gradient_clip, check_finite=check_finite,
                        clip_parameters=self._clip_parameters, observer=observer)
                else:
                    loss = normalized_gap_update(self.model, self.optimizer, batch, mean, std,
                        forward=prepared_forward, clip_norm=self.recipe.gradient_clip,
                        check_finite=check_finite, cast_target_float=True,
                        clip_parameters=self._clip_parameters, observer=observer)
                    absolute, rows = loss * int(batch.num_graphs), int(batch.num_graphs)
                after_update = mode_policy(self.recipe.family, self.recipe.mode).after_update
                if after_update is not None:
                    after_update(self.model)
                if self._qualification_execution_observer is not None:
                    self._qualification_execution_observer("after_update")
        if self.ema is not None:
            from .family_training_core import update_scope
            with update_scope(observer, "ema"):
                self.training_state.update_raw_ema()
        return loss, absolute, rows

    def forward(self, cpu_batch, *, model=None):
        """One owning forward for Live/Raw views or an independent Clean model."""
        model = self.model if model is None else model
        if self.recipe.family == "neural_atom_k1":
            from contextlib import nullcontext
            from .k1_execution_layout import cpu_layout_context
            from .k1_execution_backend import execution_context
            from .training_reproducibility import deterministic_memory_fill_context
            from .family_model_ops import k1_batch_to_device, k1_forward
            borrowed = copy(cpu_batch)
            config = getattr(self.recipe, "k1_execution", None)
            fill = (nullcontext() if config is None else
                deterministic_memory_fill_context(enabled=config.evaluation_memory_fill))
            with fill, cpu_layout_context(model, borrowed, self.device), execution_context(
                    model, borrowed, self.device, config):
                batch = k1_batch_to_device(borrowed, self.device, training=False)
                return k1_forward(model, batch)
        from .gptrans_execution_backend import prepared_forward
        return prepared_forward(model, self._gptrans_batch(cpu_batch, training=False))

    def close(self):
        self.weight_view.clear()

    def evaluate(self, loaders, *, weights=None, on_evaluation=None):
        """Only the explicitly selected internal-development view is consumed."""
        from .family_training_core import evaluate_development
        data = loaders.data
        if self.recipe.identity != loaders.recipe.identity or data.development is None:
            raise ValueError("No matching selected development role")
        if on_evaluation is not None and not callable(on_evaluation):
            raise TypeError("Evaluation observer must be a trusted callable")
        weights = weights or self.recipe.selection_weights
        if weights not in {"live", "ema", "clean-ema"} or (weights != "live" and self.ema is None):
            raise ValueError("Unavailable evaluation weights")
        def evaluate(model):
            self._evaluation_model = model
            try:
                result = evaluate_development(model, loaders.dev(),
                    forward=lambda model, batch: self.forward(batch) if model is self.model else self.forward(batch, model=model),
                    rows=data.development_rows, source_start=data.development_source_start,
                    mean=self.recipe.target_mean_eV, std=self.recipe.target_std_eV)
                if on_evaluation is not None:
                    on_evaluation(result)
                return result
            finally:
                self._evaluation_model = None
        return self.training_state.evaluate("raw-ema" if weights == "ema" else weights, evaluate,
            calibration_policy=self.recipe.clean_bn if weights == "clean-ema" else None)


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
