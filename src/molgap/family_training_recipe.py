"""Typed execution recipes for the shared K1/GPTrans trainer.

This is an executable configuration, not a replacement V5/RML schema or a
resource/role release. Platform names and mount paths do not select algorithms.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math

from .family_training_core import LoopPlan
from .family_training_data import DatasetSelection
from .screen_policy import canonical_fingerprint


RECIPE_FORMAT = "molgap-family-training-recipe-v1"
ADAMW_BETAS = (.9, .999)
ADAMW_EPSILON = 1e-8
MODES = {
    "neural_atom_k1": ("reference", "ssma", "mean2", "mean2_clean_second"),
    "gptrans_t": ("reference", "pair_prenorm", "centered_logits", "memory_value", "memory_message", "degree_bond_local_ema999"),
}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid integer: " + name)


def _number(value, name, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError("Invalid finite number: " + name)


@dataclass(frozen=True)
class ScheduleSpec:
    kind: str
    total_units: int
    minimum_lr: float = 1e-6
    warmup_units: int = 0
    reference_epochs: int = 60

    def __post_init__(self):
        if self.kind not in {"cosine_epoch", "cosine_step", "warmup_cosine_epoch",
                             "four_stage_warmup_cosine_step"}:
            raise ValueError("Unsupported frozen schedule")
        _integer(self.total_units, "schedule.total_units")
        _number(self.minimum_lr, "schedule.minimum_lr")
        _integer(self.warmup_units, "schedule.warmup_units", 0)
        _integer(self.reference_epochs, "schedule.reference_epochs")
        if self.kind.startswith("cosine_") and self.warmup_units != 0:
            raise ValueError("Cosine schedule has no warmup")
        if self.kind == "warmup_cosine_epoch" and not 0 < self.warmup_units < self.total_units:
            raise ValueError("Epoch warmup must leave a cosine tail")
        if self.kind == "four_stage_warmup_cosine_step" and (
                self.warmup_units != 4 or self.reference_epochs <= 4 or
                math.ceil(self.total_units * 4 / self.reference_epochs) >= self.total_units):
            raise ValueError("Four-stage warmup requires four stages and a positive prefix")

    @property
    def epoch_based(self):
        return self.kind.endswith("epoch")

    def pre_update_lr(self, unit, peak_lr):
        """Exact existing GPTrans warmup/cosine policies, parameterized by budget."""
        if type(unit) is not int:
            raise ValueError("Schedule unit must be an exact integer")
        if self.kind == "warmup_cosine_epoch":
            if not 0 <= unit < self.total_units:
                raise ValueError("Epoch is outside the frozen schedule")
            if unit < self.warmup_units:
                return peak_lr * float(unit + 1) / self.warmup_units
            progress = float(unit - self.warmup_units) / max(1, self.total_units - self.warmup_units - 1)
        elif self.kind == "four_stage_warmup_cosine_step":
            if not 1 <= unit <= self.total_units:
                raise ValueError("Step is outside the frozen schedule")
            warmup_steps = math.ceil(self.total_units * 4 / self.reference_epochs)
            if unit <= warmup_steps:
                stage = min(4, math.ceil(unit * self.reference_epochs / self.total_units))
                return peak_lr * stage / 4
            progress = (unit - warmup_steps) / (self.total_units - warmup_steps)
        else:
            raise ValueError("Post-update cosine is owned by the native Torch scheduler")
        return self.minimum_lr + (peak_lr - self.minimum_lr) * .5 * (1 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class FamilyRecipe:
    family: str
    mode: str
    seed: int
    initialization_sha256: str
    dataset: DatasetSelection
    train_rows: int
    development_rows: int
    optimizer_steps: int
    learning_rate: float
    weight_decay: float
    schedule: ScheduleSpec
    selection: str
    sampler: str
    checkpoint_every: int
    target_mean_eV: float
    target_std_eV: float
    ema_decay: float | None = None
    batch_size: int = 128
    gradient_clip: float = 1.0
    finite_check_every: int = 50
    drop_last: bool = True
    torch_version: str = "2.11.0"

    def __post_init__(self):
        from .experiment_spec import _digest
        if self.family not in MODES or self.mode not in MODES[self.family]:
            raise ValueError("Unknown executable family/addon mode")
        _integer(self.seed, "seed", 0)
        if self.seed > 2**32 - 1:
            raise ValueError("Seed exceeds the supported RNG range")
        _digest(self.initialization_sha256, "initialization_sha256")
        if type(self.dataset) is not DatasetSelection or type(self.schedule) is not ScheduleSpec:
            raise TypeError("Typed dataset and schedule are required")
        for key in ("train_rows", "optimizer_steps", "checkpoint_every", "finite_check_every", "batch_size"):
            _integer(getattr(self, key), key)
        _integer(self.development_rows, "development_rows", 0)
        if self.batch_size != 128:
            raise ValueError("Live family training retains physical batch 128")
        for key in ("learning_rate", "weight_decay", "gradient_clip"):
            _number(getattr(self, key), key)
        _number(self.target_mean_eV, "target_mean_eV")
        _number(self.target_std_eV, "target_std_eV")
        if self.target_std_eV <= 0:
            raise ValueError("Frozen train-only target standard deviation must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.schedule.minimum_lr > self.learning_rate:
            raise ValueError("Invalid learning-rate or gradient-clipping bounds")
        if self.selection not in {"best-development-live", "best-development-ema", "final-live", "final-ema"}:
            raise ValueError("Unsupported selection policy")
        if self.dataset.development != self.selection.startswith("best-development"):
            raise ValueError("Dataset role selection differs from the weight-selection policy")
        if self.dataset.development != (self.development_rows > 0):
            raise ValueError("Development row declaration differs from selected roles")
        if self.ema_decay is not None and (type(self.ema_decay) not in (float, int) or self.ema_decay not in (.999, .9999)):
            raise ValueError("Unsupported EMA decay")
        if self.selection.endswith("ema") and self.ema_decay is None:
            raise ValueError("EMA weight selection requires EMA state")
        if self.sampler not in {"python-epoch-shuffle", "torch-pass-randperm"}:
            raise ValueError("Unsupported sampler policy")
        if type(self.drop_last) is not bool:
            raise ValueError("Tail policy must be an exact boolean")
        if self.torch_version not in {"2.7.1", "2.11.0"}:
            raise ValueError("Unsupported execution runtime pin")
        plan = self.loop_plan
        if self.dataset.development and self.optimizer_steps % plan.steps_per_pass:
            raise ValueError("Development selection requires a complete terminal pass")
        maximum_steps = self.schedule.total_units * (plan.steps_per_pass if self.schedule.epoch_based else 1)
        if self.optimizer_steps > maximum_steps:
            raise ValueError("Optimizer budget exceeds the frozen schedule")

    @property
    def loop_plan(self):
        return LoopPlan(self.train_rows, self.batch_size, self.optimizer_steps,
            self.checkpoint_every, self.finite_check_every,
            checkpoint_at_pass=self.dataset.development, drop_last=self.drop_last)

    @property
    def execution_policy(self):
        return {"name": "shared-family-accelerated", "family": self.family,
            "cpu_layout": True, "precision": "fp32", "tf32": False,
            "optimizer": "native-adamw" if self.family == "neural_atom_k1" else "foreach-adamw",
            "adamw_betas": list(ADAMW_BETAS), "adamw_epsilon": ADAMW_EPSILON,
            "ema": "bound-foreach" if self.ema_decay is not None else None,
            "torch_version": self.torch_version}

    def to_dict(self):
        return {"format": RECIPE_FORMAT, **asdict(self), "execution_policy": self.execution_policy}

    @property
    def identity(self):
        return canonical_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, record):
        expected = {field.name for field in fields(cls)} | {"format", "execution_policy"}
        if type(record) is not dict or set(record) != expected or record["format"] != RECIPE_FORMAT:
            raise ValueError("Missing or unknown shared training recipe fields")
        values = {field.name: record[field.name] for field in fields(cls)}
        if type(values["dataset"]) is not dict or type(values["schedule"]) is not dict:
            raise ValueError("Expected dataset/schedule objects")
        try:
            values["dataset"] = DatasetSelection(**values["dataset"])
            values["schedule"] = ScheduleSpec(**values["schedule"])
        except TypeError as exc:
            raise ValueError("Unknown dataset/schedule fields") from exc
        recipe = cls(**values)
        if canonical_fingerprint(record["execution_policy"]) != canonical_fingerprint(recipe.execution_policy):
            raise ValueError("Execution policy differs from the accelerated family default")
        return recipe
