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
from .k1_execution_backend import K1ExecutionConfig
from .family_training_state import CalibrationPolicy


RECIPE_FORMAT = "molgap-family-training-recipe-v1"
ADAMW_BETAS = (.9, .999)
ADAMW_EPSILON = 1e-8
from .experiment_execution import TRAINING_ADAPTERS, SESSION_INTERFACES
from .family_training_policy import mode_policy

MODES = {family: tuple(dict.fromkeys(("reference", *(entry.mode for key, adapter in
    TRAINING_ADAPTERS.items() if key[0] == family for entry in adapter.addons))))
    for family, version in SESSION_INTERFACES.values()}


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
    input_sidecar: DatasetSelection | None = None
    k1_execution: K1ExecutionConfig | None = None
    addons: tuple[dict, ...] = ()
    clean_bn: CalibrationPolicy | None = None
    model_config: object | None = None
    family_version: str | None = None
    qualification: object | None = None
    formal_training: object | None = None

    def __post_init__(self):
        from .experiment_spec import _digest
        if self.family not in MODES or self.mode not in MODES[self.family]:
            raise ValueError("Unknown executable family/addon mode")
        if self.cpu_synthetic_only or self.integration_preview or self.formal_interface:
            from copy import deepcopy
            from importlib import import_module
            adapter = TRAINING_ADAPTERS[self.interface]
            config_type = import_module(adapter.model_factory.split(":")[0]).CONFIG_TYPE
            if (self.integration_preview or self.formal_interface) and type(self.dataset) is not DatasetSelection:
                raise TypeError("Integration preview requires the typed dataset selection")
            bounded_fixture = self.cpu_synthetic_only or self.dataset.manifest.startswith("synthetic/")
            configurable = self.formal_interface and getattr(self.formal_training, "configured", False)
            if (type(self.model_config) not in (dict, config_type)
                    or (not configurable and (self.mode != "reference" or self.addons))
                    or self.ema_decay is not None or self.clean_bn is not None or not self.drop_last
                    or (self.cpu_synthetic_only and not self.dataset.manifest.startswith("synthetic/"))
                    or (bounded_fixture and (self.train_rows > 256 or self.development_rows > 128 or self.optimizer_steps > 4))):
                if self.integration_preview:
                    raise ValueError("Modern2D integration preview requires explicit reference configuration; synthetic fixtures are bounded to 256 train/128 development rows, four steps")
                raise ValueError("Modern2D profile is bounded CPU synthetic only")
            if self.integration_preview and not bounded_fixture and self.qualification is None and (
                    self.train_rows not in (100000, 500000) or self.development_rows != 50000
                    or not self.dataset.development):
                raise ValueError("Real-input integration preview requires accepted indexed 100K/500K train and 50K development declarations")
            values = deepcopy(self.model_config) if type(self.model_config) is dict else asdict(self.model_config)
            if "layer_types" in values:
                values["layer_types"] = tuple(values["layer_types"])
            configuration = config_type(**values)
            object.__setattr__(self, "model_config", configuration)
        elif self.model_config is not None:
            raise ValueError("Existing families cannot reinterpret Modern2D configuration")
        if type(self.addons) is not tuple or (self.mode == "composed") != bool(self.addons):
            raise ValueError("Ordered slot addons require the explicit composed mode")
        if self.addons:
            from .shared_model_adapter import validate_family_addons
            validate_family_addons(self.family, self.addons, interface=self.interface)
        if self.k1_execution is not None and (self.family != "neural_atom_k1" or
                type(self.k1_execution) is not K1ExecutionConfig):
            raise ValueError("K1 execution options require the typed K1 family")
        _integer(self.seed, "seed", 0)
        if self.seed > 2**32 - 1:
            raise ValueError("Seed exceeds the supported RNG range")
        _digest(self.initialization_sha256, "initialization_sha256")
        if type(self.dataset) is not DatasetSelection or type(self.schedule) is not ScheduleSpec:
            raise TypeError("Typed dataset and schedule are required")
        cycle = mode_policy(self.family, self.mode).input_sidecar_required
        if cycle != (self.input_sidecar is not None):
            raise ValueError("Only the common-cycle mode requires a pinned input sidecar")
        if self.input_sidecar is not None and (type(self.input_sidecar) is not DatasetSelection or
                self.input_sidecar.development != self.dataset.development):
            raise ValueError("Sidecar roles differ from selected fixed-data roles")
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
        if self.selection not in {"best-development-live", "best-development-ema", "final-live", "final-ema",
                                  "best-development-clean-ema", "final-clean-ema"}:
            raise ValueError("Unsupported selection policy")
        if self.dataset.development != self.selection.startswith("best-development"):
            raise ValueError("Dataset role selection differs from the weight-selection policy")
        if self.dataset.development != (self.development_rows > 0):
            raise ValueError("Development row declaration differs from selected roles")
        if self.ema_decay is not None and (type(self.ema_decay) not in (float, int) or self.ema_decay not in (.999, .9999)):
            raise ValueError("Unsupported EMA decay")
        if self.selection.endswith("ema") and self.ema_decay is None:
            raise ValueError("EMA weight selection requires EMA state")
        if (self.clean_bn is not None) != self.selection.endswith("clean-ema"):
            raise ValueError("Clean selection requires an explicit calibration policy")
        if self.clean_bn is not None and (type(self.clean_bn) is not CalibrationPolicy or
                self.clean_bn.input_identity != self.dataset.manifest_sha256 or
                self.clean_bn.batch_size != self.batch_size or
                self.clean_bn.batches * self.batch_size > self.train_rows):
            raise ValueError("Calibration differs from selected training data/batch boundary")
        if self.sampler not in {"python-epoch-shuffle", "torch-pass-randperm"}:
            raise ValueError("Unsupported sampler policy")
        if type(self.drop_last) is not bool:
            raise ValueError("Tail policy must be an exact boolean")
        if self.torch_version not in {"2.7.1", "2.11.0"}:
            raise ValueError("Unsupported execution runtime pin")
        plan = self.loop_plan
        evaluation_interval = getattr(self.formal_training, "evaluation_every_steps", None)
        if self.dataset.development and self.optimizer_steps % plan.steps_per_pass and not (
                getattr(self.formal_training, "configured", False) and evaluation_interval is not None
                and self.optimizer_steps % evaluation_interval == 0):
            raise ValueError("Development selection requires a complete terminal pass")
        maximum_steps = self.schedule.total_units * (plan.steps_per_pass if self.schedule.epoch_based else 1)
        if self.optimizer_steps > maximum_steps:
            raise ValueError("Optimizer budget exceeds the frozen schedule")
        if self.qualification is not None:
            from .modern2d_qualification import QualificationProbe
            if type(self.qualification) is not QualificationProbe:
                raise TypeError("Expected explicit Modern2D QualificationProbe")
            self.qualification.validate(self)
        if self.formal_interface:
            from .modern2d_training import Modern2DFormalTrainingProfile
            if type(self.formal_training) is not Modern2DFormalTrainingProfile:
                raise TypeError("Formal interface requires an explicit Modern2DFormalTrainingProfile")
            self.formal_training.validate(self)
        elif self.formal_training is not None:
            raise ValueError("Formal profile cannot admit a preview or historical interface")

    @property
    def cpu_synthetic_only(self):
        return TRAINING_ADAPTERS[self.interface].cpu_synthetic_only

    @property
    def integration_preview(self):
        return TRAINING_ADAPTERS[self.interface].integration_preview

    @property
    def formal_interface(self):
        return TRAINING_ADAPTERS[self.interface].formal_training

    @property
    def modern2d_model(self):
        return self.cpu_synthetic_only or self.integration_preview or self.formal_interface

    @property
    def interface(self):
        if self.family_version is None:
            return SESSION_INTERFACES[self.family]
        key = (self.family,self.family_version)
        if (type(self.family_version) is not str or key not in TRAINING_ADAPTERS or
                not (TRAINING_ADAPTERS[key].cpu_synthetic_only or TRAINING_ADAPTERS[key].integration_preview
                     or TRAINING_ADAPTERS[key].formal_training)):
            raise ValueError("Explicit family version is restricted to reviewed CPU synthetic or integration preview interfaces")
        return key

    @property
    def selection_weights(self):
        return "clean-ema" if self.selection.endswith("clean-ema") else self.selection.rsplit("-", 1)[1]

    @property
    def loop_plan(self):
        return LoopPlan(self.train_rows, self.batch_size, self.optimizer_steps,
            self.checkpoint_every, self.finite_check_every,
            checkpoint_at_pass=self.dataset.development, drop_last=self.drop_last)

    @property
    def execution_policy(self):
        policy = {"name": "shared-family-accelerated", "family": self.family,
            "cpu_layout": True, "precision": "fp32", "tf32": False,
            "optimizer": "native-adamw" if self.family == "neural_atom_k1" else "foreach-adamw",
            "adamw_betas": list(ADAMW_BETAS), "adamw_epsilon": ADAMW_EPSILON,
            "ema": "bound-foreach" if self.ema_decay is not None else None,
            "torch_version": self.torch_version}
        if self.k1_execution is not None:
            policy["k1_execution"] = self.k1_execution.to_dict()
        if self.cpu_synthetic_only:
            policy.update(name="modern2d-cpu-v1", scope="CPU_SYNTHETIC_ONLY",
                          cpu_layout=False, optimizer="native-adamw")
        if self.integration_preview:
            from .modern2d_feature_binding import indexed_transform_identity
            policy.update(name="modern2d-session-preview-v1", scope="INTEGRATION_PREVIEW_ONLY",
                          cpu_layout=False, device="cpu", native_qualification="INTEGRATION_PENDING",
                          feature_transform_source_sha256=indexed_transform_identity(self.interface))
        if self.qualification is not None:
            policy.update(scope="QUALIFICATION_ONLY", device="cpu_or_admitted_cuda",
                          native_qualification="NOT_EVALUATED")
        if self.formal_interface:
            from .modern2d_feature_binding import indexed_transform_identity
            policy.update(name="modern2d-formal-session-v1", scope="FORMAL_100K_TRAINING",
                          cpu_layout=False, device="admitted_cuda_or_bounded_synthetic_cpu",
                          native_qualification="requires_exact_formal_runtime_certificate",
                          feature_transform_source_sha256=indexed_transform_identity(self.interface,
                              feature_only=self.formal_training.configured))
            if self.formal_training.configured:
                policy["scope"] = "CONFIGURED_SCREEN"
        return policy

    def to_dict(self):
        values = asdict(self)
        if values.get("formal_training") and values["formal_training"].get("evaluation_every_steps") is None:
            values["formal_training"].pop("evaluation_every_steps", None)
        if (values.get("formal_training") or {}).get("execution_mode") == "original":
            # Preserve historical serialized recipes while new execution modes
            # remain explicit in newly frozen contracts.
            values["formal_training"].pop("execution_mode")
        if self.family_version is None:
            values.pop("family_version")
        if self.qualification is None:
            values.pop("qualification")
        if self.formal_training is None:
            values.pop("formal_training")
        if self.model_config is None:
            values.pop("model_config")
        if self.input_sidecar is None:
            values.pop("input_sidecar")
        if self.k1_execution is None:
            values.pop("k1_execution")
        if not self.addons:
            values.pop("addons")
        else:
            values["addons"] = list(values["addons"])
        if self.clean_bn is None:
            values.pop("clean_bn")
        else:
            values["clean_bn"] = self.clean_bn.to_dict()
        return {"format": RECIPE_FORMAT, **values, "execution_policy": self.execution_policy}

    @property
    def identity(self):
        return canonical_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, record):
        expected = {field.name for field in fields(cls)} | {"format", "execution_policy"}
        if type(record) is dict and "input_sidecar" not in record:
            expected.remove("input_sidecar")
        if type(record) is dict and "k1_execution" not in record:
            expected.remove("k1_execution")
        for name in ("addons", "clean_bn", "model_config", "family_version", "qualification", "formal_training"):
            if type(record) is dict and name not in record:
                expected.remove(name)
        if type(record) is not dict or set(record) != expected or record["format"] != RECIPE_FORMAT:
            raise ValueError("Missing or unknown shared training recipe fields")
        values = {field.name: record[field.name] for field in fields(cls) if field.name in record}
        if type(values["dataset"]) is not dict or type(values["schedule"]) is not dict:
            raise ValueError("Expected dataset/schedule objects")
        try:
            values["dataset"] = DatasetSelection(**values["dataset"])
            values["schedule"] = ScheduleSpec(**values["schedule"])
            if "formal_training" in values:
                from .modern2d_training import Modern2DFormalTrainingProfile
                if type(values["formal_training"]) is not dict:
                    raise ValueError("Formal profile must be an object")
                values["formal_training"] = Modern2DFormalTrainingProfile(**values["formal_training"])
            if "qualification" in values:
                from .modern2d_qualification import QualificationProbe
                if type(values["qualification"]) is not dict:
                    raise ValueError("Qualification probe must be an object")
                values["qualification"] = QualificationProbe(**values["qualification"])
            if "input_sidecar" in values:
                values["input_sidecar"] = DatasetSelection(**values["input_sidecar"])
            if "k1_execution" in values:
                values["k1_execution"] = K1ExecutionConfig.from_dict(values["k1_execution"])
            if "addons" in values:
                if type(values["addons"]) is not list:
                    raise ValueError("Ordered addons must be a list")
                values["addons"] = tuple(values["addons"])
            if "clean_bn" in values:
                values["clean_bn"] = CalibrationPolicy.from_dict(values["clean_bn"])
        except TypeError as exc:
            raise ValueError("Unknown dataset/schedule fields") from exc
        recipe = cls(**values)
        if canonical_fingerprint(record["execution_policy"]) != canonical_fingerprint(recipe.execution_policy):
            raise ValueError("Execution policy differs from the accelerated family default")
        return recipe
