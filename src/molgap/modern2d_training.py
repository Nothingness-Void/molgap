"""Explicit formal Modern2D admission; the ordinary family Session owns training.

Model versions remain unchanged. These interfaces bind a separately authorized
100K run to its exact source, recipe, accepted roles and allocation.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

from .screen_policy import canonical_fingerprint


PROFILE = "modern2d-formal-session-v1"
INTERFACES = (("ppgt_small", "3"), ("esa_small", "4"))
MODEL_INTERFACES = (("ppgt_small", "2"), ("esa_small", "3"))
AUTHORITY_FORMAT = "molgap-modern2d-formal-training-authority-v1"
FORMAL_STEPS = 60 * (100000 // 128)
_ISSUER = object()


@dataclass(frozen=True)
class Modern2DFormalTrainingProfile:
    purpose: str = "formal-100k-common-dev"
    phase: str = "train"
    horizon_passes: int = 60
    horizon_steps: int = FORMAL_STEPS

    def __post_init__(self):
        if self.configured:
            if (self.phase not in {"train", "native-preflight"}
                    or type(self.horizon_passes) is not int or self.horizon_passes < 1
                    or type(self.horizon_steps) is not int or self.horizon_steps < 1):
                raise ValueError("Configured training requires a positive explicit horizon")
            return
        if (self.purpose != "formal-100k-common-dev" or self.phase not in {"train", "native-preflight"}
                or type(self.horizon_passes) is not int or self.horizon_passes != 60
                or type(self.horizon_steps) is not int or self.horizon_steps != FORMAL_STEPS):
            raise ValueError("Formal Modern2D profile fixes a sixty-pass 46860-step horizon")

    @property
    def configured(self):
        return self.purpose == "configured-screen"

    def validate(self, recipe):
        synthetic = recipe.dataset.manifest.startswith("synthetic/")
        probe = self.phase == "native-preflight"
        if self.configured:
            if (recipe.interface not in INTERFACES or recipe.mode not in {"reference", "composed"} or recipe.seed != 42
                    or recipe.batch_size != 128 or not recipe.drop_last
                    or recipe.qualification is not None or recipe.input_sidecar is not None
                    or recipe.k1_execution is not None or recipe.ema_decay is not None or recipe.clean_bn is not None
                    or recipe.schedule.epoch_based or recipe.schedule.total_units != self.horizon_steps
                    or recipe.schedule.reference_epochs != self.horizon_passes):
                raise ValueError("Configured profile differs from the model recipe or explicit schedule")
            if synthetic:
                if recipe.train_rows not in (128, 256) or recipe.development_rows > 128 or recipe.optimizer_steps > 4:
                    raise ValueError("Configured CPU fixtures retain bounded synthetic data and four updates")
            elif (recipe.train_rows not in (100000, 500000)
                    or self.horizon_steps != self.horizon_passes * (recipe.train_rows // recipe.batch_size)):
                raise ValueError("Configured horizon differs from the selected training scale")
            if probe:
                if (recipe.optimizer_steps != 2 or recipe.dataset.development or recipe.development_rows
                        or recipe.selection != "final-live" or recipe.checkpoint_every != 1):
                    raise ValueError("Configured preflight requires two train-only updates")
            elif (recipe.optimizer_steps != self.horizon_steps
                    or recipe.selection not in {"best-development-live", "final-live"}
                    or (not synthetic and recipe.dataset.development and recipe.development_rows != 50000)):
                raise ValueError("Configured endpoint or selected development role differs")
            return
        if (recipe.interface not in INTERFACES or recipe.mode != "reference" or recipe.seed != 42
                or recipe.batch_size != 128 or not recipe.drop_last or recipe.ema_decay is not None
                or recipe.clean_bn is not None or recipe.addons or recipe.input_sidecar is not None
                or recipe.k1_execution is not None or recipe.qualification is not None
                or recipe.schedule.kind not in {"cosine_step", "four_stage_warmup_cosine_step"}
                or recipe.schedule.total_units != FORMAL_STEPS or recipe.schedule.reference_epochs != 60
                or recipe.schedule.warmup_units != (4 if recipe.schedule.kind == "four_stage_warmup_cosine_step" else 0)
                or recipe.checkpoint_every > 100000 // 128):
            raise ValueError("Formal Modern2D requires seed42 BS128 FP32/noEMA and a frozen 46860-step cosine schedule")
        if synthetic:
            if (recipe.train_rows not in (128, 256) or not 0 <= recipe.development_rows <= 128
                    or not 1 <= recipe.optimizer_steps <= 4):
                raise ValueError("Formal synthetic verification stays within 256 train/128 Dev rows and four steps")
        elif recipe.train_rows != 100000:
            raise ValueError("Formal authority releases train100K only")
        if probe:
            if (recipe.optimizer_steps != 2 or recipe.checkpoint_every != 1
                    or recipe.dataset.development or recipe.development_rows != 0
                    or recipe.selection != "final-live"):
                raise ValueError("Formal native preflight requires exactly two train-only scratch updates")
        elif (not recipe.dataset.development or recipe.selection != "best-development-live"
                or (not synthetic and (recipe.optimizer_steps != FORMAL_STEPS or recipe.development_rows != 50000))
                or (synthetic and recipe.development_rows < 1)):
            raise ValueError("Formal training requires the full horizon and CommonDev live-weight selection")


@dataclass(frozen=True)
class Modern2DFormalTrainingContext:
    run_context: object
    recipe_identity: str
    authority_sha256: str
    train_permit: object
    development_permit: object
    train_feature_binding_sha256: str
    development_feature_binding_sha256: str
    max_wall_seconds: int = 16200
    allocated_accelerators: tuple[str, ...] = ("NvidiaTeslaT4", "NvidiaTeslaT4")
    configured: bool = False

    def __post_init__(self):
        from .experiment_family_workflow import RunContext, _digest
        from .data.indexed_inputs import AcceptedIndexedInputPermit
        if type(self.run_context) is not RunContext:
            raise TypeError("Formal training requires a source/package-bound RunContext")
        for name in ("recipe_identity", "authority_sha256", "train_feature_binding_sha256",
                     "development_feature_binding_sha256"):
            _digest(getattr(self, name), name)
        if (type(self.train_permit) is not AcceptedIndexedInputPermit
                or type(self.development_permit) is not AcceptedIndexedInputPermit
                or self.train_permit.role not in (("train_100k", "train_500k") if self.configured else ("train_100k",)) or self.development_permit.role != "common_dev"
                or self.train_permit.version_identity != self.development_permit.version_identity
                or self.train_permit.acceptance_sha256 != self.development_permit.acceptance_sha256):
            raise ValueError("ROLE_BLOCKED: formal training requires independently bound train100K/CommonDev permits")
        if ((self.run_context.family_name, self.run_context.family_version) not in INTERFACES
                or self.run_context.platform != "kaggle" or type(self.max_wall_seconds) is not int
                or type(self.configured) is not bool
                or not 1 <= self.max_wall_seconds <= (43200 if self.configured else 16200)
                or type(self.allocated_accelerators) is not tuple
                or self.allocated_accelerators != ("NvidiaTeslaT4", "NvidiaTeslaT4")):
            raise ValueError("Formal allocation is exactly T4x2 within 4.5h/aggregate9 T4-hours")

    @property
    def identity(self):
        return canonical_fingerprint(self.to_dict())

    def to_dict(self):
        values = asdict(self)
        if not self.configured:
            values.pop("configured")
        values["allocated_accelerators"] = list(self.allocated_accelerators)
        return values

    @classmethod
    def from_dict(cls, record):
        from .experiment_family_workflow import RunContext
        from .data.indexed_inputs import AcceptedIndexedInputPermit
        if type(record) is not dict:
            raise TypeError("Formal context must be an object")
        values = dict(record)
        values["run_context"] = RunContext(**values["run_context"])
        for name in ("train_permit", "development_permit"):
            values[name] = AcceptedIndexedInputPermit(**values[name])
        if type(values.get("allocated_accelerators")) is not list:
            raise ValueError("Formal context requires an explicit allocation list")
        values["allocated_accelerators"] = tuple(values["allocated_accelerators"])
        return cls(**values)

    def validate_recipe(self, recipe):
        if type(recipe.formal_training) is not Modern2DFormalTrainingProfile:
            raise TypeError("Explicit formal training profile required")
        recipe.formal_training.validate(recipe)
        if recipe.formal_training.configured != self.configured:
            raise ValueError("Configured screen requires its separately bound native context")
        if (recipe.formal_training.phase != "train" or recipe.dataset.manifest.startswith("synthetic/")
                or recipe.identity != self.recipe_identity
                or recipe.interface != (self.run_context.family_name, self.run_context.family_version)):
            raise ValueError("Formal context requires the exact real-data training recipe")
        if recipe.family == "esa_small" and recipe.model_config.tensor_device == "cpu":
            raise ValueError("A CPU-only ESA configuration cannot authorize native CUDA training")

    def validate_before_payload(self, *, recipe, context, spec, authority_path, input_root,
                                training_only=False):
        """Read only small pinned authority/input metadata before graph payload IO."""
        from .experiment_family_workflow import _json
        from .family_training_data import resolve_pinned_manifest
        from .family_training_runner import normalized_file_sha
        self.validate_recipe(recipe)
        if context != self.run_context or spec.identity != context.spec_identity:
            raise ValueError("Formal source/package/recipe/account/run binding differs")
        if authority_path is None:
            raise ValueError("Formal training requires bound human authority before payload loading")
        path = Path(authority_path)
        if path.stat().st_size > 1024**2 or normalized_file_sha(path) != self.authority_sha256:
            raise ValueError("Formal authority bytes changed or exceed the metadata ceiling")
        authority = _json(path)
        declaration = spec.to_dict()
        arms, platform = declaration["arms"], declaration["platform"]
        gpu = authority.get("gpu", {})
        if self.configured:
            from .modern2d_training_release import validate_configured_authority
            validate_configured_authority(authority, declaration)
            if authority.get("purpose") == "CONFIGURED_EXECUTION_PROFILE" and not training_only:
                raise ValueError("Execution-profile authority cannot open development or release formal training")
            horizon = {"horizon_passes": recipe.formal_training.horizon_passes,
                       "horizon_steps": recipe.formal_training.horizon_steps}
            if (authority.get("recipes", {}).get(context.arm_id) != horizon
                    or authority["run_reference"] != context.run_reference
                    or gpu["owner"] != context.account or gpu["wall_ceiling_seconds"] != self.max_wall_seconds
                    or authority["roles"][0] != self.train_permit.role
                    or recipe.train_rows != (100000 if self.train_permit.role == "train_100k" else 500000)):
                raise ValueError("Configured runtime differs from frozen authority")
            self._validate_input_binding(recipe, input_root)
            return
        if (authority.get("format") != AUTHORITY_FORMAT or authority.get("owner") != "server"
                or authority.get("user_authorized") is not True or authority.get("purpose") != "FORMAL_100K_TRAINING"
                or authority.get("logical_run_id") != context.logical_run_id
                or authority.get("run_reference") != context.run_reference
                or authority.get("roles") != ["train_100k", "common_dev"]
                or authority.get("interfaces") != [list(item) for item in INTERFACES]
                or authority.get("seeds") != [42] or authority.get("physical_batch_size") != 128
                or authority.get("horizon_passes") != 60 or authority.get("horizon_steps") != FORMAL_STEPS
                or authority.get("scientific_training_authorized") is not True
                or any(authority.get(key) is not False for key in (
                    "protected_role_access_authorized", "extra_seed_authorized", "automatic_successor_authorized",
                    "scale_500k_authorized", "full_training_authorized"))
                or platform["name"] != "kaggle" or platform["accelerator"] != "NvidiaTeslaT4"
                or platform["device_count"] != 2 or len(arms) != 2
                or {(a["family"]["name"], a["family"]["version"]) for a in arms} != set(INTERFACES)
                or authority.get("arms") != [a["arm_id"] for a in arms]
                or [(a["family"]["name"], a["family"]["version"]) for a in arms] != list(INTERFACES)
                or any(a["initialization"]["seed"] != 42 or a["addons"]
                    or {r["role"] for r in a["data"]["roles"]} != {"train", "development"} for a in arms)
                or gpu.get("owner") != context.account or gpu.get("accelerator") != "NvidiaTeslaT4"
                or gpu.get("allocated_devices") != 2 or gpu.get("wall_ceiling_seconds") != self.max_wall_seconds
                or gpu.get("allocated_t4_hours_ceiling") != 9):
            raise ValueError("Formal authority does not cover the exact two-arm train100K/CommonDev allocation")
        self._validate_input_binding(recipe, input_root)

    def _validate_input_binding(self, recipe, input_root):
        from .family_training_data import resolve_pinned_manifest
        from .family_training_runner import normalized_file_sha
        from .experiment_family_workflow import _json
        binding_path = resolve_pinned_manifest(Path(input_root).absolute(), recipe.dataset)
        if binding_path.stat().st_size > 1024**2 or normalized_file_sha(binding_path) != recipe.dataset.manifest_sha256:
            raise ValueError("Formal indexed binding changed or exceeds the metadata ceiling")
        binding = _json(binding_path)
        if (set(binding) != {"format", "data_root", "scope", "train", "development"}
                or binding["format"] != "molgap-indexed-training-input-v1"
                or binding["scope"] != "ACCEPTED_INDEXED_INPUT_V1"):
            raise ValueError("Formal training requires an accepted indexed input binding")
        for role, permit in (("train", self.train_permit), ("development", self.development_permit)):
            item = binding[role]
            if (type(item) is not dict or set(item) != {"permit", "authority", "role_history"}
                    or item["permit"] != asdict(permit)):
                raise ValueError("ROLE_BLOCKED: formal role differs from its independently frozen permit")
            permit.verify(item["authority"], item["role_history"])

    def validate_loaded_data(self, recipe, data, *, training_only):
        from .data.indexed_inputs import IndexedRoleDataset
        from .modern2d_feature_binding import build_indexed_feature_binding
        if data.synthetic or not data.indexed or data.train_rows != (recipe.train_rows if self.configured else 100000):
            raise ValueError("Formal native admission requires actual accepted indexed100K")
        roles = [("train", self.train_permit, self.train_feature_binding_sha256)]
        if not training_only:
            roles.append(("development", self.development_permit, self.development_feature_binding_sha256))
        for role, permit, digest in roles:
            dataset = getattr(data, role)
            if type(dataset) is not IndexedRoleDataset or dataset.permit != permit:
                raise ValueError("Formal decoded dataset differs from its frozen accepted role")
            binding = build_indexed_feature_binding(dataset, family=recipe.interface, configuration=recipe.model_config,
                                                    **({"feature_only": True} if self.configured else {}))
            if canonical_fingerprint(binding) != digest:
                raise ValueError("Formal feature/source/config/history binding differs")


class _FormalTrainingAdmission:
    def __init__(self, recipe, *, context, synthetic, issuer):
        if issuer is not _ISSUER:
            raise TypeError("Formal admission must be issued after source and role validation")
        from .family_training_runner import _probe_recipe
        if not synthetic:
            if type(context) is not Modern2DFormalTrainingContext:
                raise TypeError("Exact formal training context required")
            context.validate_recipe(recipe)
        self.recipe_identities = (recipe.identity, _probe_recipe(recipe).identity)
        self.synthetic = synthetic

    def validate(self, recipe, *, device):
        if recipe.identity not in self.recipe_identities or ((str(device).split(":")[0] == "cpu") != self.synthetic):
            raise ValueError("Formal admission recipe/device differs from the bound execution scope")


def _admit_formal_training(context, recipe):
    return _FormalTrainingAdmission(recipe, context=context, synthetic=False, issuer=_ISSUER)


def admit_synthetic_formal_training(recipe, data, *, authority):
    """Generated CPU fixtures only; cannot issue native training admission."""
    from .family_training_data import TrainingData
    if (not recipe.formal_interface or recipe.formal_training.phase != "train"
            or not recipe.dataset.manifest.startswith("synthetic/") or type(data) is not TrainingData
            or data.synthetic is not True or type(data.train) not in (list, tuple)
            or type(data.development) not in (list, tuple)
            or data.train_rows != len(data.train) or data.development_rows != len(data.development)
            or data.train_rows != recipe.train_rows or data.development_rows != recipe.development_rows
            or data.manifest_sha256 != recipe.dataset.manifest_sha256
            or type(authority) is not dict or authority.get("user_authorized") is not True
            or authority != {"scope": "CPU_SYNTHETIC_ONLY", "user_authorized": True,
                             "recipe_identity": recipe.identity}):
        raise ValueError("Synthetic formal admission requires exact bounded generated data and explicit fixture authority")
    recipe.formal_training.validate(recipe)
    return _FormalTrainingAdmission(recipe, context=None, synthetic=True, issuer=_ISSUER)
