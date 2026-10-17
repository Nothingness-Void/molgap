"""Bounded execution admission for Modern2D; never a scientific release.

The platform owns isolation, the allocation deadline and durable retrieval.
This adapter owns only independently frozen bindings and the scratch ceiling;
the existing preflight/Session owns every update and serialized resume.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import time

from .screen_policy import canonical_fingerprint


INTERFACES = (("ppgt_small", "2"), ("esa_small", "3"))
CERTIFICATE_FORMAT = "molgap-modern2d-qualification-certificate-v1"
_ISSUER = object()


@dataclass(frozen=True)
class QualificationProbe:
    purpose: str = "numerical-resume-resources-only"
    steps_per_path: int = 2
    scratch_updates_per_model: int = 6

    def __post_init__(self):
        if (self.purpose != "numerical-resume-resources-only" or
                type(self.steps_per_path) is not int or self.steps_per_path != 2 or
                type(self.scratch_updates_per_model) is not int or self.scratch_updates_per_model != 6):
            raise ValueError("Qualification permits exactly three two-update scratch paths")

    def validate(self, recipe):
        synthetic = recipe.dataset.manifest.startswith("synthetic/")
        if (recipe.interface not in INTERFACES or recipe.seed != 42 or recipe.batch_size != 128 or
                recipe.optimizer_steps != self.steps_per_path or recipe.checkpoint_every != 1 or
                recipe.finite_check_every != 1 or recipe.dataset.development or recipe.development_rows != 0 or
                recipe.selection != "final-live" or recipe.mode != "reference" or not recipe.drop_last or
                recipe.ema_decay is not None or recipe.clean_bn is not None or recipe.addons or
                recipe.input_sidecar is not None or recipe.k1_execution is not None or
                (synthetic and recipe.train_rows not in (128, 256)) or
                (not synthetic and recipe.train_rows != 100000)):
            raise ValueError("Qualification requires seed42 BS128 train-only 100K, two scratch steps; synthetic CPU fixtures stay separate")


@dataclass(frozen=True)
class Modern2DQualificationContext:
    """Externally reviewed pins; never derive these from the mounted input.

    run_context is the exact RunContext including source/package/recipe/account.
    Both accepted role permits are frozen even though only train is opened.
    authority_sha256 identifies the human-approved bounded request record.
    """
    run_context: object
    recipe_identity: str
    authority_sha256: str
    train_permit: object
    development_permit: object
    train_feature_binding_sha256: str
    max_wall_seconds: int = 1800
    allocated_accelerators: tuple[str, ...] = ("NvidiaTeslaT4", "NvidiaTeslaT4")
    purpose: str = "numerical-resume-resources-only"

    def __post_init__(self):
        from .experiment_family_workflow import RunContext, _digest
        from .data.indexed_inputs import AcceptedIndexedInputPermit
        if type(self.run_context) is not RunContext:
            raise TypeError("Qualification requires an exact source-bound RunContext")
        for name in ("recipe_identity", "authority_sha256", "train_feature_binding_sha256"):
            _digest(getattr(self, name), name)
        if (type(self.train_permit) is not AcceptedIndexedInputPermit or
                type(self.development_permit) is not AcceptedIndexedInputPermit or
                self.train_permit.role != "train_100k" or self.development_permit.role != "common_dev" or
                self.train_permit.version_identity != self.development_permit.version_identity or
                self.train_permit.acceptance_sha256 != self.development_permit.acceptance_sha256):
            raise ValueError("ROLE_BLOCKED: exact accepted 100K/Common-Dev permits required")
        if (self.purpose != "numerical-resume-resources-only" or
                type(self.max_wall_seconds) is not int or not 1 <= self.max_wall_seconds <= 1800 or
                type(self.allocated_accelerators) is not tuple or
                self.allocated_accelerators != ("NvidiaTeslaT4", "NvidiaTeslaT4")):
            raise ValueError("Qualification allocation is bounded to thirty minutes of T4x2")
        if (self.run_context.platform != "kaggle" or
                (self.run_context.family_name, self.run_context.family_version) not in INTERFACES):
            raise ValueError("Only Kaggle PPGT/2 and ESA/3 qualification is admitted")

    @property
    def identity(self):
        return canonical_fingerprint(self.to_dict())

    def to_dict(self):
        values = asdict(self)
        values["allocated_accelerators"] = list(self.allocated_accelerators)
        return values

    @classmethod
    def from_dict(cls, record):
        from .experiment_family_workflow import RunContext
        from .data.indexed_inputs import AcceptedIndexedInputPermit
        if type(record) is not dict:
            raise TypeError("Qualification context must be an object")
        values = dict(record)
        values["run_context"] = RunContext(**values["run_context"])
        for name in ("train_permit", "development_permit"):
            values[name] = AcceptedIndexedInputPermit(**values[name])
        if type(values.get("allocated_accelerators")) is not list:
            raise ValueError("Qualification context requires an explicit allocation list")
        values["allocated_accelerators"] = tuple(values["allocated_accelerators"])
        return cls(**values)

    def validate_recipe(self, recipe):
        if type(recipe.qualification) is not QualificationProbe:
            raise ValueError("Explicit qualification-only recipe required")
        recipe.qualification.validate(recipe)
        if recipe.dataset.manifest.startswith("synthetic/") or recipe.identity != self.recipe_identity:
            raise ValueError("Native qualification requires the exact real-input recipe")
        if recipe.interface != (self.run_context.family_name, self.run_context.family_version):
            raise ValueError("Qualification interface differs from reviewed context")
        if recipe.family == "esa_small" and recipe.model_config.tensor_device == "cpu":
            raise ValueError("CPU-only ESA configuration cannot qualify CUDA")


class _QualificationAdmission:
    """Process-local, expiring update capability; not deserialized from JSON."""
    def __init__(self, context, recipe, *, began, issuer):
        if issuer is not _ISSUER or type(context) is not Modern2DQualificationContext:
            raise TypeError("Qualification admission must be issued by the bound adapter")
        context.validate_recipe(recipe)
        self.context, self.recipe_identity = context, recipe.identity
        self.deadline = began + context.max_wall_seconds
        self.updates = 0

    def validate(self, recipe):
        if recipe.identity != self.recipe_identity:
            raise ValueError("Qualification admission cannot authorize a different recipe")
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("Qualification wall budget exhausted; platform deadline remains mandatory")

    def before_update(self, recipe):
        self.validate(recipe)
        if self.updates >= recipe.qualification.scratch_updates_per_model:
            raise ValueError("Qualification scratch optimizer ceiling exhausted")
        self.updates += 1


def bootstrap_qualification(context, *, began, **options):
    """Validate scope, source and small role bindings before any graph payload."""
    from .experiment_family_workflow import RunContext, _artifact_path, _json
    from .family_training_runner import (
        read_recipe, validate_screen_recipe, _loaded_source, _assert_target_statistics,
        normalized_file_sha)
    from .family_training_data import TrainingData, load_training_data
    from .data.indexed_inputs import AcceptedIndexedInputPermit, IndexedRoleDataset
    from .modern2d_feature_binding import build_indexed_feature_binding
    from .gptrans_execution_tuning import ExecutionTuning
    from .training_reproducibility import (
        build_runtime_manifest, configure_fp32_determinism, validate_accelerator_request)
    from .v4_runtime import inspect_frozen_state_artifact, normalized_source_sha256
    if type(context) is not Modern2DQualificationContext:
        raise TypeError("Reviewed Modern2DQualificationContext required")
    required = {"spec", "package_dir", "expected_package_identity", "arm_id", "mode", "recipe_path",
                "initial_state_path", "input_root", "account", "run_reference", "authority_path"}
    if not required <= set(options) or set(options) - required - {"trajectory_id", "prospective_root"}:
        raise ValueError("Unknown or missing bounded qualification options")
    record, recipe = read_recipe(options["recipe_path"])
    context.validate_recipe(recipe)
    admission = _QualificationAdmission(context, recipe, began=began, issuer=_ISSUER)
    spec = options["spec"]
    declaration = spec.to_dict()
    platform = declaration["platform"]
    if (spec.identity != context.run_context.spec_identity or
            len(declaration["arms"]) != 2 or
            {tuple(a["family"][k] for k in ("name", "version")) for a in declaration["arms"]} != set(INTERFACES) or
            platform["name"] != "kaggle" or platform["accelerator"] != "NvidiaTeslaT4" or
            platform["device_count"] != 2):
        raise ValueError("Qualification requires exactly the reviewed two-model T4x2 Spec")
    if options.get("mode") != "reference":
        raise ValueError("Qualification supports reference interfaces only")
    authority_path = Path(options["authority_path"])
    if authority_path.stat().st_size > 1024**2 or normalized_file_sha(authority_path) != context.authority_sha256:
        raise ValueError("Qualification authority bytes changed or exceed the metadata ceiling")
    authority = _json(authority_path)
    gpu = authority.get("gpu", {})
    if (authority.get("format") != "molgap-modern2d-bounded-qualification-authority-v1" or
            authority.get("owner") != "server" or authority.get("user_authorized") is not True or
            authority.get("purpose") != "EXECUTION_QUALIFICATION_ONLY" or
            authority.get("roles") != ["train_100k", "common_dev"] or
            authority.get("interfaces") != [list(item) for item in INTERFACES] or
            authority.get("seeds") != [42] or authority.get("physical_batch_size") != 128 or
            authority.get("arms") != ["ppgt_author_v2", "esa_native_v3"] or
            {a["arm_id"] for a in declaration["arms"]} != set(authority["arms"]) or
            any(a["initialization"]["seed"] != 42 or a["addons"] or
                [r["role"] for r in a["data"]["roles"]] != ["train"] for a in declaration["arms"]) or
            any(authority.get(key) is not False for key in ("scientific_training_authorized",
                "protected_role_access_authorized", "extra_seed_authorized", "automatic_successor_authorized")) or
            gpu.get("platform") != "kaggle2" or gpu.get("owner") != context.run_context.account or
            gpu.get("accelerator") != "NvidiaTeslaT4" or gpu.get("allocated_devices") != 2 or
            gpu.get("wall_ceiling_seconds") != 1800 or gpu.get("allocated_t4_hours_ceiling") != 1):
        raise ValueError("Qualification authority does not cover this exact bounded two-arm allocation")
    if {a["arm_id"]: (a["family"]["name"], a["family"]["version"]) for a in declaration["arms"]} != dict(zip(authority["arms"], INTERFACES)):
        raise ValueError("Qualification arm/interface mapping differs from authority")
    tuning = ExecutionTuning(training_workers=0, development_workers=0, independent_models=2,
        total_cpu_budget=platform["cpu_cores"])
    validate_screen_recipe(spec, options["arm_id"], record)
    observed = RunContext.for_training(spec, options["package_dir"],
        expected_package_identity=options["expected_package_identity"], arm_id=options["arm_id"],
        account=options["account"], run_reference=options["run_reference"])
    if observed != context.run_context or normalized_source_sha256(Path(options["recipe_path"])) != observed.training_recipe_sha256:
        raise ValueError("Qualification source/package/recipe/account/run differs from reviewed context")
    binding_path = _artifact_path(Path(options["input_root"]).absolute(), recipe.dataset.manifest)
    if binding_path.stat().st_size > 1024**2 or normalized_file_sha(binding_path) != recipe.dataset.manifest_sha256:
        raise ValueError("Qualification input binding changed or exceeds its metadata ceiling")
    binding = _json(binding_path)
    if (set(binding) != {"format", "data_root", "scope", "train", "development"} or
            binding["format"] != "molgap-indexed-training-input-v1" or
            binding["scope"] != "ACCEPTED_INDEXED_INPUT_V1"):
        raise ValueError("Native qualification requires actual accepted indexed input")
    for role, permit in (("train", context.train_permit), ("development", context.development_permit)):
        item = binding[role]
        if (type(item) is not dict or set(item) != {"permit", "authority", "role_history"} or
                item["permit"] != asdict(permit)):
            raise ValueError("ROLE_BLOCKED: qualification role differs from independently frozen permit")
        permit.verify(item["authority"], item["role_history"])
    source_root = _loaded_source(options["package_dir"])
    admission.validate(recipe)
    initial = inspect_frozen_state_artifact(Path(options["initial_state_path"]),
        expected_state_sha256=recipe.initialization_sha256)
    data = load_training_data(Path(options["input_root"]), recipe.dataset)
    if (type(data) is not TrainingData or data.synthetic or not data.indexed or
            type(data.train) is not IndexedRoleDataset or type(data.train.permit) is not AcceptedIndexedInputPermit or
            data.train_rows != 100000 or len(data.train) != 100000 or data.development is not None or
            data.development_rows != 0 or data.train_source_start is not None or data.development_source_start is not None or
            data.manifest_sha256 != recipe.dataset.manifest_sha256 or
            type(data.train_source_indices) is not tuple or len(data.train_source_indices) != 100000 or
            data.train_source_indices != data.train.source_indices or data.train.permit != context.train_permit):
        raise ValueError("Qualification decoded input differs from actual accepted train-only 100K")
    feature = build_indexed_feature_binding(data.train, family=recipe.interface, configuration=recipe.model_config)
    if canonical_fingerprint(feature) != context.train_feature_binding_sha256:
        raise ValueError("Qualification feature/source/config/history binding differs")
    _assert_target_statistics(recipe, data)
    from .experiment_family_workflow import tensor_digest
    import torch
    arm = next(a for a in declaration["arms"] if a["arm_id"] == options["arm_id"])
    membership = tensor_digest(torch.tensor(data.train_source_indices, dtype=torch.int64), role="source_idx")
    if any(r["membership_sha256"] != membership or r["row_order_sha256"] != membership for r in arm["data"]["roles"]):
        raise ValueError("Qualification indexed members differ from Spec")
    admission.validate(recipe)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Qualification requires one isolated visible accelerator per worker")
    validate_accelerator_request("NvidiaTeslaT4", torch.cuda.get_device_name(0), hip_runtime=torch.version.hip is not None)
    torch.set_num_threads(tuning.cpu_threads)
    runtime = build_runtime_manifest(configure_fp32_determinism(recipe.seed))
    provenance = {"context": observed.to_dict(), "recipe_identity": recipe.identity,
        "dataset_manifest_sha256": recipe.dataset.manifest_sha256, "initialization": initial,
        "runtime_fingerprint": runtime["runtime_fingerprint"], "tuning_identity": tuning.identity,
        "accelerator": torch.cuda.get_device_name(0), "qualification_context": asdict(context),
        "qualification_context_identity": context.identity, "scope": "QUALIFICATION_ONLY"}
    return (record, recipe, data, tuning, observed, provenance, source_root), admission


def run_modern2d_qualification(*, qualification, authority_path, **options):
    from .family_training_runner import run_screen_preflight
    return run_screen_preflight(qualification=qualification, authority_path=authority_path, **options)


def run_synthetic_modern2d_qualification(*, recipe, data, initial_state_path, output):
    """Direct generated CPU verification route; never native acceptance."""
    from .family_training_recipe import FamilyRecipe
    from .family_training_data import TrainingData
    from .family_training_runner import _run_screen_preflight
    from .gptrans_execution_tuning import ExecutionTuning
    if (type(recipe) is not FamilyRecipe or type(data) is not TrainingData or
            type(recipe.qualification) is not QualificationProbe or
            not recipe.dataset.manifest.startswith("synthetic/") or data.synthetic is not True or
            type(data.train) not in (list, tuple) or data.development is not None):
        raise ValueError("Direct verification requires bounded generated train-only CPU fixtures")
    recipe.qualification.validate(recipe)
    if recipe.family == "esa_small" and recipe.model_config.tensor_device == "cuda":
        raise ValueError("Synthetic CPU verification requires a CPU-capable ESA configuration")
    provenance = {"context": {"scope": "CPU_SYNTHETIC_ONLY", "recipe_identity": recipe.identity},
        "recipe_identity": recipe.identity, "accelerator": "synthetic-cpu", "scope": "CPU_SYNTHETIC_ONLY"}
    tuning = ExecutionTuning(training_workers=0, development_workers=0)
    prepared = ({}, recipe, data, tuning, None, provenance, None)
    return _run_screen_preflight(prepared, output=output, options={"initial_state_path": initial_state_path},
        device="cpu", synthetic=True)
