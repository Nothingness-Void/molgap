"""Reviewed family execution interfaces, separate from declarative Spec support.

Only this registry selects executable owners. No caller-provided imports or
callbacks are accepted. A supported recipe is still not execution authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import re
from types import MappingProxyType

from .experiment_spec import ADDONS, FAMILIES, ExperimentSpec, SCHEMA_VERSION_V2, validate_addon_config


MODERN2D_PREVIEW_PROFILE = "modern2d-session-preview-v1"
MODERN2D_FORMAL_PROFILE = "modern2d-formal-session-v1"
SHARED_SCIENTIFIC_INPUT_ABI = "shared-family-v1"
MODERN2D_SCIENTIFIC_INPUT_ABI = "modern2d-indexed-v1"
SUPPORTED_SCIENTIFIC_INPUT_ABIS = frozenset({SHARED_SCIENTIFIC_INPUT_ABI,
                                             MODERN2D_SCIENTIFIC_INPUT_ABI})


def _catalog_modern2d_interfaces():
    """Project compatibility views from FamilyExecution, never hand-edit them."""
    formal, formal_model, preview = {}, {}, {}
    for family, contract in FAMILIES.items():
        entry = contract.execution
        if entry is None or entry.scientific_input_abi != MODERN2D_SCIENTIFIC_INPUT_ABI:
            continue
        if entry.artifact_adapter == MODERN2D_FORMAL_PROFILE:
            formal[family] = (entry.model_factory, contract.feature_schema)
            if entry.model_interface is None:
                raise ValueError("Formal Modern2D family lacks its input interface")
            formal_model[family] = tuple(entry.model_interface)
        elif entry.artifact_adapter == MODERN2D_PREVIEW_PROFILE:
            preview[family] = (entry.model_factory, contract.feature_schema)
    return MappingProxyType(formal), MappingProxyType(formal_model), MappingProxyType(preview)


# Compatibility projections for callers that still import these names.  The
# catalog is the only source of truth.
MODERN2D_FORMAL_INTERFACES, MODERN2D_FORMAL_MODEL_INTERFACES, MODERN2D_PREVIEW_INTERFACES = (
    _catalog_modern2d_interfaces())


@dataclass(frozen=True)
class ExecutionPhase:
    """Reviewed neutral worker descriptor; platform owns spawning and timeouts."""
    name: str
    worker: str = "molgap.experiment_training_worker"
    swap_devices: bool = False
    separate_output: bool = False
    preflight_binding: bool = False

    def __post_init__(self):
        if (type(self.name) is not str or re.fullmatch(r"[a-z][a-z0-9-]*", self.name) is None or
                type(self.worker) is not str or re.fullmatch(r"molgap(?:\.[A-Za-z_][A-Za-z0-9_]*)+", self.worker) is None or
                any(type(value) is not bool for value in (self.swap_devices, self.separate_output, self.preflight_binding))):
            raise ValueError("Invalid reviewed execution phase descriptor")

    def device(self, declared):
        return 1 - declared if self.swap_devices else declared


@dataclass(frozen=True)
class ExecutionExtension:
    phases: tuple[ExecutionPhase, ...] = (ExecutionPhase("preflight"), ExecutionPhase("train"))
    configuration: dict | None = None
    wall_ceiling_seconds: int | None = None
    artifacts: object = None
    finalize: object = None

    def __post_init__(self):
        if (type(self.phases) is not tuple or not self.phases or any(type(phase) is not ExecutionPhase for phase in self.phases) or
                self.phases[0].name != "preflight" or len({phase.name for phase in self.phases}) != len(self.phases) or
                any(hook is not None and not callable(hook) for hook in (self.artifacts, self.finalize)) or
                (self.wall_ceiling_seconds is not None and
                 (type(self.wall_ceiling_seconds) is not int or self.wall_ceiling_seconds < 1))):
            raise ValueError("Invalid bounded execution extension")




@dataclass(frozen=True)
class TrainingAddon:
    """Reviewed execution delta; declaration and source identity stay with Spec."""
    name: str
    version: str
    mode: str
    extra_source_files: tuple[str, ...] = ()
    apply_hook: str | None = None
    policy_hook: str | None = None
    conditional_optimizer_prefixes: tuple[str, ...] = ()
    conditional_optimizer_buffers: tuple[str, ...] = ()
    legacy_conditional_optimizer_tail: bool = False


@dataclass(frozen=True)
class TrainingAdapter:
    family: tuple[str, str]
    module: str
    artifact_adapter: str
    addons: tuple[TrainingAddon, ...]
    serialization_modules: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()
    model_factory: str | None = None
    target_identity_in_completion: bool = False
    profile_adapter: str | None = None
    cpu_synthetic_only: bool = False

    @property
    def integration_preview(self):
        return (self.family in MODERN2D_PREVIEW_INTERFACES and
                self.artifact_adapter == MODERN2D_PREVIEW_PROFILE)

    @property
    def formal_training(self):
        return (self.family in MODERN2D_FORMAL_INTERFACES and
                self.artifact_adapter == MODERN2D_FORMAL_PROFILE)

    @property
    def addon_modes(self) -> tuple[tuple[str, str], ...]:
        return tuple((addon.name, addon.mode) for addon in self.addons)

    def addon(self, name: str, version: str = "1") -> TrainingAddon:
        match = next((a for a in self.addons if (a.name, a.version) == (name, version)), None)
        if match is None:
            raise ValueError("Addon has no executable family adapter")
        return match

    def mode(self, arm: dict) -> str:
        addons = arm["addons"]
        if not addons:
            return "reference"
        if self.module == "molgap.family_training_runner":
            entries = [self.addon(a["name"], a["version"]) for a in addons]
            if any(entry.apply_hook is not None for entry in entries):
                if not all(entry.apply_hook is not None for entry in entries):
                    raise ValueError("Slot composition cannot mix frozen mode dispatch")
                return "composed"
        if self.model_factory is not None:
            modes = [self.addon(a["name"], a["version"]).mode for a in addons]
            return modes[0] if len(modes) == 1 else "composed"
        if len(addons) != 1:
            raise ValueError("Execution requires one supported addon")
        return self.addon(addons[0]["name"], addons[0]["version"]).mode


def _hook_source(hook: str) -> str:
    if type(hook) is not str or re.fullmatch(r"molgap(?:\.[A-Za-z_][A-Za-z0-9_]*)+:[A-Za-z_][A-Za-z0-9_]*", hook) is None:
        raise ValueError("Expected a reviewed molgap module:function hook")
    return "src/" + hook.split(":", 1)[0].replace(".", "/") + ".py"


def graph_training_adapter(family, *, model_factory, addons=(), source_files=(),
                           serialization_modules=(), cpu_synthetic_only=False,
                           integration_preview=False, formal_training=False) -> TrainingAdapter:
    """Bind a small model to the shared pure-2D normalized-Gap training owner."""
    sources = {"src/molgap/graph_screen_training.py", "src/molgap/pcqm_graph_inputs.py",
               "src/molgap/shared_model_adapter.py", "src/molgap/pcqm_topology.py",
               "src/molgap/ogb_features.py",
               _hook_source(model_factory), *source_files}
    if integration_preview or formal_training:
        if integration_preview and formal_training:
            raise ValueError("Preview and formal profiles are separate interfaces")
        interfaces = MODERN2D_FORMAL_INTERFACES if formal_training else MODERN2D_PREVIEW_INTERFACES
        expected = interfaces.get(tuple(family))
        if cpu_synthetic_only or expected is None or model_factory != expected[0] or (addons and not formal_training):
            raise ValueError("Unreviewed Modern2D session integration preview")
        sources.update({"src/molgap/family_training_runner.py", "src/molgap/family_training_kernel.py",
                        "src/molgap/family_training_recipe.py", "src/molgap/family_training_outputs.py",
                        "src/molgap/family_execution_profile.py"})
        if formal_training:
            sources.update({"src/molgap/modern2d_training.py", "src/molgap/modern2d_training_release.py",
                            "src/molgap/modern2d_feature_binding.py"})
        return TrainingAdapter(tuple(family), "molgap.family_training_runner",
            MODERN2D_FORMAL_PROFILE if formal_training else MODERN2D_PREVIEW_PROFILE,
            tuple(addons), tuple(serialization_modules), tuple(sorted(sources)), model_factory,
            target_identity_in_completion=formal_training,
            profile_adapter="molgap.family_execution_profile")
    if cpu_synthetic_only:
        sources.add("src/molgap/family_training_runner.py")
        return TrainingAdapter(tuple(family), "molgap.family_training_runner", "modern2d-cpu-v1",
            tuple(addons), tuple(serialization_modules), tuple(sorted(sources)), model_factory,
            cpu_synthetic_only=True)
    return TrainingAdapter(tuple(family), "molgap.graph_screen_training", "graph-screen-v1",
        tuple(addons), tuple(serialization_modules), tuple(sorted(sources)), model_factory)



def catalog_addons(family):
    """Read-only execution projection of the single human-maintained catalog."""
    result=[]
    for (name,version), contract in ADDONS.items():
        if contract.family != family[0]:
            continue
        modes=dict(contract.execution_modes)
        if len(modes) != len(contract.execution_modes):
            raise ValueError("Duplicate catalog execution family version")
        if family[1] in modes:
            result.append(TrainingAddon(name,version,modes[family[1]],contract.extra_source_files,
                                        contract.apply_hook,contract.policy_hook,
                                        contract.conditional_optimizer_prefixes,
                                        contract.conditional_optimizer_buffers,
                                        contract.legacy_conditional_optimizer_tail))
    return tuple(result)


def _catalog_training_adapters():
    result={}
    for key, contract in FAMILIES.items():
        entry=contract.execution
        if entry is None:
            continue
        if entry.model_factory is not None:
            adapter=graph_training_adapter(key, model_factory=entry.model_factory,
                addons=catalog_addons(key), source_files=entry.source_files,
                serialization_modules=entry.serialization_modules,
                cpu_synthetic_only=entry.cpu_synthetic_only,
                integration_preview=entry.artifact_adapter == MODERN2D_PREVIEW_PROFILE,
                formal_training=entry.artifact_adapter == MODERN2D_FORMAL_PROFILE)
            if (adapter.module,adapter.artifact_adapter) != (entry.module,entry.artifact_adapter):
                raise ValueError("Catalog graph family execution boundary differs from its owner")
        else:
            adapter=TrainingAdapter(key,entry.module,entry.artifact_adapter,catalog_addons(key),
                entry.serialization_modules,entry.source_files,entry.model_factory,
                entry.target_identity_in_completion,entry.profile_adapter,entry.cpu_synthetic_only)
        result[key]=adapter
    return result


TRAINING_ADAPTERS = MappingProxyType(_catalog_training_adapters())


def family_execution(family):
    """Return the reviewed execution/input capability for one catalog family."""
    key = tuple(family)
    contract = FAMILIES.get(key)
    if contract is None or contract.execution is None:
        raise ValueError(f"Family {key} has no executable catalog contract")
    return contract.execution


def scientific_input_abi(family):
    """Return the declared compiler ABI; ``None`` means declaration-only."""
    return family_execution(family).scientific_input_abi


def modern2d_preview_family(family):
    """Resolve a formal family to the indexed input/model interface it consumes."""
    key = tuple(family)
    entry = family_execution(key)
    if entry.scientific_input_abi != MODERN2D_SCIENTIFIC_INPUT_ABI:
        raise ValueError("Family has no reviewed Modern2D indexed input ABI")
    preview = tuple(entry.model_interface or key)
    preview_entry = family_execution(preview)
    if (preview_entry.scientific_input_abi != MODERN2D_SCIENTIFIC_INPUT_ABI or
            preview_entry.artifact_adapter != MODERN2D_PREVIEW_PROFILE):
        raise ValueError("Modern2D family lacks a reviewed preview input interface")
    return preview


def _recipe_training_addons(recipe):
    adapter = TRAINING_ADAPTERS.get(recipe.interface)
    if adapter is None:
        raise ValueError("Recipe has no executable family adapter")
    if recipe.mode == "composed":
        return [adapter.addon(item["name"], item["version"]) for item in recipe.addons]
    return [entry for entry in adapter.addons if entry.mode == recipe.mode]


def conditional_optimizer_parameters(recipe, names):
    """Select declared addon parameters that may lack optimizer moments."""
    entries = _recipe_training_addons(recipe)
    prefixes = tuple(dict.fromkeys(prefix for entry in entries
                                   for prefix in entry.conditional_optimizer_prefixes))
    buffers = {name for entry in entries for name in entry.conditional_optimizer_buffers}
    return {name for name in names if name not in buffers and any(
        name.startswith(prefix) for prefix in prefixes)}


def legacy_conditional_optimizer_tail(recipe):
    """Whether an old checkpoint may infer conditional IDs from a suffix."""
    if recipe.mode == "composed":
        return False
    entries = [entry for entry in _recipe_training_addons(recipe)
               if entry.conditional_optimizer_prefixes]
    return bool(entries) and all(entry.legacy_conditional_optimizer_tail for entry in entries)


def shared_training_adapters():
    """Project the shared recipe owners without guessing a family version."""
    result = {}
    for adapter in TRAINING_ADAPTERS.values():
        if adapter.module != "molgap.family_training_runner" or adapter.model_factory is not None:
            continue
        name = adapter.family[0]
        if name in result:
            raise ValueError("Ambiguous shared training family version: " + name)
        result[name] = adapter
    return MappingProxyType(result)


SESSION_INTERFACES = MappingProxyType({
    **{name: adapter.family for name, adapter in shared_training_adapters().items()},
    # Historical CPU defaults stay explicit; native interfaces require admission.
    "ppgt_small": ("ppgt_small", "1"), "esa_small": ("esa_small", "1"),
})


def validate_training_registry() -> None:
    """Check cross-owner registrations without importing or executing trainers."""
    from .experiment_family_artifacts import artifact_adapter
    for family, adapter in TRAINING_ADAPTERS.items():
        if family != adapter.family or family not in FAMILIES:
            raise ValueError("Training registration differs from its family contract")
        artifact_adapter(adapter.artifact_adapter, family)
        if "src/" + adapter.module.replace(".", "/") + ".py" not in adapter.source_files:
            raise ValueError("Training registration omits its owning source module")
        if adapter.integration_preview or adapter.formal_training:
            contract = FAMILIES[family]
            factory, schema = (MODERN2D_FORMAL_INTERFACES if adapter.formal_training else MODERN2D_PREVIEW_INTERFACES)[family]
            entry = contract.execution
            if (adapter.cpu_synthetic_only or adapter.module != "molgap.family_training_runner"
                    or (adapter.addons and not adapter.formal_training) or adapter.model_factory != factory
                    or contract.source_module != factory.split(":")[0]
                    or contract.feature_schema != schema or contract.recipe != (
                        "modern2d_formal_training_v1" if adapter.formal_training else "modern2d_session_preview_v1")
                    or contract.roles != ("train",) or contract.optional_roles != ("development",)
                    or contract.sampler != "frozen-family-pass-order"
                    or contract.transform != "frozen-train-mean-sample-std"
                    or adapter.profile_adapter != "molgap.family_execution_profile"
                    or (adapter.formal_training and not adapter.target_identity_in_completion)
                    or entry is None or entry.scientific_input_abi != MODERN2D_SCIENTIFIC_INPUT_ABI
                    or entry.feature_module is None or not entry.feature_config_fields
                    or (adapter.formal_training and tuple(entry.model_interface or ()) not in MODERN2D_PREVIEW_INTERFACES)
                    or ((not adapter.formal_training) and tuple(entry.model_interface or family) != family)
                    or _hook_source(factory) not in adapter.source_files):
                raise ValueError("Unreviewed Modern2D integration preview registration")
            _validate_registered_addons(family, adapter)
            continue
        if adapter.cpu_synthetic_only:
            contract = FAMILIES[family]
            expected = {("ppgt_small", "1"): "ogb-atom9-bond3-rrwp-spe-v1",
                        ("esa_small", "1"): "ogb-atom9-bond3-directed-edge-mask-v1",
                        ("esa_small", "2"): "ogb-atom9-bond3-directed-edge-mask-v1"}
            if (family not in expected or adapter.module != "molgap.family_training_runner"
                    or adapter.artifact_adapter != "modern2d-cpu-v1" or adapter.addons
                    or contract.feature_schema != expected[family] or contract.recipe != "modern2d_cpu_v1"
                    or contract.roles != ("train", "development")
                    or adapter.model_factory != contract.source_module + ":make_model"
                    or _hook_source(adapter.model_factory) not in adapter.source_files):
                raise ValueError("Unreviewed CPU-only family/profile registration")
            continue
        if adapter.model_factory is not None:
            if (adapter.module != "molgap.graph_screen_training" or adapter.artifact_adapter != "graph-screen-v1"
                    or _hook_source(adapter.model_factory) not in adapter.source_files
                    or not adapter.model_factory.endswith(":make_model")):
                raise ValueError("Shared graph model registration differs from its reviewed owner/source")
            contract = FAMILIES[family]
            if _hook_source(adapter.model_factory) != "src/" + contract.source_module.replace(".", "/") + ".py":
                raise ValueError("Shared graph factory differs from its declared model source")
            if (contract.feature_schema != "ogb-atom9-bond3-rwse16-v1"
                    or contract.roles != ("train", "development")
                    or contract.recipe != "graph_gap_screen_v1"
                    or contract.sampler != "seed42-epoch-global-randperm-v1"
                    or contract.transform != "train-mean-unbiased-std"):
                raise ValueError("Shared graph training requires its explicit feature/role/recipe contract")
        elif adapter.module == "molgap.graph_screen_training":
            raise ValueError("Shared graph training requires a reviewed model factory")
        _validate_registered_addons(family, adapter)


def _validate_registered_addons(family, adapter):
    if len({(a.name, a.version) for a in adapter.addons}) != len(adapter.addons):
        raise ValueError("Duplicate executable addon registration")
    for addon in adapter.addons:
        contract = ADDONS.get((addon.name, addon.version))
        if contract is None or contract.family != family[0]:
            raise ValueError("Executable addon differs from its declaration contract")
        if contract.family_versions and family[1] not in contract.family_versions:
            raise ValueError("Executable addon differs from its declared family version")
        if type(addon.mode) is not str or not addon.mode:
            raise ValueError("Executable addon requires an explicit owning mode")
        if addon.policy_hook is not None and (adapter.module != "molgap.family_training_runner" or
                not addon.policy_hook.endswith(":policy")):
            raise ValueError("Policy hooks require the reviewed family session")
        if addon.policy_hook is not None:
            _hook_source(addon.policy_hook)
        if (any(type(prefix) is not str or not prefix for prefix in addon.conditional_optimizer_prefixes)
                or any(type(name) is not str or not name for name in addon.conditional_optimizer_buffers)
                or any(not any(name.startswith(prefix) for prefix in addon.conditional_optimizer_prefixes)
                       for name in addon.conditional_optimizer_buffers)
                or type(addon.legacy_conditional_optimizer_tail) is not bool):
            raise ValueError("Conditional optimizer metadata is not a reviewed parameter prefix")
        if adapter.model_factory is not None or addon.apply_hook is not None:
            if (addon.apply_hook is None or not addon.apply_hook.endswith(":apply_addon")
                    or _hook_source(addon.apply_hook) != "src/" + contract.source_module.replace(".", "/") + ".py"):
                raise ValueError("Shared graph addon requires its reviewed model-delta hook")
            if adapter.model_factory is None and adapter.module != "molgap.family_training_runner":
                raise ValueError("Slot addons require the accelerated family session")


def build_addon_declaration(family, addon, *, repo_root, config=None, version="1") -> dict:
    """Bind a reviewed addon source; its owning Spec validates configuration."""
    from .experiment_package import _source
    from .v4_runtime import normalized_source_sha256
    adapter = TRAINING_ADAPTERS.get(tuple(family))
    if adapter is None:
        raise ValueError("No executable family adapter")
    adapter.addon(addon, version)
    contract = ADDONS[(addon, version)]
    if type(config) is not dict and config is not None:
        raise ValueError("Addon configuration must be an explicit mapping")
    if config is None:
        if any(field.kind != "literal" for field in contract.config_fields):
            raise ValueError("This addon requires an explicit bounded configuration")
        config = {field.name: field.value for field in contract.config_fields}
    validate_addon_config(contract, config, name=addon)
    source = _source(Path(repo_root).resolve(), "src/" + contract.source_module.replace(".", "/") + ".py")
    return {"name": addon, "version": version, "config": dict(config),
            "source_sha256": normalized_source_sha256(source)}


def workflow_capabilities(spec: ExperimentSpec) -> dict:
    """Expose route selection without claiming scientific or platform authority."""
    supported, unsupported = {}, {}
    integration_pending = False
    for arm in spec.to_dict()["arms"]:
        try:
            adapter = training_adapter(arm)
            if adapter.cpu_synthetic_only:
                unsupported[arm["arm_id"]] = "CPU synthetic registration only; native training is not released"
                continue
            if adapter.integration_preview:
                integration_pending = True
                unsupported[arm["arm_id"]] = (
                    "INTEGRATION_PENDING: session preview only; family fidelity, real inputs and native qualification remain unreleased")
                continue
            supported[arm["arm_id"]] = {"mode": adapter.mode(arm), "trainer": adapter.module,
                                        "artifact_profile": adapter.artifact_adapter,
                                        "profile_adapter": adapter.profile_adapter}
            if adapter.model_factory is not None:
                supported[arm["arm_id"]]["model_factory"] = adapter.model_factory
        except ValueError as exc:
            unsupported[arm["arm_id"]] = str(exc)
    registered = not unsupported and spec.to_dict()["platform"]["name"] == "kaggle"
    return {"spec_identity": spec.identity, "registered_training_arms": supported,
            "unsupported_training_arms": unsupported,
            "preparation_entry": "prepare-workflow" if registered else "prepare-release",
            "execution_qualification": "INTEGRATION_PENDING" if integration_pending else "requires_owning_contract_and_runtime",
            "submitted": False}


def training_adapter(arm: dict) -> TrainingAdapter:
    key = (arm["family"]["name"], arm["family"]["version"])
    try:
        adapter = TRAINING_ADAPTERS[key]
    except KeyError as exc:
        raise ValueError(f"Family {key} has declaration/model support only; no training adapter") from exc
    adapter.mode(arm)
    return adapter


def build_family_recipe(family: tuple[str, str], *, addon: str | tuple[str, ...] | None = None,
                        source_idx_sha256: str | None = None, target_sha256: str | None = None,
                        addon_version: str = "1", recipe_config: dict | None = None,
                        configuration=None, train_role_identity=None, development_role_identity=None,
                        logging=None) -> dict:
    """Build owned constants before freezing the Spec; never read development rows."""
    adapter = TRAINING_ADAPTERS.get(family)
    if adapter is None:
        raise ValueError("No executable family recipe builder")
    if logging is not None and adapter.module != "molgap.family_training_runner":
        raise ValueError("Configurable training logging requires the shared family session")
    if isinstance(addon, tuple):
        if (adapter.model_factory is None and adapter.module != "molgap.family_training_runner") or len(addon) < 2 or len(set(addon)) != len(addon):
            raise ValueError("Composed addons require distinct reviewed shared-model hooks")
        for name in addon:
            adapter.addon(name, addon_version)
        mode = "composed"
    else:
        mode = "reference" if addon is None else adapter.addon(addon, addon_version).mode
    if adapter.module == "molgap.family_training_runner":
        names = addon if isinstance(addon, tuple) else (() if addon is None else (addon,))
        if configuration is None or tuple(a["name"] for a in configuration.addons) != (
                names if mode == "composed" else ()):
            raise ValueError("Recipe ordered slot addons differ from the selected declarations")
        if recipe_config is not None:
            raise ValueError("Shared family configuration must use FamilyRecipe")
        return import_module(adapter.module).build_screen_recipe(mode, configuration=configuration,
            family=family, source_idx_sha256=source_idx_sha256, target_sha256=target_sha256,
            train_role_identity=train_role_identity, development_role_identity=development_role_identity,
            logging=logging)
    if configuration is not None or train_role_identity is not None or development_role_identity is not None:
        raise ValueError("Explicit shared configuration is not supported by historical adapters")
    if adapter.model_factory is not None:
        return import_module(adapter.module).build_screen_recipe(mode, family=family,
            source_idx_sha256=source_idx_sha256, target_sha256=target_sha256,
            recipe_config=recipe_config)
    if recipe_config is not None:
        raise ValueError("Frozen family recipes do not accept generic configuration")
    return import_module(adapter.module).build_screen_recipe(mode,
        source_idx_sha256=source_idx_sha256, target_sha256=target_sha256)


def check_family_recipes(spec, repo_root, recipes):
    """Call the selected family's static validator on its frozen recipe."""
    from .experiment_family_workflow import _json, _artifact_path
    from .v4_runtime import normalized_source_sha256
    results = {}
    for arm in spec.to_dict()["arms"]:
        adapter = training_adapter(arm)
        owner = import_module(adapter.module)
        if not Path(owner.__file__).resolve().is_relative_to(Path(repo_root).resolve()):
            raise ValueError("Family validator was imported from another checkout")
        recipe_path = _artifact_path(repo_root, recipes[arm["arm_id"]])
        if normalized_source_sha256(recipe_path) != arm["training"]["recipe"]["sha256"]:
            raise ValueError("Family recipe/Spec hash mismatch")
        record = _json(recipe_path)
        if "profiling" in record:
            if adapter.profile_adapter is None:
                raise ValueError("Family has no registered execution profiler: " + arm["arm_id"])
            from .family_execution_profile import ProfileConfig
            ProfileConfig.from_dict(record["profiling"])
            profile = import_module(adapter.profile_adapter)
            if not Path(profile.__file__).resolve().is_relative_to(Path(repo_root).resolve()):
                raise ValueError("Profile adapter was imported from another checkout")
        results[arm["arm_id"]] = owner.validate_screen_recipe(spec, arm["arm_id"], record)
    return results


def validate_execution_plan(spec: ExperimentSpec, jobs: list[dict]) -> list[dict]:
    """Bind every independent arm and its assigned device before spawning."""
    declaration = spec.to_dict()
    if any(training_adapter(arm).cpu_synthetic_only for arm in declaration["arms"]):
        raise ValueError("CPU synthetic profile cannot enter a platform execution plan")
    if any(training_adapter(arm).integration_preview for arm in declaration["arms"]):
        raise ValueError("INTEGRATION_PENDING: Modern2D session preview cannot enter a platform execution plan")
    if declaration["schema_version"] != SCHEMA_VERSION_V2:
        raise ValueError("Training workflow requires per-arm prospective Spec v2")
    arms = declaration["arms"]
    if type(jobs) is not list or len(jobs) != len(arms):
        raise ValueError("Execution requires exactly all Spec arms")
    bindings = {b["arm_id"]: b for b in declaration["prospective"]["arms"]}
    seen, devices, result = set(), set(), []
    for job in jobs:
        if type(job) is not dict or set(job) != {"arm_id", "device", "recipe", "initial_state", "trajectory_id"}:
            raise ValueError("Expected explicit arm/device/recipe/initial_state/trajectory_id")
        arm = next((a for a in arms if a["arm_id"] == job["arm_id"]), None)
        if arm is None or job["arm_id"] in seen:
            raise ValueError("Unknown or duplicate arm")
        device = job["device"]
        if type(device) is not int or not 0 <= device < declaration["platform"]["device_count"] or device in devices:
            raise ValueError("Every concurrent arm needs a distinct declared device")
        for field in ("recipe", "initial_state"):
            value = job[field]
            if type(value) is not str or not value or any(p in {"", ".", ".."} for p in value.split("/")) or "\\" in value or ":" in value or value.startswith("/"):
                raise ValueError("Expected confined relative POSIX execution path")
        if job["trajectory_id"] != bindings[job["arm_id"]]["trajectory_id"]:
            raise ValueError("Execution/prospective trajectory mismatch")
        adapter = training_adapter(arm)
        required_modules = [adapter.module, *adapter.serialization_modules]
        if adapter.module == "molgap.family_training_runner":
            # _bootstrap imports this dependency inside the function, so an
            # import of the runner alone does not exercise the worker boundary.
            required_modules.append("molgap.gptrans_execution_tuning")
        result.append({**job, "mode": adapter.mode(arm), "module": adapter.module,
                       "required_modules": required_modules})
        seen.add(job["arm_id"])
        devices.add(device)
    return result


def validate_staged_trajectory(spec, arm_id, input_root: Path, *, expected_sha256=None) -> dict:
    """Require the published prospective record, not merely a declared ID."""
    from .experiment_family_workflow import _json, _artifact_path
    from .screen_policy import canonical_fingerprint
    binding = next(b for b in spec.to_dict()["prospective"]["arms"] if b["arm_id"] == arm_id)
    arm = next(a for a in spec.to_dict()["arms"] if a["arm_id"] == arm_id)
    relative = f"prospective/{arm_id}/trajectory.json"
    direct = Path(input_root) / relative
    matches = [direct] if direct.is_file() else list(Path(input_root).glob("**/" + relative))
    if len(matches) != 1:
        raise ValueError("Expected exactly one staged prospective trajectory per arm")
    path = _artifact_path(Path(input_root), matches[0].relative_to(input_root).as_posix())
    if expected_sha256 is not None:
        from .research_memory.trace import file_digest
        if file_digest(path) != expected_sha256:
            raise ValueError("Staged prospective record hash mismatch")
    trajectory = _json(path)
    state = trajectory.get("state_at_start", {})
    if (trajectory.get("record_mode") != "prospective" or
        trajectory.get("trajectory_id") != binding["trajectory_id"] or
        state.get("source_config_identity") != canonical_fingerprint(arm)):
        raise ValueError("Published prospective trajectory/arm identity mismatch")
    # A copied terminal history never authorizes another attempt.
    from .research_memory.schemas import validate_trajectory
    validate_trajectory(trajectory)
    if trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Staged trajectory is terminal")
    return trajectory


def execute_training_phase(*, spec, job, phase, source_root, package_dir,
                           expected_package_identity, input_root, output,
                           account, run_reference, staged_root=None, prospective_sha256=None,
                           preflight_dir=None, resume_output=None, target_identity=None,
                           formal_training=None, authority_path=None):
    """Static dispatch; each owning trainer retains its recipe/runtime gates."""
    from .experiment_family_workflow import RunContext, _artifact_path
    from .v4_runtime import inspect_frozen_state_artifact, normalized_source_sha256
    if phase not in {"preflight", "train"}:
        raise ValueError("Unsupported execution phase")
    arm = next(a for a in spec.to_dict()["arms"] if a["arm_id"] == job["arm_id"])
    adapter = training_adapter(arm)
    if adapter.integration_preview:
        raise ValueError("INTEGRATION_PENDING: Modern2D session preview has no native execution release")
    context = RunContext.for_training(spec, package_dir,
        expected_package_identity=expected_package_identity, arm_id=job["arm_id"],
        account=account, run_reference=run_reference)
    staged_root = Path(staged_root or input_root)
    if prospective_sha256 is None:
        raise ValueError("Execution requires the prepared prospective record digest")
    trajectory = validate_staged_trajectory(spec, job["arm_id"], staged_root, expected_sha256=prospective_sha256)
    if trajectory["state_at_start"].get("source_commit") != context.source_commit:
        raise ValueError("Published trajectory/source commit mismatch")
    recipe = _artifact_path(source_root, job["recipe"])
    if normalized_source_sha256(recipe) != context.training_recipe_sha256:
        raise ValueError("Executable recipe/Spec hash mismatch")
    initial = _artifact_path(staged_root, job["initial_state"])
    if arm["initialization"]["state_sha256"] is None:
        raise ValueError("Screen adapter requires pinned initialization state")
    if adapter.formal_training:
        if formal_training is None:
            from .modern2d_training_release import build_formal_context
            formal_training, authority_path = build_formal_context(spec=spec, job=job,
                source_root=source_root, staged_root=staged_root, input_root=input_root,
                run_context=context)
        # The runner validates authority and both role permits before state IO.
    else:
        inspect_frozen_state_artifact(initial, expected_state_sha256=arm["initialization"]["state_sha256"])
    # The import string comes exclusively from the immutable reviewed registry.
    # A new family extends that registry, never this orchestration function.
    owner = import_module(adapter.module)
    runner = owner.run_screen_preflight if phase == "preflight" else owner.run_screen_arm
    options = {} if phase == "preflight" else {"preflight_dir": preflight_dir or output}
    if adapter.formal_training:
        options.update(formal_training=formal_training, authority_path=authority_path)
    elif formal_training is not None or authority_path is not None:
        raise ValueError("Formal admission requires its reviewed versioned interface")
    if adapter.module == "molgap.family_training_runner":
        options["prospective_root"] = staged_root
        if phase == "train":
            options["target_identity"] = target_identity
    if phase == "preflight" and resume_output is not None:
        runner = getattr(owner, "run_screen_resume_preflight", None)
        if not callable(runner):
            raise ValueError("Owning family has no portable resume preflight hook")
        options["resume_output"] = resume_output
    return runner(spec=spec, package_dir=package_dir,
        expected_package_identity=expected_package_identity, arm_id=job["arm_id"],
        mode=adapter.mode(arm), recipe_path=recipe, initial_state_path=initial,
        input_root=input_root, output=output, account=account,
        run_reference=run_reference, trajectory_id=job["trajectory_id"], **options)
