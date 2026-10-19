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


@dataclass(frozen=True)
class TrainingAddon:
    """Reviewed execution delta; declaration and source identity stay with Spec."""
    name: str
    version: str
    mode: str
    extra_source_files: tuple[str, ...] = ()
    apply_hook: str | None = None
    policy_hook: str | None = None


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

    def __post_init__(self):
        # Preserve reviewed legacy tuple registrations while exposing typed hooks.
        addons = tuple(a if isinstance(a, TrainingAddon) else TrainingAddon(a[0], "1", a[1])
                       for a in self.addons)
        object.__setattr__(self, "addons", addons)
        if not self.source_files:
            object.__setattr__(self, "source_files",
                               ("src/" + self.module.replace(".", "/") + ".py",))

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
                           serialization_modules=(), cpu_synthetic_only=False) -> TrainingAdapter:
    """Bind a small model to the shared pure-2D normalized-Gap training owner."""
    sources = {"src/molgap/graph_screen_training.py", "src/molgap/pcqm_graph_inputs.py",
               "src/molgap/shared_model_adapter.py", "src/molgap/pcqm_topology.py",
               "src/molgap/ogb_features.py",
               _hook_source(model_factory), *source_files}
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
                                        contract.apply_hook,contract.policy_hook))
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
                cpu_synthetic_only=entry.cpu_synthetic_only)
            if (adapter.module,adapter.artifact_adapter) != (entry.module,entry.artifact_adapter):
                raise ValueError("Catalog graph family execution boundary differs from its owner")
        else:
            adapter=TrainingAdapter(key,entry.module,entry.artifact_adapter,catalog_addons(key),
                entry.serialization_modules,entry.source_files,entry.model_factory,
                entry.target_identity_in_completion,entry.profile_adapter,entry.cpu_synthetic_only)
        result[key]=adapter
    return result


TRAINING_ADAPTERS = MappingProxyType(_catalog_training_adapters())


def validate_training_registry() -> None:
    """Check cross-owner registrations without importing or executing trainers."""
    from .experiment_family_artifacts import artifact_adapter
    for family, adapter in TRAINING_ADAPTERS.items():
        if family != adapter.family or family not in FAMILIES:
            raise ValueError("Training registration differs from its family contract")
        artifact_adapter(adapter.artifact_adapter, family)
        if "src/" + adapter.module.replace(".", "/") + ".py" not in adapter.source_files:
            raise ValueError("Training registration omits its owning source module")
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
        for module in adapter.serialization_modules:
            _hook_source(module + ":deserialize")
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
            if addon.policy_hook is not None:
                if adapter.module != "molgap.family_training_runner" or not addon.policy_hook.endswith(":policy"):
                    raise ValueError("Policy hooks require the reviewed family session")
                _hook_source(addon.policy_hook)
            if adapter.model_factory is not None or addon.apply_hook is not None:
                if (addon.apply_hook is None or not addon.apply_hook.endswith(":apply_addon")
                        or _hook_source(addon.apply_hook) != "src/" + contract.source_module.replace(".", "/") + ".py"):
                    raise ValueError("Shared graph addon requires its reviewed model-delta hook")
                if adapter.model_factory is None and adapter.module != "molgap.family_training_runner":
                    raise ValueError("Slot addons require the accelerated family session")


def build_addon_declaration(family, addon, *, repo_root, config=None, version="1",
                            initialization_seed=42) -> dict:
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
        if any(field.kind not in {"literal", "initialization_seed"} for field in contract.config_fields):
            raise ValueError("This addon requires an explicit bounded configuration")
        config = {field.name: initialization_seed if field.kind == "initialization_seed" and family[1] == "3"
                  else field.value for field in contract.config_fields}
    validate_addon_config(contract, config, name=addon, family_version=family[1],
                          initialization_seed=initialization_seed)
    source = _source(Path(repo_root).resolve(), "src/" + contract.source_module.replace(".", "/") + ".py")
    return {"name": addon, "version": version, "config": dict(config),
            "source_sha256": normalized_source_sha256(source)}


def workflow_capabilities(spec: ExperimentSpec) -> dict:
    """Expose route selection without claiming scientific or platform authority."""
    supported, unsupported = {}, {}
    for arm in spec.to_dict()["arms"]:
        try:
            adapter = training_adapter(arm)
            supported[arm["arm_id"]] = {"mode": adapter.mode(arm), "trainer": adapter.module,
                                        "artifact_profile": adapter.artifact_adapter,
                                        "profile_adapter": adapter.profile_adapter}
            if adapter.model_factory is not None:
                supported[arm["arm_id"]]["model_factory"] = adapter.model_factory
        except ValueError as exc:
            unsupported[arm["arm_id"]] = str(exc)
    declaration = spec.to_dict()
    entry = None
    reason = "No registered preparation route for this Spec/platform"
    if not unsupported and declaration["schema_version"] == SCHEMA_VERSION_V2:
        if declaration["platform"]["name"] == "kaggle":
            entry, reason = "prepare-workflow", None
        elif all(TRAINING_ADAPTERS[(a["family"]["name"], a["family"]["version"])].module
                 == "molgap.family_training_runner" for a in declaration["arms"]):
            entry, reason = "prepare-family-training", None
    return {"spec_identity": spec.identity, "registered_training_arms": supported,
            "unsupported_training_arms": unsupported,
            "preparation_entry": entry, "preparation_unavailable_reason": reason,
            "execution_qualification": "requires_owning_contract_and_runtime",
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
                        addon_version: str = "1", recipe_config: dict | None = None,
                        source_idx_sha256: str | None = None, target_sha256: str | None = None,
                        seed: int | None = None, initialization_sha256: str | None = None,
                        execution_torch_version: str | None = None,
                        configuration=None, train_role_identity=None, development_role_identity=None,
                        logging=None) -> dict:
    """Build owned constants before freezing the Spec; never read development rows."""
    adapter = TRAINING_ADAPTERS.get(family)
    if adapter is None:
        raise ValueError("No executable family recipe builder")
    if logging is not None and adapter.module != "molgap.family_training_runner":
        raise ValueError("Configurable training logging requires the shared family session")
    if isinstance(addon, tuple):
        if ((adapter.model_factory is None and adapter.module != "molgap.family_training_runner")
                or len(addon) < 2 or len(set(addon)) != len(addon)):
            raise ValueError("Composed addons require distinct reviewed shared-model hooks")
        for name in addon:
            adapter.addon(name, addon_version)
        mode = "composed"
    else:
        mode = "reference" if addon is None else adapter.addon(addon, addon_version).mode
    if adapter.model_factory is not None:
        if any(value is not None for value in (seed, initialization_sha256, execution_torch_version,
                                               configuration, train_role_identity, development_role_identity)):
            raise ValueError("Generic graph recipes require their explicit recipe_config")
        return import_module(adapter.module).build_screen_recipe(mode, family=family,
            source_idx_sha256=source_idx_sha256, target_sha256=target_sha256,
            recipe_config=recipe_config)
    if recipe_config is not None:
        raise ValueError("Owned family configuration must use its existing recipe interface")
    if adapter.module == "molgap.family_training_runner":
        if any(value is not None for value in (seed, initialization_sha256, execution_torch_version)):
            raise ValueError("Shared recipe pins seed/initialization/runtime in configuration")
        names = addon if isinstance(addon, tuple) else (() if addon is None else (addon,))
        if configuration is None or tuple(a["name"] for a in configuration.addons) != (
                names if mode == "composed" else ()):
            raise ValueError("Recipe ordered slot addons differ from the selected declarations")
        if mode == "composed" and any(adapter.addon(name, addon_version).apply_hook is None for name in names):
            raise ValueError("Slot composition requires reviewed model-delta hooks")
        return import_module(adapter.module).build_screen_recipe(mode, configuration=configuration,
            family=family, source_idx_sha256=source_idx_sha256, target_sha256=target_sha256,
            train_role_identity=train_role_identity,
            development_role_identity=development_role_identity, logging=logging)
    if configuration is not None or train_role_identity is not None or development_role_identity is not None:
        raise ValueError("Explicit shared configuration requires the shared family interface")
    options = {}
    if execution_torch_version is not None:
        if family != ("neural_atom_k1", "2"):
            raise ValueError("Execution runtime pin requires K1")
        options["execution_torch_version"] = execution_torch_version
    if seed is not None or initialization_sha256 is not None:
        if family != ("neural_atom_k1", "2"):
            raise ValueError("Explicit seed initialization is not supported by this trainer")
        if seed is not None:
            options["seed"] = seed
        if initialization_sha256 is not None:
            options["initialization_sha256"] = initialization_sha256
    return import_module(adapter.module).build_screen_recipe(mode,
        source_idx_sha256=source_idx_sha256, target_sha256=target_sha256, **options)


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
                           target_identity=None, preflight_dir=None, resume_output=None):
    """Static dispatch; each owning trainer retains its recipe/runtime gates."""
    from .experiment_family_workflow import RunContext, _artifact_path
    from .research_memory.trace import file_digest
    from .v4_runtime import inspect_frozen_state_artifact
    if phase not in {"preflight", "train"}:
        raise ValueError("Unsupported execution phase")
    arm = next(a for a in spec.to_dict()["arms"] if a["arm_id"] == job["arm_id"])
    adapter = training_adapter(arm)
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
    if file_digest(recipe) != context.training_recipe_sha256:
        raise ValueError("Executable recipe/Spec hash mismatch")
    initial = _artifact_path(staged_root, job["initial_state"])
    if arm["initialization"]["state_sha256"] is None:
        raise ValueError("Screen adapter requires pinned initialization state")
    inspect_frozen_state_artifact(initial, expected_state_sha256=arm["initialization"]["state_sha256"])
    # The import string comes exclusively from the immutable reviewed registry.
    # A new family extends that registry, never this orchestration function.
    owner = import_module(adapter.module)
    runner = owner.run_screen_preflight if phase == "preflight" else owner.run_screen_arm
    options = {} if phase == "preflight" else {"preflight_dir": preflight_dir or output}
    if phase == "preflight" and resume_output is not None:
        runner = getattr(owner, "run_screen_resume_preflight", None)
        if not callable(runner):
            raise ValueError("Owning family has no portable resume preflight hook")
        options["resume_output"] = resume_output
    if staged_root.resolve() != Path(input_root).resolve():
        if adapter.family != ("neural_atom_k1", "2") and adapter.module != "molgap.family_training_runner":
            raise ValueError("Separate prospective/data roots are not supported by this family")
        options["prospective_root"] = staged_root
    if phase == "train" and adapter.target_identity_in_completion:
        options["target_identity"] = target_identity
    result = runner(spec=spec, package_dir=package_dir,
        expected_package_identity=expected_package_identity, arm_id=job["arm_id"],
        mode=adapter.mode(arm), recipe_path=recipe, initial_state_path=initial,
        input_root=input_root, output=output, account=account,
        run_reference=run_reference, trajectory_id=job["trajectory_id"], **options)
    if phase == "train" and target_identity is not None and not adapter.target_identity_in_completion:
        # Retained family trainers keep their frozen completion call. The owning
        # inspection applies the explicit contract encoding at dispatch closure.
        from .experiment_family_workflow import inspect_output
        normalized = Path(output) / "normalized" if adapter.artifact_adapter == "gptrans-v1" else Path(output)
        import json
        expected = json.loads(recipe.read_text(encoding="utf-8"))["acceptance_requirements"]
        return inspect_output(normalized, context=context, expected=expected, target_identity=target_identity)
    return result
