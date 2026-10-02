"""Reviewed family execution interfaces, separate from declarative Spec support.

Only this registry selects executable owners. No caller-provided imports or
callbacks are accepted. A supported recipe is still not execution authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import MappingProxyType

from .experiment_spec import ADDONS, FAMILIES, ExperimentSpec, SCHEMA_VERSION_V2, validate_addon_config


@dataclass(frozen=True)
class TrainingAddon:
    """Reviewed execution delta; declaration and source identity stay with Spec."""
    name: str
    version: str
    mode: str
    extra_source_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingAdapter:
    family: tuple[str, str]
    module: str
    artifact_adapter: str
    addons: tuple[TrainingAddon, ...]
    serialization_modules: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()

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
        if len(addons) != 1:
            raise ValueError("Execution requires one supported addon")
        return self.addon(addons[0]["name"], addons[0]["version"]).mode


TRAINING_ADAPTERS = MappingProxyType({
    ("neural_atom_k1", "2"): TrainingAdapter(
        ("neural_atom_k1", "2"), "molgap.k1_screen_training", "k1-screen-v1",
        (TrainingAddon("k1_joint_aggregation", "1", "ssma"),), ("molgap.pcqm_wedge",),
        ("src/molgap/k1_screen_training.py", "src/molgap/qm9_neural_atom.py",
         "src/molgap/qm9_local_hierarchy.py", "src/molgap/qm9_gape.py",
         "src/molgap/pcqm_gap_architecture.py", "src/molgap/gps.py", "src/molgap/pcqm_wedge.py")),
    ("gptrans_t", "1"): TrainingAdapter(
        ("gptrans_t", "1"), "molgap.gptrans_screen_workflow", "gptrans-v1",
        tuple(TrainingAddon(name, "1", name) for name in
              ("pair_prenorm", "centered_logits", "memory_value", "memory_message")),
        ("molgap.pcqm_wedge",),
        ("src/molgap/gptrans_screen_workflow.py", "src/molgap/pcqm_gptrans_v4.py",
         "src/molgap/gptrans.py", "src/molgap/gptrans_variants.py", "src/molgap/pcqm_wedge.py")),
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
    for arm in spec.to_dict()["arms"]:
        try:
            adapter = training_adapter(arm)
            supported[arm["arm_id"]] = {"mode": adapter.mode(arm), "trainer": adapter.module,
                                        "artifact_profile": adapter.artifact_adapter}
        except ValueError as exc:
            unsupported[arm["arm_id"]] = str(exc)
    registered = not unsupported and spec.to_dict()["platform"]["name"] == "kaggle"
    return {"spec_identity": spec.identity, "registered_training_arms": supported,
            "unsupported_training_arms": unsupported,
            "preparation_entry": "prepare-workflow" if registered else "prepare-release",
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


def build_family_recipe(family: tuple[str, str], *, addon: str | None = None,
                        source_idx_sha256: str, target_sha256: str,
                        addon_version: str = "1") -> dict:
    """Build owned constants before freezing the Spec; never read development rows."""
    adapter = TRAINING_ADAPTERS.get(family)
    if adapter is None:
        raise ValueError("No executable family recipe builder")
    mode = "reference" if addon is None else adapter.addon(addon, addon_version).mode
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
        results[arm["arm_id"]] = owner.validate_screen_recipe(spec, arm["arm_id"], _json(recipe_path))
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
        result.append({**job, "mode": adapter.mode(arm), "module": adapter.module,
                       "required_modules": [adapter.module, *adapter.serialization_modules]})
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
                           preflight_dir=None, resume_output=None):
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
    return runner(spec=spec, package_dir=package_dir,
        expected_package_identity=expected_package_identity, arm_id=job["arm_id"],
        mode=adapter.mode(arm), recipe_path=recipe, initial_state_path=initial,
        input_root=input_root, output=output, account=account,
        run_reference=run_reference, trajectory_id=job["trajectory_id"], **options)
