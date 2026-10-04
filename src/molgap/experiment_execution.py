"""Reviewed family execution interfaces, separate from declarative Spec support.

Only this registry selects executable owners. No caller-provided imports or
callbacks are accepted. A supported recipe is still not execution authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import MappingProxyType

from .experiment_spec import ExperimentSpec, SCHEMA_VERSION_V2


@dataclass(frozen=True)
class TrainingAdapter:
    family: tuple[str, str]
    module: str
    artifact_adapter: str
    addon_modes: tuple[tuple[str, str], ...]
    serialization_modules: tuple[str, ...] = ()

    def mode(self, arm: dict) -> str:
        addons = arm["addons"]
        if not addons:
            return "reference"
        if len(addons) != 1 or addons[0]["version"] != "1":
            raise ValueError("Execution requires one supported addon")
        modes = dict(self.addon_modes)
        if addons[0]["name"] not in modes:
            raise ValueError("Addon has no executable family adapter")
        return modes[addons[0]["name"]]


TRAINING_ADAPTERS = MappingProxyType({
    ("neural_atom_k1", "2"): TrainingAdapter(
        ("neural_atom_k1", "2"), "molgap.k1_screen_training", "k1-screen-v1",
        (("k1_joint_aggregation", "ssma"),), ("molgap.pcqm_wedge",)),
    ("gptrans_t", "1"): TrainingAdapter(
        ("gptrans_t", "1"), "molgap.gptrans_screen_workflow", "gptrans-v1",
        (("pair_prenorm", "pair_prenorm"), ("centered_logits", "centered_logits"),
         ("memory_value", "memory_value"), ("memory_message", "memory_message")), ("molgap.pcqm_wedge",)),
})


def training_adapter(arm: dict) -> TrainingAdapter:
    key = (arm["family"]["name"], arm["family"]["version"])
    try:
        adapter = TRAINING_ADAPTERS[key]
    except KeyError as exc:
        raise ValueError(f"Family {key} has declaration/model support only; no training adapter") from exc
    adapter.mode(arm)
    return adapter


def build_family_recipe(family: tuple[str, str], *, addon: str | None = None,
                        source_idx_sha256: str, target_sha256: str) -> dict:
    """Build owned constants before freezing the Spec; never read development rows."""
    adapter = TRAINING_ADAPTERS.get(family)
    if adapter is None:
        raise ValueError("No executable family recipe builder")
    mode = "reference" if addon is None else dict(adapter.addon_modes).get(addon)
    if mode is None:
        raise ValueError("No executable recipe for this addon")
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
                           target_identity=None):
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
    options = {} if phase == "preflight" else {"preflight_dir": output}
    result = runner(spec=spec, package_dir=package_dir,
        expected_package_identity=expected_package_identity, arm_id=job["arm_id"],
        mode=adapter.mode(arm), recipe_path=recipe, initial_state_path=initial,
        input_root=input_root, output=output, account=account,
        run_reference=run_reference, trajectory_id=job["trajectory_id"], **options)
    if phase == "train" and target_identity is not None:
        # Retained family trainers keep their frozen completion call. The owning
        # inspection applies the explicit contract encoding at dispatch closure.
        from .experiment_family_workflow import inspect_output
        normalized = Path(output) / "normalized" if adapter.artifact_adapter == "gptrans-v1" else Path(output)
        import json
        expected = json.loads(recipe.read_text(encoding="utf-8"))["acceptance_requirements"]
        return inspect_output(normalized, context=context, expected=expected, target_identity=target_identity)
    return result
