"""Explicit baseline pins and editable drafts for scientific inputs."""
from copy import deepcopy
import json
import re
from pathlib import Path

from .common.io import _safe_local, publish_immutable_bytes
from .common.paths import validate_relative_path
from .common.spec import _digest, _identifier, _object, _unique_object
from .experiment_launch import canonical_json
from .research_memory.trace import file_digest


_PRESET_FORMAT = "molgap-scientific-baseline-preset-v1"
_INPUT_FORMAT = "molgap-scientific-experiment-v1"


def _read(path):
    path = Path(path).absolute()
    _safe_local(path)
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _pin(root, value, label):
    root = Path(root).absolute()
    _safe_local(root)
    path = value if Path(value).is_absolute() else root / value
    path = Path(path).absolute()
    _safe_local(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    try:
        name = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the checkout") from exc
    validate_relative_path(name)
    return {"path": name, "sha256": file_digest(path)}


def build_baseline_preset(repo_root, name, family, recipe, data_spec,
                          data_arm_id, target_manifest, initial_state,
                          reference=None, reference_arm_id=None, version="v1"):
    """Derive an existing baseline preset's pins from retained files."""
    root = Path(repo_root).absolute()
    _safe_local(root)
    if type(family) is not tuple or len(family) != 2:
        raise ValueError("family must be one catalog (name, version) tuple")
    from .experiment_catalog import FAMILIES
    if family not in FAMILIES:
        raise ValueError(f"Family {family!r} is not present in the catalog")
    _identifier(name, "baseline preset name")
    if type(version) is not str or re.fullmatch(r"v[1-9][0-9]*", version) is None:
        raise ValueError("Baseline preset requires an immutable explicit version")
    _identifier(data_arm_id, "baseline data arm_id")

    recipe_pin = _pin(root, recipe, "baseline recipe")
    data_pin = _pin(root, data_spec, "baseline data specification")
    target_pin = _pin(root, target_manifest, "baseline target manifest")
    from .family_training_runner import read_recipe
    recipe_record, parsed_recipe = read_recipe(root / recipe_pin["path"])
    from .v4_runtime import inspect_frozen_state_artifact
    state_sha = parsed_recipe.initialization_sha256
    state_path = Path(initial_state)
    if not state_path.is_absolute():
        state_path = root / state_path
    state_path = state_path.absolute()
    _safe_local(state_path)
    inspect_frozen_state_artifact(state_path, expected_state_sha256=state_sha)
    target = _read(root / target_pin["path"])
    if (type(target) is not dict or target.get("target_encoding") not in {
            "little-endian-float32-contiguous-raw-bytes",
            "little-endian-float64-contiguous-raw-bytes"}
            or target.get("development_target_sha256") !=
            recipe_record["acceptance_requirements"]["target_sha256"]):
        raise ValueError("Baseline target manifest differs from the recipe")
    from .experiment_spec import ExperimentSpec
    source = ExperimentSpec(_read(root / data_pin["path"])).to_dict()
    arm = next((item for item in source["arms"] if item["arm_id"] == data_arm_id), None)
    if arm is None or arm["family"] != {"name": family[0], "version": family[1]}:
        raise ValueError("Baseline data arm differs from the selected catalog family")
    if arm["training"]["recipe"]["sha256"] != recipe_pin["sha256"]:
        raise ValueError("Baseline data arm is not bound to the selected recipe")
    if arm["initialization"]["state_sha256"] != state_sha:
        raise ValueError("Baseline data arm is not bound to the selected initialization")

    reference_pin = None
    if reference is not None:
        if reference_arm_id is None:
            raise ValueError("baseline reference_arm_id is required with reference")
        _identifier(reference_arm_id, "baseline reference_arm_id")
        reference_pin = {**_pin(root, reference, "baseline reference"),
                         "arm_id": reference_arm_id}
    elif reference_arm_id is not None:
        raise ValueError("baseline reference requires a reference path")
    return {"format": _PRESET_FORMAT, "name": name, "version": version,
            "model": {"family": family[0], "version": family[1]},
            "recipe": recipe_pin, "data": {**data_pin, "arm_id": data_arm_id},
            "target_manifest": target_pin, "initialization_state_sha256": state_sha,
            "reference": reference_pin}


def export_baseline_preset(output, preset):
    """Atomically publish one already-built baseline preset."""
    if type(preset) is not dict or preset.get("format") != _PRESET_FORMAT:
        raise ValueError("Expected a molgap scientific baseline preset")
    path = Path(output).absolute()
    _safe_local(path)
    return publish_immutable_bytes(path, canonical_json(preset).encode("utf-8"))


def build_scientific_input_template(run_directory, repo_root, experiment_id,
                                    logical_run_id, attempt_id="v1",
                                    machine_profile=None):
    """Copy the frozen package's human declaration into an editable draft."""
    run = Path(run_directory).absolute()
    _safe_local(run)
    inputs, package = run / "inputs", run / "prepared/package"
    workflow = _read(inputs / "workflow_plan.json")
    original = _read(inputs / "experiment_spec.json")
    if original.get("format") == _INPUT_FORMAT:
        raise ValueError("Frozen run input sidecar is not an executable Spec")
    _identifier(experiment_id, "experiment_id")
    _identifier(logical_run_id, "logical_run_id")
    _identifier(attempt_id, "attempt_id")
    if logical_run_id == original.get("logical_run_id"):
        raise ValueError("Template requires a fresh logical_run_id")
    _safe_local(Path(repo_root).absolute())

    from .experiment_package import _PackageVerification, read_packaged_text, verify_experiment_source_package
    verification = _PackageVerification()
    manifest = verify_experiment_source_package(package, _verification=verification)
    from .experiment_spec import ExperimentSpec
    spec_identity = ExperimentSpec(original).identity
    if (workflow.get("spec_identity") != spec_identity
            or manifest.get("spec_identity") != spec_identity):
        raise ValueError("Frozen Spec, workflow, and source package identities differ")
    configuration = manifest.get("configuration_source") or workflow.get("configuration_source")
    if type(configuration) is not dict or type(configuration.get("path")) is not str:
        raise ValueError("Frozen package lacks the original scientific configuration")
    draft = json.loads(read_packaged_text(package, configuration["path"],
                                          _verification=verification),
                       object_pairs_hook=_unique_object)
    if draft.get("format") != _INPUT_FORMAT:
        raise ValueError("Frozen configuration is not a scientific input")
    draft = deepcopy(draft)
    draft.update(experiment_id=experiment_id, logical_run_id=logical_run_id,
                 attempt_id=attempt_id,
                 monitor_paths=[f"experiments/{experiment_id}/STATUS.md"],
                 decision_ref=f"experiments/{experiment_id}/decision.md")
    if machine_profile is not None:
        draft["machine_profile"] = str(machine_profile)
    for pointer in draft["monitor_paths"]:
        validate_relative_path(pointer)
    validate_relative_path(draft["decision_ref"])
    for field in ("authority", "authority_ref", "role_ref"):
        draft.pop(field, None)
    return draft


def export_scientific_input_template(output, template, *, frozen_run=None):
    """Publish a fresh template outside the source frozen run."""
    if type(template) is not dict or template.get("format") != _INPUT_FORMAT:
        raise ValueError("Expected a molgap scientific input template")
    path = Path(output).absolute()
    _safe_local(path)
    if path.exists():
        raise ValueError("Scientific input template output must be fresh")
    if frozen_run is not None and path.is_relative_to(Path(frozen_run).absolute()):
        raise ValueError("Scientific input template must be outside the frozen run")
    return publish_immutable_bytes(path, canonical_json(template).encode("utf-8"))


def expand_baseline_preset(item, *, read_local):
    """Expand retained pins; state locations, platform and authority stay explicit."""
    def pinned(pointer):
        _object(pointer, "path sha256", "baseline preset pin")
        validate_relative_path(pointer["path"])
        _digest(pointer["sha256"], "baseline preset digest")
        path = read_local(pointer["path"])
        if file_digest(path) != pointer["sha256"]:
            raise ValueError("Baseline preset pinned bytes changed")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))

    result = deepcopy(item)
    preset = pinned(result.pop("preset_ref"))
    _object(preset, "format name version model recipe data target_manifest initialization_state_sha256 reference",
            "baseline preset")
    if preset["format"] != _PRESET_FORMAT:
        raise ValueError("Unknown baseline preset format")
    _identifier(preset["name"], "baseline preset name")
    if type(preset["version"]) is not str or re.fullmatch(r"v[1-9][0-9]*", preset["version"]) is None:
        raise ValueError("Baseline preset requires an immutable explicit version")
    _object(preset["model"], "family version", "baseline preset model")
    _digest(preset["initialization_state_sha256"], "baseline preset initialization")
    pinned(preset["recipe"])
    data = preset["data"]
    _object(data, "path sha256 arm_id", "baseline preset data")
    _identifier(data["arm_id"], "baseline preset data arm")
    pinned({key: data[key] for key in ("path", "sha256")})
    target = pinned(preset["target_manifest"])
    if type(target) is not dict or target.get("target_encoding") not in {
            "little-endian-float32-contiguous-raw-bytes", "little-endian-float64-contiguous-raw-bytes"}:
        raise ValueError("Baseline preset requires a frozen target encoding")
    _digest(target.get("development_target_sha256"), "baseline preset target")
    expanded = {"recipe_ref": preset["recipe"]["path"],
                "data_ref": {"spec_ref": data["path"], "arm_id": data["arm_id"]}}
    if preset["reference"] is not None:
        reference = preset["reference"]
        _object(reference, "path sha256 arm_id", "baseline reference")
        _identifier(reference["arm_id"], "baseline reference arm")
        pinned({key: reference[key] for key in ("path", "sha256")})
        expanded["reference_ref"] = {"plan_ref": reference["path"], "arm_id": reference["arm_id"]}
    for key, value in expanded.items():
        if key in result and result[key] != value:
            raise ValueError("Scientific input cannot override a frozen baseline preset binding")
        result[key] = value
    model = result.get("model", {})
    if type(model) is not dict or set(model) - {"family", "version", "addons"}:
        raise ValueError("Preset model accepts only family/version and addon declarations")
    for key, value in preset["model"].items():
        if key in model and model[key] != value:
            raise ValueError("Scientific model differs from the frozen baseline preset")
    result["model"] = {**preset["model"], "addons": model.get("addons", [])}
    if "initial_state_ref" not in result:
        raise ValueError("Baseline preset requires an explicit local initial_state_ref")
    return result, {**preset, "target": target}
