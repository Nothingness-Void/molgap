"""Explicit versioned baseline pins for the existing scientific input compiler."""
from copy import deepcopy
import json
import re

from .common.paths import validate_relative_path
from .common.spec import _digest, _identifier, _object, _unique_object
from .research_memory.trace import file_digest


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
    if preset["format"] != "molgap-scientific-baseline-preset-v1":
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
        _object(reference, "path sha256 arm_id", "baseline preset reference")
        _identifier(reference["arm_id"], "baseline preset reference arm")
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
