"""Reviewed model/addon ABI for the normalized pure-2D graph screen.

The generic graph trainer owns optimization, durable output, and resume.  A
family contributes only a model factory and, when needed, small reviewed model
delta hooks.  Import strings are registry values, never values read from a
Spec or from a caller supplied configuration.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import re
from pathlib import Path
from types import ModuleType

HOOK_PATTERN = re.compile(
    r"molgap(?:\.[A-Za-z_][A-Za-z0-9_]*)+:[A-Za-z_][A-Za-z0-9_]*$"
)
MODEL_FACTORY_NAME = "make_model"
ADDON_HOOK_NAME = "apply_addon"


@dataclass(frozen=True)
class GraphModelMetadata:
    """Static identity returned to the diagnostic runner before construction."""

    family: object
    source_module: str
    model_factory: str
    addon_hooks: tuple[str, ...]


def _hook(value: str, *, function: str, label: str) -> tuple[ModuleType, object]:
    if type(value) is not str or HOOK_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a reviewed molgap module:function hook")
    module_name, function_name = value.split(":", 1)
    if function_name != function:
        raise ValueError(f"{label} must use :{function}")
    module = importlib.import_module(module_name)
    target = getattr(module, function_name, None)
    if not callable(target):
        raise ValueError(f"{label} is not callable")
    return module, target


def hook_source_path(value: str, *, function: str) -> Path:
    """Return the source path for a reviewed hook after importing its owner."""
    module, _ = _hook(value, function=function, label="hook")
    source = getattr(module, "__file__", None)
    if not isinstance(source, str):
        raise ValueError("Reviewed hook has no source file")
    path = Path(source).resolve()
    if path.suffix != ".py" or not path.is_file():
        raise ValueError("Reviewed hook source must be a regular Python file")
    return path


def normalized_hook_source_sha256(value: str, *, function: str) -> str:
    from .v4_runtime import normalized_source_sha256

    return normalized_source_sha256(hook_source_path(value, function=function))


def resolve_model_factory(adapter):
    """Resolve the registry-owned ``make_model(arm)`` hook."""
    value = getattr(adapter, "model_factory", None)
    if value is None:
        raise ValueError("Shared graph adapter has no reviewed model factory")
    _, factory = _hook(value, function=MODEL_FACTORY_NAME, label="model_factory")
    return factory


def _torch_module(value):
    import torch

    if not isinstance(value, torch.nn.Module):
        raise TypeError("Model factory/addon must return torch.nn.Module")
    for name, parameter in value.named_parameters():
        if parameter.is_floating_point() and parameter.dtype is not torch.float32:
            raise TypeError(f"Model parameter {name} is not FP32")
    for name, buffer in value.named_buffers():
        if buffer.is_floating_point() and buffer.dtype is not torch.float32:
            raise TypeError(f"Model buffer {name} is not FP32")
    return value


def _validate_factory_contract(arm: dict, factory: str) -> None:
    from .experiment_spec import FAMILIES

    family = arm.get("family")
    contract = FAMILIES.get(((family or {}).get("name"), (family or {}).get("version"))) if isinstance(family, dict) else None
    if contract is None:
        raise ValueError("Shared graph family has no static FamilyContract")
    module = factory.split(":", 1)[0]
    if contract.source_module != module:
        raise ValueError("Model factory module differs from FamilyContract.source_module")


def validate_factory_source(adapter, arm: dict) -> str:
    """Bind the model source to the arm's frozen base reference."""
    source_sha = normalized_hook_source_sha256(
        adapter.model_factory, function=MODEL_FACTORY_NAME
    )
    base = arm.get("base")
    if type(base) is not dict or base.get("sha256") != source_sha:
        raise ValueError("Model factory source differs from the frozen arm base")
    return source_sha


def _addon_hook(adapter, arm: dict, addon: dict):
    entry = adapter.addon(addon["name"], addon["version"])
    hook = entry.apply_hook
    if hook is None:
        raise ValueError("Shared graph addon has no reviewed apply hook")
    actual = normalized_hook_source_sha256(hook, function=ADDON_HOOK_NAME)
    if actual != addon["source_sha256"]:
        raise ValueError("Addon source differs from its frozen Spec digest")
    _, function = _hook(hook, function=ADDON_HOOK_NAME, label="addon.apply_hook")
    return function


def apply_reviewed_addons(adapter, arm: dict, model):
    """Apply ordered, bounded addon deltas from the reviewed registry."""
    model = _torch_module(model)
    for addon in arm.get("addons", ()):
        function = _addon_hook(adapter, arm, addon)
        result = function(model, deepcopy(addon["config"]))
        if result is not None:
            model = _torch_module(result)
        else:
            # In-place hooks are allowed, but they cannot bypass the same
            # dtype/device ABI gate as hooks returning a replacement module.
            model = _torch_module(model)
    return model


def build_model(adapter, arm: dict, *, initial_state_path: Path | None = None):
    """Build one model and optionally load the Spec-bound initial state."""
    _validate_factory_contract(arm, adapter.model_factory)
    validate_factory_source(adapter, arm)
    factory = resolve_model_factory(adapter)
    model = _torch_module(factory(deepcopy(arm)))
    model = apply_reviewed_addons(adapter, arm, model)
    if initial_state_path is not None:
        load_initial_state(model, Path(initial_state_path),
                           expected_state_sha256=arm["initialization"]["state_sha256"])
    return model


def graph_metadata(spec, arm_id: str) -> GraphModelMetadata:
    """Resolve a generic graph family without importing model tensors."""
    from .experiment_execution import training_adapter

    arm = next((item for item in spec.to_dict()["arms"] if item["arm_id"] == arm_id), None)
    if arm is None:
        raise ValueError(f"Unknown arm: {arm_id}")
    adapter = training_adapter(arm)
    if adapter.model_factory is None or adapter.module != "molgap.graph_screen_training":
        raise ValueError("Arm is not registered for shared graph training")
    _validate_factory_contract(arm, adapter.model_factory)
    module_name = adapter.model_factory.split(":", 1)[0]
    return GraphModelMetadata(
        family=spec.family_contract(arm_id), source_module=module_name,
        model_factory=adapter.model_factory,
        addon_hooks=tuple(
            adapter.addon(a["name"], a["version"]).apply_hook for a in arm["addons"]
        ),
    )


def build_graph_model(spec, arm_id: str):
    """Construct a generic model for adapter diagnostics.

    This intentionally does not load a frozen initialization file.  Training
    and release owners call :func:`build_model` with the staged artifact after
    the package/Spec identity has been checked.
    """
    from .experiment_execution import training_adapter

    arm = next((item for item in spec.to_dict()["arms"] if item["arm_id"] == arm_id), None)
    if arm is None:
        raise ValueError(f"Unknown arm: {arm_id}")
    adapter = training_adapter(arm)
    if adapter.model_factory is None or adapter.module != "molgap.graph_screen_training":
        raise ValueError("Arm is not registered for shared graph training")
    _validate_factory_contract(arm, adapter.model_factory)
    return build_model(adapter, arm)


def load_initial_state(model, path: Path, *, expected_state_sha256: str) -> dict:
    """Load a safe CPU tensor state and verify its canonical tensor digest."""
    from .v4_runtime import inspect_frozen_state_artifact, model_state_sha256, torch_load_compat

    path = Path(path).resolve()
    inspected = inspect_frozen_state_artifact(
        path, expected_state_sha256=expected_state_sha256
    )
    payload = torch_load_compat(path, map_location="cpu", weights_only=True)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("Frozen initialization has no tensor state")
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("Frozen initialization keys differ from the model factory") from exc
    if model_state_sha256(model) != expected_state_sha256:
        raise ValueError("Loaded initial model state differs from the frozen Spec")
    return inspected


def validate_model_output(output, *, rows: int, device=None):
    """Normalize a model's ``(N,)``/``(N,1)`` output to finite ``(N,)``."""
    import torch

    if not torch.is_tensor(output):
        raise ValueError("Model forward must return a tensor")
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    if output.ndim != 1 or output.shape[0] != rows:
        raise ValueError("Model output must contain one normalized Gap per graph")
    if device is not None and output.device != device:
        raise ValueError("Model output device differs from its input batch")
    if output.dtype is not torch.float32 or not bool(torch.isfinite(output).all()):
        raise ValueError("Model output must be finite FP32")
    return output


def forward_normalized_gap(model, batch, *, validate_batch=True):
    """Run the reviewed model ABI on one already staged graph batch."""
    if validate_batch:
        from .pcqm_topology import validate_ogb_gap_batch

        rows = validate_ogb_gap_batch(batch)
    else:
        rows = int(batch.y.reshape(-1).shape[0])
    output = model(batch)
    return validate_model_output(output, rows=rows, device=batch.y.device)


__all__ = [
    "ADDON_HOOK_NAME",
    "HOOK_PATTERN",
    "MODEL_FACTORY_NAME",
    "apply_reviewed_addons",
    "GraphModelMetadata",
    "build_model",
    "build_graph_model",
    "forward_normalized_gap",
    "graph_metadata",
    "hook_source_path",
    "load_initial_state",
    "normalized_hook_source_sha256",
    "resolve_model_factory",
    "validate_factory_source",
    "validate_model_output",
]
