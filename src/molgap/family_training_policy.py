"""Typed lifecycle policy selected by the existing static addon registry."""
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module


@dataclass(frozen=True)
class FamilyModePolicy:
    construct: object = None
    mount: object = None  # (recipe, loaded_model) -> model; before optimizer/EMA creation.
    prepare_batch: object = None
    objective: object = None
    after_update: object = None
    attach_inputs: object = None
    input_sidecar_required: bool = False
    extra_development_weights: tuple[str, ...] = ()
    train_metric_unit: str = "normalized-Gap"
    process_timing: bool = False
    segment_passes: int | None = None
    chunk_passes: int = 10
    chunk_keep_count: int | None = None
    qualify_resources: object = None
    refresh_resources: object = None
    objective_name: str | None = None
    standalone_model: bool = False
    keep_geometry: bool = False
    extra_batch_fields: tuple[str, ...] = ()

    def __post_init__(self):
        if any(value is not None and not callable(value) for value in (
                self.construct, self.mount, self.prepare_batch, self.objective, self.after_update, self.attach_inputs,
                self.qualify_resources, self.refresh_resources)):
            raise TypeError("Policy hooks must be trusted registered callables")
        if type(self.standalone_model) is not bool or type(self.keep_geometry) is not bool:
            raise ValueError("Standalone/geometry policy flags must be exact booleans")
        if self.standalone_model and self.construct is None:
            raise ValueError("Standalone model policy requires an explicit construct hook")
        if type(self.extra_batch_fields) is not tuple or any(type(f) is not str or not f for f in self.extra_batch_fields):
            raise ValueError("Extra batch fields must be a tuple of nonempty strings")
        if self.train_metric_unit not in {"normalized-Gap", "eV"} or any(
                value not in {"live", "ema"} for value in self.extra_development_weights):
            raise ValueError("Unsupported explicit metric/state policy")
        if self.segment_passes is not None and (type(self.segment_passes) is not int or self.segment_passes < 1):
            raise ValueError("Invalid explicit durable segment")
        if type(self.chunk_passes) is not int or self.chunk_passes < 1:
            raise ValueError("Invalid retained chunk cadence")
        if self.chunk_keep_count is not None and (type(self.chunk_keep_count) is not int or self.chunk_keep_count < 1):
            raise ValueError("Invalid explicit retained chunk count")


GEOMETRY_BATCH_FIELDS = ("pos", "geometry_valid", "edge_distance", "wedge_angle_cos", "wedge_edge_ids")


def policy_batch_fields(policy: FamilyModePolicy) -> tuple[str, ...]:
    return (*(GEOMETRY_BATCH_FIELDS if policy.keep_geometry else ()), *policy.extra_batch_fields)


@lru_cache(maxsize=None)
def mode_policy(family, mode):
    from .experiment_execution import shared_training_adapters
    adapter = shared_training_adapters()[family]
    if mode in {"reference", "composed"}:
        return FamilyModePolicy()
    entries = [entry for entry in adapter.addons if entry.mode == mode]
    if not entries:
        raise ValueError("No registered family mode policy")
    hooks = {entry.policy_hook for entry in entries}
    if len(hooks) != 1:
        raise ValueError("Ambiguous family mode policy")
    hook = hooks.pop()
    if hook is None:
        return FamilyModePolicy()
    module, function = hook.split(":")
    policy = getattr(import_module(module), function)(mode)
    if type(policy) is not FamilyModePolicy:
        raise TypeError("Registered hook must return FamilyModePolicy")
    return policy
