"""Typed lifecycle policy selected by the existing static addon registry."""
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module


@dataclass(frozen=True)
class FamilyModePolicy:
    construct: object = None
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
    qualify_resources: object = None
    refresh_resources: object = None
    objective_name: str | None = None

    def __post_init__(self):
        if any(value is not None and not callable(value) for value in (
                self.construct, self.prepare_batch, self.objective, self.after_update, self.attach_inputs,
                self.qualify_resources, self.refresh_resources)):
            raise TypeError("Policy hooks must be trusted registered callables")
        if self.train_metric_unit not in {"normalized-Gap", "eV"} or any(
                value not in {"live", "ema"} for value in self.extra_development_weights):
            raise ValueError("Unsupported explicit metric/state policy")
        if self.segment_passes is not None and (type(self.segment_passes) is not int or self.segment_passes < 1):
            raise ValueError("Invalid explicit durable segment")
        if type(self.chunk_passes) is not int or self.chunk_passes < 1:
            raise ValueError("Invalid retained chunk cadence")


@lru_cache(maxsize=None)
def mode_policy(family, mode):
    from .experiment_execution import TRAINING_ADAPTERS
    adapter = TRAINING_ADAPTERS[(family, "3" if family == "neural_atom_k1" else "2")]
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
