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
    chunk_keep_count: int | None = None
    qualify_resources: object = None
    refresh_resources: object = None
    observe_pass: object = None
    objective_name: str | None = None

    def __post_init__(self):
        if any(value is not None and not callable(value) for value in (
                self.construct, self.prepare_batch, self.objective, self.after_update, self.attach_inputs,
                self.qualify_resources, self.refresh_resources, self.observe_pass)):
            raise TypeError("Policy hooks must be trusted registered callables")
        if self.train_metric_unit not in {"normalized-Gap", "eV"} or any(
                value not in {"live", "ema"} for value in self.extra_development_weights):
            raise ValueError("Unsupported explicit metric/state policy")
        if self.segment_passes is not None and (type(self.segment_passes) is not int or self.segment_passes < 1):
            raise ValueError("Invalid explicit durable segment")
        if type(self.chunk_passes) is not int or self.chunk_passes < 1:
            raise ValueError("Invalid retained chunk cadence")
        if self.chunk_keep_count is not None and (type(self.chunk_keep_count) is not int or self.chunk_keep_count < 1):
            raise ValueError("Invalid explicit retained chunk count")


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


def recipe_policy(recipe):
    """Resolve lifecycle hooks from reviewed addon declarations, never JSON imports."""
    if recipe.mode != "composed":
        return mode_policy(recipe.family, recipe.mode)
    from .experiment_execution import shared_training_adapters
    adapter = shared_training_adapters()[recipe.family]
    hooks = {adapter.addon(addon["name"], addon["version"]).policy_hook
             for addon in recipe.addons}
    hooks.discard(None)
    if not hooks:
        return FamilyModePolicy()
    if len(hooks) != 1:
        raise ValueError("Composed addons require one unambiguous reviewed lifecycle policy")
    module, function = hooks.pop().split(":")
    policy = getattr(import_module(module), function)("composed")
    if type(policy) is not FamilyModePolicy:
        raise TypeError("Registered hook must return FamilyModePolicy")
    return policy
