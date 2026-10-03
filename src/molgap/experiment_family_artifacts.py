"""Static family differences for the shared, metadata-only output inspector.

These profiles describe the new output protocol, not every historical runner.
Registering one does not provide a trainer or authorize execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ArtifactAdapter:
    name: str
    family: tuple[str, str] | None
    resume_keys: tuple[str, ...]
    ema_required: bool


ADAPTERS = MappingProxyType({
    "graph-screen-v1": ArtifactAdapter("graph-screen-v1", None,
                                      ("model", "optimizer", "scheduler", "rng_state"), False),
    "k1-screen-v1": ArtifactAdapter("k1-screen-v1", ("neural_atom_k1", "2"),
                                    ("model", "optimizer", "scheduler", "rng_state"), False),
    "edge-state-v1": ArtifactAdapter("edge-state-v1", ("edge_state_gps", "1"),
                                     ("model", "optimizer", "scheduler", "rng_state"), False),
    "k1-v1": ArtifactAdapter("k1-v1", ("neural_atom_k1", "1"),
                             ("model", "optimizer", "scheduler", "rng_state"), False),
    "gptrans-v1": ArtifactAdapter("gptrans-v1", ("gptrans_t", "1"),
                                  ("model", "ema", "optimizer", "rng_state"), True),
})


def artifact_adapter(name: str, family: tuple[str, str]) -> ArtifactAdapter:
    try:
        adapter = ADAPTERS[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unsupported artifact adapter: {name}") from exc
    if adapter.family is None:
        from .experiment_execution import TRAINING_ADAPTERS
        owner = TRAINING_ADAPTERS.get(family)
        if (owner is None or owner.artifact_adapter != name
                or owner.module != "molgap.graph_screen_training" or not owner.model_factory):
            raise ValueError("Generic artifact profile requires a reviewed graph training registration")
    elif adapter.family != family:
        raise ValueError("Artifact adapter/family mismatch")
    return adapter
