"""Static family differences for the shared, metadata-only output inspector.

These profiles describe the new output protocol, not every historical runner.
Registering one does not provide a trainer or authorize execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

# Retained Session observations use the same allowlist in publication,
# inspection and selective platform retrieval. Historical outputs omit them.
SESSION_OPTIONAL_ARTIFACTS = frozenset({"role_audit", "runtime_certificate", "runtime_provenance", "final_predictions"})

@dataclass(frozen=True)
class ArtifactAdapter:
    name: str
    family: tuple[str, str] | None
    resume_keys: tuple[str, ...]
    ema_required: bool
    protocol: str = "screen"


ADAPTERS = MappingProxyType({
    "modern2d-formal-session-v1": ArtifactAdapter("modern2d-formal-session-v1", None,
        ("model", "optimizer", "scheduler", "rng_state"), False, "family-session"),
    "modern2d-session-preview-v1": ArtifactAdapter("modern2d-session-preview-v1", None,
        ("model", "optimizer", "scheduler", "rng_state"), False, "family-session"),
    "modern2d-cpu-v1": ArtifactAdapter("modern2d-cpu-v1", None,
        ("model", "optimizer", "scheduler", "rng_state"), False, "family-session"),
    "graph-screen-v1": ArtifactAdapter("graph-screen-v1", None,
                                      ("model", "optimizer", "scheduler", "rng_state"), False),
    "k1-session-v1": ArtifactAdapter("k1-session-v1", ("neural_atom_k1", "3"),
        ("model", "optimizer", "scheduler", "rng_state"), False, "family-session"),
    "gptrans-session-v1": ArtifactAdapter("gptrans-session-v1", ("gptrans_t", "2"),
        ("model", "optimizer", "scheduler", "rng_state"), False, "family-session"),
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
        if name == "modern2d-formal-session-v1":
            if (owner is None or not owner.formal_training or owner.cpu_synthetic_only
                    or owner.integration_preview or owner.module != "molgap.family_training_runner"
                    or owner.artifact_adapter != name):
                raise ValueError("Formal artifact profile requires its separately admitted Modern2D interface")
            return adapter
        if name == "modern2d-session-preview-v1":
            if (owner is None or not owner.integration_preview or owner.cpu_synthetic_only
                    or owner.module != "molgap.family_training_runner" or owner.artifact_adapter != name):
                raise ValueError("Preview artifact profile requires its explicit Modern2D session interface")
            return adapter
        if name == "modern2d-cpu-v1":
            if owner is None or not owner.cpu_synthetic_only or owner.artifact_adapter != name:
                raise ValueError("CPU artifact profile requires its static synthetic family")
            return adapter
        if (owner is None or owner.artifact_adapter != name
                or owner.module != "molgap.graph_screen_training" or not owner.model_factory):
            raise ValueError("Generic artifact profile requires a reviewed graph training registration")
    elif adapter.family != family:
        raise ValueError("Artifact adapter/family mismatch")
    return adapter
