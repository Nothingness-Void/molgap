"""Strict K1 model construction; no runner, admission, or replay authority."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .experiment_spec import ExperimentSpec, FAMILIES, FamilyContract
from .screen_policy import canonical_fingerprint

__all__ = ["K1AdapterMetadata", "k1_metadata", "build_k1_model"]


@dataclass(frozen=True)
class K1AdapterMetadata:
    family: FamilyContract
    mode: str
    addon: tuple[str, str] | None
    spec_identity: str
    arm_id: str
    arm_identity: str
    scientific_role: str
    source_module: str
    factory: str
    architecture_base_factory: str = "molgap.qm9_local_hierarchy.make_encoder"
    checkpoint_resume_owner: str = "molgap.pcqm_k1_full_runner"
    checkpoint_resume_implemented: bool = False
    runtime_evidence_statement: str = (
        "Construction does not prove runtime evidence or replay eligibility. "
        "Declared source/data/state digests are not authenticated by this adapter."
    )
    required_checkpoint_identity_fields: tuple[str, ...] = (
        "spec_identity", "arm_id", "arm_identity", "family.name",
        "family.version", "mode", "addon",
    )


def _resolve(spec: ExperimentSpec, arm_id: str) -> K1AdapterMetadata:
    if type(spec) is not ExperimentSpec:
        raise TypeError("Expected an ExperimentSpec, not a dict or custom provider")
    if type(arm_id) is not str:
        raise TypeError("arm_id must be a string")
    # Reject instance-level method/provider injection before invoking any method.
    if set(vars(spec)) != {"_canonical_json"} or type(spec._canonical_json) is not str:
        raise ValueError("Invalid ExperimentSpec snapshot")
    validated = ExperimentSpec.from_json(spec.to_json())
    rebuilt = ExperimentSpec(spec.to_dict())
    if (rebuilt.identity != spec.identity
            or validated.to_json() != spec.to_json()):
        raise ValueError("Invalid ExperimentSpec canonical identity")
    arm = next((item for item in validated.to_dict()["arms"]
                if item["arm_id"] == arm_id), None)
    if arm is None:
        raise ValueError(f"Unknown arm: {arm_id}")
    key = (arm["family"]["name"], arm["family"]["version"])
    if key != ("neural_atom_k1", "1"):
        raise ValueError("K1 adapter requires neural_atom_k1 family/version 1")
    addons = arm["addons"]
    mode, addon = "neural_atom_k1", None
    source_module = "molgap.qm9_neural_atom"
    if addons:
        if (len(addons) != 1
                or (addons[0]["name"], addons[0]["version"]) != ("k1_pair_value", "1")
                or addons[0]["config"] != {}):
            raise ValueError("Unsupported K1 addon/version/config combination")
        mode = "neural_atom_k1_pair_token_value_decoupled"
        addon = ("k1_pair_value", "1")
        source_module = "molgap.k1_pair_token"
    return K1AdapterMetadata(
        family=FAMILIES[key], mode=mode, addon=addon,
        spec_identity=validated.identity, arm_id=arm_id,
        arm_identity=canonical_fingerprint(arm), scientific_role=arm["scientific_role"],
        source_module=source_module, factory=source_module + ".make_encoder",
    )


def k1_metadata(spec: ExperimentSpec, arm_id: str) -> K1AdapterMetadata:
    """Return detached declarations without importing model dependencies."""
    return _resolve(spec, arm_id)


def build_k1_model(
    spec: ExperimentSpec, arm_id: str, *, initial_state_path: Path | None = None,
):
    """Construct with caller-controlled RNG; checkpoint loading is unsupported.

    Initialization declarations are prospective. In particular, a frozen_state
    declaration does not cause loading: admission and state binding need a runner.
    """
    metadata = _resolve(spec, arm_id)
    if initial_state_path is not None:
        raise ValueError("K1 adapter does not implement initial-state loading or resume")
    if metadata.addon is None:
        from .qm9_neural_atom import make_encoder
    else:
        from .k1_pair_token import make_encoder
    return make_encoder(metadata.mode)
