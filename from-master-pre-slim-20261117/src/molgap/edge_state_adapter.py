"""Spec-bound EdgeState model construction, without training or evidence authority."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .experiment_spec import EDGE_STATE_BASE_LAYERS, ExperimentSpec, FAMILIES, FamilyContract
from .screen_policy import canonical_fingerprint

__all__ = ["EdgeStateAdapterMetadata", "edge_state_metadata", "build_edge_state_model"]


@dataclass(frozen=True)
class EdgeStateAdapterMetadata:
    family: FamilyContract
    variant: str
    num_layers: int
    addon: tuple[str, str] | None
    spec_identity: str
    arm_id: str
    arm_identity: str
    scientific_role: str
    source_module: str
    factory: str
    checkpoint_resume_implemented: bool = False
    runtime_evidence_statement: str = (
        "Model construction does not authenticate source/data/state declarations, "
        "implement training or checkpoint resume, or establish replay eligibility."
    )
    required_checkpoint_identity_fields: tuple[str, ...] = (
        "spec_identity", "arm_id", "arm_identity", "family.name", "family.version",
        "variant", "num_layers", "addon",
    )


def _resolve(spec: ExperimentSpec, arm_id: str) -> EdgeStateAdapterMetadata:
    if type(spec) is not ExperimentSpec:
        raise TypeError("Expected an ExperimentSpec, not a dict or custom provider")
    if type(arm_id) is not str:
        raise TypeError("arm_id must be a string")
    if set(vars(spec)) != {"_canonical_json"} or type(spec._canonical_json) is not str:
        raise ValueError("Invalid ExperimentSpec snapshot")
    validated = ExperimentSpec.from_json(spec._canonical_json)
    if validated.to_json() != spec._canonical_json or validated.identity != spec.identity:
        raise ValueError("Invalid ExperimentSpec canonical identity")
    arm = next((item for item in validated.to_dict()["arms"] if item["arm_id"] == arm_id), None)
    if arm is None:
        raise ValueError(f"Unknown arm: {arm_id}")
    key = (arm["family"]["name"], arm["family"]["version"])
    if key != ("edge_state_gps", "1"):
        raise ValueError("EdgeState adapter requires edge_state_gps family/version 1")

    variant, layers, addon = "edge_state", EDGE_STATE_BASE_LAYERS, None
    source_module = "molgap.edge_state_model_only_v1"
    addons = arm["addons"]
    if addons:
        if len(addons) != 1:
            raise ValueError("EdgeState architecture addons are mutually exclusive")
        extension = addons[0]
        addon = (extension["name"], extension["version"])
        if addon == ("edge_state_depth", "1"):
            variant = "edge_state_depth"
            layers = extension["config"]["num_layers"]
        elif addon == ("neural_atom_k1", "1"):
            variant = "neural_atom_k1"
            source_module = "molgap.qm9_neural_atom"
        else:
            raise ValueError("Unsupported EdgeState addon/version")
    return EdgeStateAdapterMetadata(
        family=FAMILIES[key], variant=variant, num_layers=layers, addon=addon,
        spec_identity=validated.identity, arm_id=arm_id,
        arm_identity=canonical_fingerprint(arm), scientific_role=arm["scientific_role"],
        source_module=source_module, factory=source_module + ".make_encoder",
    )


def edge_state_metadata(spec: ExperimentSpec, arm_id: str) -> EdgeStateAdapterMetadata:
    """Resolve architecture declarations without importing Torch or model code."""
    return _resolve(spec, arm_id)


def build_edge_state_model(
    spec: ExperimentSpec, arm_id: str, *, initial_state_path: Path | None = None,
):
    """Construct with caller-controlled RNG; state loading belongs to a trainer addon."""
    metadata = _resolve(spec, arm_id)
    if initial_state_path is not None:
        raise ValueError("EdgeState adapter does not implement initial-state loading or resume")
    if metadata.variant == "neural_atom_k1":
        from .qm9_neural_atom import make_encoder

        return make_encoder("neural_atom_k1")
    from .edge_state_model_only_v1 import make_encoder

    return make_encoder(metadata.num_layers)
