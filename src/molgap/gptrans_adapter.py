"""ExperimentSpec-to-GPTrans-T construction only; no runner or admission gate.

The frozen factory owns initialization and attention replacement. Construction
does not verify declared source/data digests, seed the RNG, authorize roles, or
implement checkpoint/resume, terminal descriptors, or RML wiring.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .experiment_spec import ADDONS, FAMILIES, ExperimentSpec, FamilyContract

__all__ = ["GPTransAdapterMetadata", "gptrans_metadata", "build_gptrans_model"]


@dataclass(frozen=True)
class GPTransAdapterMetadata:
    """Declarations a future runner must bind and enforce, not a certificate."""

    family: FamilyContract
    variant: str
    spec_identity: str
    arm_id: str
    factory: str = "molgap.pcqm_gptrans_v4._make_model"
    checkpoint_resume_owner: str = "existing trainer"
    checkpoint_resume_implemented: bool = False
    # Parameter-free variants share tensor keys/shapes, so tensor compatibility
    # alone cannot establish that a resumed run has the same model semantics.
    required_checkpoint_identity_fields: tuple[str, ...] = (
        "spec_identity", "arm_id", "family.name", "family.version", "variant",
    )


def _resolve(spec: ExperimentSpec, arm_id: str):
    """Resolve only a validated spec, never author-supplied dispatch objects."""
    if type(spec) is not ExperimentSpec:
        raise TypeError("Expected an ExperimentSpec, not a dict or custom provider")
    if type(arm_id) is not str:
        raise TypeError("arm_id must be a string")
    # Revalidate the immutable snapshot at this execution boundary; subclasses
    # and manually forged snapshots must not bypass the Stage 1 registry.
    validated = ExperimentSpec.from_json(spec.to_json())
    arm = next((item for item in validated.to_dict()["arms"]
                if item["arm_id"] == arm_id), None)
    if arm is None:
        raise ValueError(f"Unknown arm: {arm_id}")
    key = (arm["family"]["name"], arm["family"]["version"])
    if key != ("gptrans_t", "1"):
        raise ValueError("GPTrans adapter requires gptrans_t family/version 1")
    addons = arm["addons"]
    if len(addons) > 1:
        raise ValueError("GPTrans attention addons are mutually exclusive")
    variant = "reference"
    if addons:
        addon_key = (addons[0]["name"], addons[0]["version"])
        if addon_key not in (("pair_prenorm", "1"), ("centered_logits", "1"),
                             ("rwse16", "1"), ("rwse16_local_edge", "1")):
            raise ValueError("Unsupported GPTrans addon/version")
        if ADDONS[addon_key].family != key[0]:
            raise ValueError("Incompatible GPTrans addon family")
        variant = addon_key[0]
    return GPTransAdapterMetadata(
        family=FAMILIES[key], variant=variant,
        spec_identity=validated.identity, arm_id=arm_id,
    )


def gptrans_metadata(spec: ExperimentSpec, arm_id: str) -> GPTransAdapterMetadata:
    """Look up family and resume identity requirements without model imports.

    A future runner must persist/compare these identities in addition to the
    existing trainer's checkpoint guards. This function implements neither.
    The frozen V4 runner still rejects architecture variants.
    """
    return _resolve(spec, arm_id)


def build_gptrans_model(
    spec: ExperimentSpec,
    arm_id: str,
    *,
    initial_state_path: Path | None = None,
):
    """Build via the frozen factory, including its initial-state validation.

    With no path, consume the caller's RNG exactly as the existing factory
    does. A supplied path is validated/loaded before applying the addon by
    that factory. It is an initial-state artifact, not a resume checkpoint.
    Spec digest/initialization declarations still require runner admission;
    this constructor alone does not certify that they match runtime bytes.
    """
    metadata = _resolve(spec, arm_id)
    from .pcqm_gptrans_v4 import _make_model

    return _make_model(initial_state_path=initial_state_path, variant=metadata.variant)
