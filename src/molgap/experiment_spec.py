"""Compatibility namespace; one shared parser consumes explicit catalog data."""
from dataclasses import dataclass

from .common.spec import (
    ExperimentSpec as _Spec, FamilyContract, AddonContract, AddonConfigField,
    SCHEMA_VERSION, SCHEMA_VERSION_V2, TERMINAL_PROTOCOL,
    EDGE_STATE_BASE_LAYERS, EDGE_STATE_MIN_LAYERS, EDGE_STATE_MAX_LAYERS,
    validate_addon_config, validate_experiment_spec as _validate, _arm as _validate_arm,
    _object, _text, _digest, _identifier, _integer, _choice, _reference, _list,
    _repo_path, _canonical, _unique_object,
)
from .experiment_catalog import FAMILIES, ADDONS


def _arm(arm):
    return _validate_arm(arm, families=FAMILIES, addons=ADDONS)


def validate_experiment_spec(payload):
    return _validate(payload, families=FAMILIES, addons=ADDONS)


@dataclass(frozen=True, init=False)
class ExperimentSpec(_Spec):
    """Existing import/constructor retained; catalog is not a machine adapter."""

    def __init__(self, payload):
        # Keep the wire-only snapshot checked by existing launch/model adapters.
        object.__setattr__(self, "_canonical_json", _canonical(validate_experiment_spec(payload)))

    def family_contract(self, arm_id):
        return self._family_contract(arm_id, FAMILIES)
