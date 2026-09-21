"""Synthetic declaration tests; no protected data or model imports."""
import copy
import json

import pytest

from molgap.experiment_spec import (
    ExperimentSpec, FAMILIES, SCHEMA_VERSION, TERMINAL_PROTOCOL,
)


def ref(name):
    return {"name": name, "version": "1", "sha256": "a" * 64}


def addon(name):
    return {"name": name, "version": "1", "config": {}, "source_sha256": "b" * 64}


def arm(family="gptrans_t"):
    contract = FAMILIES[(family, "1")]
    return {
        "arm_id": family, "scientific_role": "reference",
        "family": {"name": family, "version": "1"}, "base": ref("synthetic-base"),
        "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": "c" * 64},
        "data": {
            "dataset": ref("pcqm4mv2"), "split": ref("synthetic-split"),
            "roles": [
                {"role": role, "membership_sha256": "d" * 64,
                 "row_order_sha256": "e" * 64, "usage_sha256": "f" * 64}
                for role in contract.roles
            ],
            "feature_schema": contract.feature_schema, "feature_sha256": "a" * 64,
            "target": "pcqm4mv2-gap-eV-direct",
        },
        "training": {
            "recipe": ref(contract.recipe), "overrides": {},
            "objective": ref("normalized-gap-l1"), "sampler": ref(contract.sampler),
            "transform": ref(contract.transform),
        },
        "addons": [], "addon_semantics": "baseline",
    }


@pytest.fixture
def payload():
    return {
        "schema_version": SCHEMA_VERSION, "experiment_id": "synthetic-question",
        "logical_run_id": "synthetic-run", "arms": [arm(), arm("neural_atom_k1")],
        "platform": {
            "name": "local", "accelerator": "synthetic-device", "device_count": 1,
            "cpu_cores": 4, "memory_gib": 32, "atomic_checkpoints": True,
            "retrievable_chunks": True,
        },
        "prospective": {
            "trajectory_id": "synthetic-trajectory", "hypothesis": "A declared question",
            "cheapest_falsifier": "A synthetic diagnostic", "stop_rule": "Stop at budget",
            "budget_sha256": "a" * 64,
        },
        "evidence": {"policy": ref("molgap-v5"), "required_artifacts": [
            "v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact",
        ]},
        "terminal_protocol": TERMINAL_PROTOCOL,
    }


def test_canonical_round_trip_and_detachment(payload):
    spec = ExperimentSpec(payload)
    assert ExperimentSpec.from_json(spec.to_json()) == spec
    assert ExperimentSpec(dict(reversed(list(payload.items())))).identity == spec.identity
    assert spec.family_contract("neural_atom_k1").source_module == "molgap.qm9_neural_atom"
    payload["logical_run_id"] = "changed"
    exported = spec.to_dict()
    exported["arms"].clear()
    assert spec.to_dict()["logical_run_id"] == "synthetic-run"
    assert len(spec.to_dict()["arms"]) == 2


@pytest.mark.parametrize("device_count", [1, 2, 4])
def test_positive_device_count_declaration_accepted(payload, device_count):
    """Schema acceptance for two synthetic arms, not real device execution."""
    payload["platform"]["device_count"] = device_count
    spec = ExperimentSpec(payload)
    exported = ExperimentSpec.from_json(spec.to_json()).to_dict()
    assert exported["platform"]["device_count"] == device_count
    assert len(exported["arms"]) == 2


@pytest.mark.parametrize("device_count", [0, -1, 1.0, 2.5, "2", True, False, None, [], {}])
def test_invalid_device_count_declaration_rejected(payload, device_count):
    payload["platform"]["device_count"] = device_count
    with pytest.raises(ValueError, match=r"platform\.device_count: expected integer >= 1"):
        ExperimentSpec(payload)


@pytest.mark.parametrize("location", [
    (), ("arms", 0), ("arms", 0, "family"), ("arms", 0, "base"),
    ("arms", 0, "initialization"), ("arms", 0, "data"),
    ("arms", 0, "data", "roles", 0), ("arms", 0, "training"),
    ("arms", 0, "training", "recipe"), ("platform",), ("prospective",), ("evidence",),
])
@pytest.mark.parametrize("field", ["unexpected", "replay_ready", "ready_for_desktop"])
def test_unknown_fields_and_authority_rejected(payload, location, field):
    target = payload
    for key in location:
        target = target[key]
    target[field] = True
    with pytest.raises(ValueError):
        ExperimentSpec(payload)


@pytest.mark.parametrize("field,value", [("name", "unknown"), ("version", "99")])
def test_unknown_family(payload, field, value):
    payload["arms"][0]["family"][field] = value
    with pytest.raises(ValueError, match="Unknown family"):
        ExperimentSpec(payload)


@pytest.mark.parametrize("extensions", [
    [addon("unknown")], [dict(addon("pair_prenorm"), version="99")],
    [addon("pair_prenorm"), addon("centered_logits")],
    [addon("pair_prenorm"), addon("pair_prenorm")],
])
def test_unknown_or_incompatible_addons(payload, extensions):
    payload["arms"][0].update(addons=extensions, addon_semantics="ordered")
    with pytest.raises(ValueError):
        ExperimentSpec(payload)


def test_addon_family_and_config(payload):
    payload["arms"][1].update(addons=[addon("pair_prenorm")], addon_semantics="ordered")
    with pytest.raises(ValueError, match="Incompatible"):
        ExperimentSpec(payload)
    payload["arms"].pop()
    payload["arms"][0].update(addons=[addon("pair_prenorm")], addon_semantics="ordered")
    ExperimentSpec(payload)
    payload["arms"][0]["addons"][0]["config"]["replay_ready"] = True
    with pytest.raises(ValueError):
        ExperimentSpec(payload)


@pytest.mark.parametrize("role", ["official_validation", "test_dev", "test_challenge", "unknown"])
def test_invalid_role(payload, role):
    payload["arms"][0]["data"]["roles"][0]["role"] = role
    with pytest.raises(ValueError):
        ExperimentSpec(payload)


def test_empty_addon_has_explicit_baseline_semantics(payload):
    spec = ExperimentSpec.from_json(ExperimentSpec(payload).to_json())
    assert spec.to_dict()["arms"][0]["addons"] == []
    assert spec.to_dict()["arms"][0]["addon_semantics"] == "baseline"
    payload["arms"][0]["addon_semantics"] = "ordered"
    with pytest.raises(ValueError):
        ExperimentSpec(payload)


@pytest.mark.parametrize("path,value", [
    (("logical_run_id",), "different-run"),
    (("arms", 0, "base", "sha256"), "b" * 64),
    (("arms", 0, "initialization", "state_sha256"), "b" * 64),
    (("arms", 0, "data", "split", "sha256"), "b" * 64),
    (("arms", 0, "training", "recipe", "sha256"), "b" * 64),
    (("arms", 0, "training", "objective", "sha256"), "b" * 64),
    (("arms", 0, "training", "sampler", "sha256"), "b" * 64),
    (("arms", 0, "training", "transform", "sha256"), "b" * 64),
    (("platform", "memory_gib"), 64),
    (("platform", "device_count"), 2),
    (("prospective", "stop_rule"), "Different stop rule"),
])
def test_meaningful_changes_change_identity(payload, path, value):
    original = ExperimentSpec(payload).identity
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert ExperimentSpec(payload).identity != original


def test_arm_order_and_addon_identity(payload):
    original = ExperimentSpec(payload).identity
    payload["arms"].reverse()
    assert ExperimentSpec(payload).identity != original
    payload["arms"].reverse()
    payload["arms"][0].update(addons=[addon("pair_prenorm")], addon_semantics="ordered")
    extended = ExperimentSpec(payload).identity
    assert extended != original
    payload["arms"][0]["addons"][0]["name"] = "centered_logits"
    assert ExperimentSpec(payload).identity != extended


@pytest.mark.parametrize("override", ["seed", "learning_rate", "epochs", "replay_ready"])
def test_no_unapproved_overrides(payload, override):
    payload["arms"][0]["training"]["overrides"][override] = 1
    with pytest.raises(ValueError):
        ExperimentSpec(payload)


@pytest.mark.parametrize("mutation", [
    lambda p: p["arms"][0]["initialization"].update(seed=43),
    lambda p: p["arms"][0]["initialization"].update(seed=True),
    lambda p: p["platform"].update(device_count=True),
    lambda p: p["platform"].update(atomic_checkpoints=False),
    lambda p: p["arms"].append(copy.deepcopy(p["arms"][0])),
    lambda p: p["arms"][0]["data"]["roles"].pop(),
    lambda p: p["evidence"]["required_artifacts"].pop(),
    lambda p: p.update(schema_version="unknown"),
    lambda p: p.update(terminal_protocol="unknown"),
    lambda p: p["arms"][0]["training"]["objective"].update(name="unknown"),
    lambda p: p["arms"][0]["base"].update(sha256="not-a-digest"),
])
def test_malformed_contracts(payload, mutation):
    mutation(payload)
    with pytest.raises(ValueError):
        ExperimentSpec(payload)


def test_duplicate_json_keys_fail_closed(payload):
    text = json.dumps(payload)
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        ExperimentSpec.from_json('{"experiment_id":"hidden",' + text[1:])
