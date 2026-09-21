"""Synthetic adapter contracts; no datasets, checkpoint artifacts, or training jobs."""
import json
from dataclasses import FrozenInstanceError
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from molgap import gptrans_variants, pcqm_gptrans_v4
from molgap.experiment_spec import (
    ExperimentSpec, FAMILIES, SCHEMA_VERSION, TERMINAL_PROTOCOL,
)
from molgap.gptrans import GraphPropagationAttention
from molgap.gptrans_adapter import build_gptrans_model, gptrans_metadata
from molgap.gptrans_variants import RelationFlowAttention


VARIANTS = ("reference", "pair_prenorm", "centered_logits")


def _reference(name):
    return {"name": name, "version": "1", "sha256": "a" * 64}


def _addon(name):
    return {"name": name, "version": "1", "config": {}, "source_sha256": "b" * 64}


def _arm(variant="reference", family="gptrans_t"):
    contract = FAMILIES[(family, "1")]
    return {
        "arm_id": variant, "scientific_role": "reference" if variant == "reference" else "candidate",
        "family": {"name": family, "version": "1"},
        "base": _reference("synthetic-base"),
        "initialization": {"kind": "random", "seed": 42, "state_sha256": "c" * 64},
        "data": {
            "dataset": _reference("pcqm4mv2"), "split": _reference("synthetic-split"),
            "roles": [
                {"role": role, "membership_sha256": "d" * 64,
                 "row_order_sha256": "e" * 64, "usage_sha256": "f" * 64}
                for role in contract.roles
            ],
            "feature_schema": contract.feature_schema, "feature_sha256": "a" * 64,
            "target": "pcqm4mv2-gap-eV-direct",
        },
        "training": {
            "recipe": _reference(contract.recipe), "overrides": {},
            "objective": _reference("normalized-gap-l1"),
            "sampler": _reference(contract.sampler), "transform": _reference(contract.transform),
        },
        "addons": [] if variant == "reference" else [_addon(variant)],
        "addon_semantics": "baseline" if variant == "reference" else "ordered",
    }


@pytest.fixture
def payload():
    # Digests are synthetic declarations, never assertions about real artifacts.
    return {
        "schema_version": SCHEMA_VERSION, "experiment_id": "synthetic-adapter",
        "logical_run_id": "synthetic-run", "arms": [_arm(v) for v in VARIANTS],
        "platform": {
            "name": "local", "accelerator": "cpu", "device_count": 1,
            "cpu_cores": 1, "memory_gib": 4, "atomic_checkpoints": True,
            "retrievable_chunks": True,
        },
        "prospective": {
            "trajectory_id": "synthetic-adapter", "hypothesis": "Adapter preserves dispatch",
            "cheapest_falsifier": "Synthetic unit test", "stop_rule": "No training",
            "budget_sha256": "a" * 64,
        },
        "evidence": {
            "policy": _reference("molgap-v5"),
            "required_artifacts": [
                "v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact",
            ],
        },
        "terminal_protocol": TERMINAL_PROTOCOL,
    }


@pytest.fixture
def spec(payload):
    return ExperimentSpec(payload)


@pytest.fixture
def synthetic_inputs():
    # Unequal graph sizes exercise attention padding as well as connected edges.
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long)
    return {
        "x": torch.zeros((5, 9), dtype=torch.long),
        "edge_index": edge_index,
        "edge_attr": torch.zeros((edge_index.shape[1], 3), dtype=torch.long),
        "batch": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    }


def test_reference_matches_frozen_factory(spec, synthetic_inputs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        expected = pcqm_gptrans_v4._make_model(variant="reference")
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        actual = build_gptrans_model(spec, "reference")
        assert torch.equal(torch.get_rng_state(), expected_rng)

    assert actual is not expected
    assert sum(p.numel() for p in actual.parameters()) == (
        sum(p.numel() for p in expected.parameters())
    ) == pcqm_gptrans_v4.EXPECTED_PARAMETERS
    actual_state, expected_state = actual.state_dict(), expected.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name, tensor in actual_state.items():
        other = expected_state[name]
        assert tensor.shape == other.shape, name
        assert tensor.dtype == other.dtype, name
        assert torch.equal(tensor, other), name
    assert len(actual.blocks) == 12
    assert all(type(block.attention) is GraphPropagationAttention for block in actual.blocks)
    actual.eval()
    expected.eval()
    with torch.no_grad():
        prediction = actual(**synthetic_inputs)
        reference_prediction = expected(**synthetic_inputs)
    assert prediction.shape == (2, 1)
    assert torch.isfinite(prediction).all()
    torch.testing.assert_close(prediction, reference_prediction, rtol=0, atol=0)


@pytest.mark.parametrize("variant,helper", [
    ("pair_prenorm", "normalize_pair"), ("centered_logits", "center_valid_logits"),
])
def test_real_addon_executes_in_every_block_and_receives_gradients(
    spec, synthetic_inputs, monkeypatch, variant, helper,
):
    operation = Mock(wraps=getattr(gptrans_variants, helper))
    monkeypatch.setattr(gptrans_variants, helper, operation)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = build_gptrans_model(spec, variant)
    assert len(model.blocks) == 12
    for block in model.blocks:
        assert type(block.attention) is RelationFlowAttention
        assert block.attention.variant == variant

    # Eval disables stochastic depth/dropout; autograd remains active for every block.
    model.eval()
    prediction = model(**synthetic_inputs)
    assert prediction.shape == (2, 1)
    assert torch.isfinite(prediction).all()
    assert operation.call_count == len(model.blocks)
    prediction.sum().backward()
    for index, block in enumerate(model.blocks):
        for name, parameter in block.attention.named_parameters():
            assert parameter.grad is not None, (index, name)
            assert torch.isfinite(parameter.grad).all(), (index, name)
        assert torch.count_nonzero(block.attention.qkv.weight.grad) > 0, index
        assert torch.count_nonzero(block.attention.pair_to_node.weight.grad) > 0, index
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("variant", VARIANTS)
def test_metadata_declares_identity_factory_and_unimplemented_resume(spec, variant):
    metadata = gptrans_metadata(spec, variant)
    assert metadata.family == FAMILIES[("gptrans_t", "1")]
    assert metadata.family.recipe == "pcqm_gptrans_v4"
    assert metadata.factory == "molgap.pcqm_gptrans_v4._make_model"
    assert metadata.variant == variant
    assert metadata.arm_id == variant
    assert metadata.spec_identity == spec.identity
    assert metadata.checkpoint_resume_owner == "existing trainer"
    assert metadata.checkpoint_resume_implemented is False
    assert set(metadata.required_checkpoint_identity_fields) == {
        "spec_identity", "arm_id", "family.name", "family.version", "variant",
    }
    changed = spec.to_dict()
    changed["logical_run_id"] = "another-synthetic-run"
    assert gptrans_metadata(ExperimentSpec(changed), variant).spec_identity != metadata.spec_identity
    with pytest.raises(FrozenInstanceError):
        metadata.variant = "reference"


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("use_path", [False, True])
def test_public_initial_state_path_delegates_to_existing_factory(
    spec, tmp_path, monkeypatch, variant, use_path,
):
    path = tmp_path / "unwritten-initial-state.pt" if use_path else None
    sentinel = object()
    factory = Mock(return_value=sentinel)
    monkeypatch.setattr(pcqm_gptrans_v4, "_make_model", factory)
    assert build_gptrans_model(spec, variant, initial_state_path=path) is sentinel
    factory.assert_called_once_with(initial_state_path=path, variant=variant)
    if path is not None:
        assert not path.exists()


def test_initial_state_validation_error_is_not_swallowed(spec, tmp_path, monkeypatch):
    error = ValueError("frozen initial state rejected")
    factory = Mock(side_effect=error)
    monkeypatch.setattr(pcqm_gptrans_v4, "_make_model", factory)
    path = tmp_path / "unwritten.pt"
    with pytest.raises(ValueError, match="frozen initial state rejected") as caught:
        build_gptrans_model(spec, "pair_prenorm", initial_state_path=path)
    assert caught.value is error
    factory.assert_called_once_with(initial_state_path=path, variant="pair_prenorm")


@pytest.fixture
def forbidden_factory(monkeypatch):
    factory = Mock(side_effect=AssertionError("Invalid declarations reached model construction"))
    monkeypatch.setattr(pcqm_gptrans_v4, "_make_model", factory)
    yield factory
    factory.assert_not_called()


@pytest.mark.parametrize("entrypoint", [gptrans_metadata, build_gptrans_model])
def test_unknown_arm_rejected(spec, forbidden_factory, entrypoint):
    with pytest.raises(ValueError, match="Unknown arm"):
        entrypoint(spec, "missing")


@pytest.mark.parametrize("entrypoint", [gptrans_metadata, build_gptrans_model])
def test_registered_non_gptrans_family_rejected(payload, forbidden_factory, entrypoint):
    payload["arms"] = [_arm(family="neural_atom_k1")]
    with pytest.raises(ValueError, match="requires gptrans_t family/version 1"):
        entrypoint(ExperimentSpec(payload), "reference")


@pytest.mark.parametrize("entrypoint", [gptrans_metadata, build_gptrans_model])
@pytest.mark.parametrize("case,match", [
    ("family", "Unknown family/version"), ("version", "Unknown family/version"),
    ("addon", "Unknown addon/version"), ("addon_version", "Unknown addon/version"),
    ("multiple", "Incompatible addon combination/family"),
    ("duplicate", "Incompatible addon combination/family"),
    ("module", "expected exactly fields"),
])
def test_forged_snapshot_is_revalidated(payload, forbidden_factory, entrypoint, case, match):
    arm = payload["arms"][0]
    if case in ("family", "version"):
        arm["family"]["name" if case == "family" else "version"] = "unsupported"
    elif case == "module":
        arm["family"]["module"] = "author.supplied.Model"
    else:
        arm.update(addons=[_addon("pair_prenorm")], addon_semantics="ordered")
        if case == "addon":
            arm["addons"][0]["name"] = "memory_value"
        elif case == "addon_version":
            arm["addons"][0]["version"] = "99"
        else:
            arm["addons"].append(_addon("centered_logits" if case == "multiple" else "pair_prenorm"))
    with pytest.raises(ValueError, match=match):
        ExperimentSpec(payload)
    # Exercise the adapter boundary even if an author bypasses __init__.
    forged = object.__new__(ExperimentSpec)
    object.__setattr__(forged, "_canonical_json", json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        entrypoint(forged, "reference")


@pytest.mark.parametrize("entrypoint", [gptrans_metadata, build_gptrans_model])
@pytest.mark.parametrize("kind", ["dict", "callable", "module", "subclass"])
def test_custom_provider_bypasses_rejected(payload, forbidden_factory, entrypoint, kind):
    provider = Mock(side_effect=AssertionError("Author provider executed"))
    if kind == "dict":
        supplied = payload
    elif kind == "callable":
        supplied = provider
    elif kind == "module":
        supplied = ModuleType("author_module")
        supplied.to_json = provider
    else:
        class AuthorSpec(ExperimentSpec):
            def to_json(self):
                return provider()
        supplied = AuthorSpec(payload)
    with pytest.raises(TypeError, match="Expected an ExperimentSpec"):
        entrypoint(supplied, "reference")
    provider.assert_not_called()


@pytest.mark.parametrize("entrypoint", [gptrans_metadata, build_gptrans_model])
def test_non_string_arm_rejected(spec, forbidden_factory, entrypoint):
    supplied = Mock(side_effect=AssertionError("Author arm executed"))
    with pytest.raises(TypeError, match="arm_id must be a string"):
        entrypoint(spec, supplied)
    supplied.assert_not_called()


@pytest.mark.parametrize("field", ["factory", "module", "model", "apply_variant"])
def test_author_dispatch_kwargs_rejected(spec, forbidden_factory, field):
    supplied = Mock(side_effect=AssertionError("Author dispatch executed"))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_gptrans_model(spec, "reference", **{field: supplied})
    supplied.assert_not_called()


@pytest.mark.parametrize("kind", ["callable", "module"])
def test_author_objects_in_declarations_rejected(payload, forbidden_factory, kind):
    supplied = Mock() if kind == "callable" else ModuleType("author_module")
    payload["arms"][0]["family"]["name"] = supplied
    with pytest.raises(ValueError, match="family.name: expected nonempty trimmed string"):
        ExperimentSpec(payload)
    if kind == "callable":
        supplied.assert_not_called()
