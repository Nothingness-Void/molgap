"""Synthetic K1 construction contracts; no datasets or training loops."""
import copy
import json
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from molgap import k1_pair_token, qm9_local_hierarchy, qm9_neural_atom
from molgap.experiment_spec import ExperimentSpec, FAMILIES, SCHEMA_VERSION, TERMINAL_PROTOCOL
from molgap.k1_adapter import build_k1_model, k1_metadata
from molgap.pcqm_gap_architecture import OGBEdgeStateStructuralGPSWrapper

VARIANTS = ("reference", "k1_pair_value")

def _reference(name):
    return {"name": name, "version": "1", "sha256": "a" * 64}


def _addon(name):
    return {"name": name, "version": "1", "config": {}, "source_sha256": "b" * 64}


def _arm(variant="reference", family="neural_atom_k1"):
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
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    return {
        "x": torch.zeros((5, 9), dtype=torch.long),
        "edge_index": edge_index,
        "edge_attr": torch.zeros((6, 3), dtype=torch.long),
        "batch": torch.tensor([0, 0, 0, 1, 1]),
        "random_walk_pe": torch.zeros((5, 16)),
    }


def _same_state(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key


def test_k1_is_edge_state_architecture_variant(spec):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        edge_state = qm9_local_hierarchy.make_encoder()
        torch.manual_seed(42)
        k1 = qm9_neural_atom.make_encoder("neural_atom_k1")

    assert type(edge_state) is OGBEdgeStateStructuralGPSWrapper
    assert isinstance(k1, OGBEdgeStateStructuralGPSWrapper)
    assert spec.family_contract("reference").architecture_base == "ogb_edge_state_structural_gps9"
    assert k1_metadata(spec, "reference").architecture_base_factory == (
        "molgap.qm9_local_hierarchy.make_encoder"
    )
    for name in ("node_emb", "edge_emb", "rwse_encoder", "edge_updates", "head"):
        _same_state(getattr(k1, name).state_dict(), getattr(edge_state, name).state_dict())
    assert len(k1.local_blocks) == len(edge_state.convs)
    assert len(k1.neural_atom_mixers) > 0


def test_baseline_matches_old_factory(spec, synthetic_inputs):
    before = spec.to_json()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        expected = qm9_neural_atom.make_encoder("neural_atom_k1")
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        actual = build_k1_model(spec, "reference")
        assert torch.equal(torch.get_rng_state(), expected_rng)
    _same_state(actual.state_dict(), expected.state_dict())
    assert sum(p.numel() for p in actual.parameters()) == sum(
        p.numel() for p in expected.parameters()
    )
    actual.eval()
    expected.eval()
    with torch.no_grad():
        prediction = actual(**synthetic_inputs)
        assert prediction.shape == (2, 1)
        assert torch.isfinite(prediction).all()
        torch.testing.assert_close(prediction, expected(**synthetic_inputs), rtol=0, atol=0)
    assert spec.to_json() == before


def test_candidate_real_module_identity_and_mechanism(spec, synthetic_inputs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        baseline = qm9_neural_atom.make_encoder("neural_atom_k1")
        torch.manual_seed(42)
        model = build_k1_model(spec, "k1_pair_value")
        actual_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        direct = k1_pair_token.make_encoder(k1_pair_token.VALUE_DECOUPLED_MODE)
        assert torch.equal(torch.get_rng_state(), actual_rng)
    _same_state(model.state_dict(), direct.state_dict())
    _same_state(model.base.state_dict(), baseline.state_dict())
    assert sum(p.numel() for p in model.parameters()) > sum(
        p.numel() for p in baseline.parameters()
    )
    token = model.relation_token
    assert token.mode == k1_pair_token.VALUE_DECOUPLED_MODE
    assert torch.equal(token.value_projection.weight, torch.eye(32))
    assert torch.count_nonzero(token.return_projection.weight) == 0
    hidden = torch.linspace(-1, 1, 5 * 192).reshape(5, 192)
    update, diagnostics = token.compute_update(hidden, synthetic_inputs["batch"])
    assert torch.count_nonzero(update) == 0
    assignment, mask = diagnostics["assignment"], diagnostics["pair_valid"]
    assert torch.equal(mask.sum((1, 2)), torch.tensor([9, 4]))
    assert torch.count_nonzero(assignment.masked_select(~mask)) == 0
    torch.testing.assert_close(assignment.sum((1, 2)), torch.ones(2), rtol=0, atol=1e-6)

    # Compare to the coupled pair-token function with a nonzero return to avoid
    # a vacuous comparison hidden by zero initialization of the return matrix.
    coupled = copy.deepcopy(token)
    coupled.value_projection = torch.nn.Identity()
    with torch.no_grad():
        token.return_projection.weight.fill_(0.01)
        coupled.return_projection.weight.copy_(token.return_projection.weight)
        actual_update, actual_diag = token.compute_update(hidden, synthetic_inputs["batch"])
        coupled_update, coupled_diag = coupled.compute_update(hidden, synthetic_inputs["batch"])
        torch.testing.assert_close(actual_update, coupled_update, rtol=0, atol=1e-6)
        torch.testing.assert_close(actual_diag["token"], coupled_diag["token"], rtol=0, atol=1e-6)
        token.return_projection.weight.zero_()
    model.eval()
    baseline.eval()
    operation = Mock(return_value=None)
    handle = token.register_forward_hook(lambda *_: operation())
    # The hook counts actual integration calls without executing a second token.
    try:
        with torch.no_grad():
            prediction = model(**synthetic_inputs)
            assert prediction.shape == (2, 1)
            assert torch.isfinite(prediction).all()
            torch.testing.assert_close(prediction, baseline(**synthetic_inputs), rtol=0, atol=0)
        assert operation.call_count == 1
    finally:
        handle.remove()


@pytest.mark.parametrize("arm_id", VARIANTS)
def test_metadata_and_dispatch(spec, monkeypatch, arm_id):
    before = spec.to_json()
    metadata = k1_metadata(spec, arm_id)
    assert metadata.family == FAMILIES[("neural_atom_k1", "1")]
    assert metadata.spec_identity == spec.identity
    assert metadata.arm_id == arm_id
    assert metadata.scientific_role == ("reference" if arm_id == "reference" else "candidate")
    assert metadata.checkpoint_resume_owner == "molgap.pcqm_k1_full_runner"
    assert metadata.checkpoint_resume_implemented is False
    assert "does not prove runtime evidence or replay eligibility" in metadata.runtime_evidence_statement
    module = qm9_neural_atom if arm_id == "reference" else k1_pair_token
    expected_mode = "neural_atom_k1" if arm_id == "reference" else k1_pair_token.VALUE_DECOUPLED_MODE
    assert metadata.mode == expected_mode
    assert metadata.addon == (None if arm_id == "reference" else ("k1_pair_value", "1"))
    assert metadata.source_module == module.__name__
    assert metadata.factory == module.__name__ + ".make_encoder"
    factory = Mock(return_value=object())
    monkeypatch.setattr(module, "make_encoder", factory)
    assert build_k1_model(spec, arm_id) is factory.return_value
    factory.assert_called_once_with(expected_mode)
    changed = spec.to_dict()
    changed["arms"][VARIANTS.index(arm_id)]["base"]["sha256"] = "d" * 64
    other = k1_metadata(ExperimentSpec(changed), arm_id)
    assert other.spec_identity != metadata.spec_identity
    assert other.arm_identity != metadata.arm_identity
    assert spec.to_json() == before


@pytest.fixture
def forbidden_factory(monkeypatch):
    factory = Mock(side_effect=AssertionError("Invalid input reached construction"))
    monkeypatch.setattr(qm9_neural_atom, "make_encoder", factory)
    monkeypatch.setattr(k1_pair_token, "make_encoder", factory)
    yield factory
    factory.assert_not_called()


@pytest.mark.parametrize("entrypoint", [k1_metadata, build_k1_model])
@pytest.mark.parametrize("kind", ["dict", "callable", "module", "subclass"])
def test_provider_rejected(payload, entrypoint, kind, forbidden_factory):
    provider = Mock()
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


@pytest.mark.parametrize("entrypoint", [k1_metadata, build_k1_model])
def test_unknown_arm_family_and_nonstring(spec, payload, entrypoint, forbidden_factory):
    with pytest.raises(ValueError, match="Unknown arm"):
        entrypoint(spec, "missing")
    with pytest.raises(TypeError, match="arm_id must be a string"):
        entrypoint(spec, spec.to_dict()["arms"][0])
    payload["arms"] = [_arm(family="gptrans_t")]
    with pytest.raises(ValueError, match="requires neural_atom_k1"):
        entrypoint(ExperimentSpec(payload), "reference")


@pytest.mark.parametrize("entrypoint", [k1_metadata, build_k1_model])
@pytest.mark.parametrize("case", [
    "unknown", "version", "duplicate", "gptrans", "mixed", "config",
    "family", "family_version", "module", "snapshot", "duplicate_json",
])
def test_forged_snapshot_rejected(payload, entrypoint, case, forbidden_factory):
    arm = payload["arms"][1]
    if case == "unknown":
        arm["addons"][0]["name"] = "unknown"
    elif case == "version":
        arm["addons"][0]["version"] = "99"
    elif case == "duplicate":
        arm["addons"] *= 2
    elif case == "gptrans":
        arm["addons"] = [_addon("pair_prenorm")]
    elif case == "mixed":
        arm["addons"].append(_addon("centered_logits"))
    elif case == "config":
        arm["addons"][0]["config"] = {"factory": "author.Model"}
    elif case == "family":
        arm["family"]["name"] = "unknown"
    elif case == "family_version":
        arm["family"]["version"] = "99"
    elif case == "module":
        arm["family"]["module"] = "author.Module"
    elif case == "snapshot":
        arm["arm_snapshot"] = copy.deepcopy(arm)
    text = json.dumps(payload)
    if case == "duplicate_json":
        text = '{"experiment_id":"hidden",' + text[1:]
    with pytest.raises(ValueError):
        ExperimentSpec.from_json(text)
    forged = object.__new__(ExperimentSpec)
    object.__setattr__(forged, "_canonical_json", text)
    with pytest.raises(ValueError):
        entrypoint(forged, "k1_pair_value")


@pytest.mark.parametrize("entrypoint", [k1_metadata, build_k1_model])
@pytest.mark.parametrize("field", ["factory", "module", "provider", "arm_snapshot"])
def test_kwargs_rejected(spec, entrypoint, field, forbidden_factory):
    supplied = Mock()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        entrypoint(spec, "reference", **{field: supplied})
    supplied.assert_not_called()


@pytest.mark.parametrize("entrypoint", [k1_metadata, build_k1_model])
def test_instance_method_injection_rejected(spec, entrypoint, forbidden_factory):
    provider = Mock()
    object.__setattr__(spec, "to_json", provider)
    with pytest.raises(ValueError, match="snapshot"):
        entrypoint(spec, "reference")
    provider.assert_not_called()


@pytest.mark.parametrize("arm_id", VARIANTS)
def test_state_path_rejected(spec, tmp_path, arm_id, forbidden_factory):
    with pytest.raises(ValueError, match="does not implement initial-state loading"):
        build_k1_model(spec, arm_id, initial_state_path=tmp_path / "unwritten.pt")


@pytest.mark.parametrize("value", [Mock(), ModuleType("author_module")])
def test_objects_in_spec_rejected(payload, value):
    payload["arms"][1]["addons"][0]["config"] = {"provider": value}
    with pytest.raises(ValueError):
        ExperimentSpec(payload)



