"""Synthetic EdgeState family contracts; no training or protected data."""
import copy
import json

import pytest
import torch

from molgap import edge_state_model_only_v1, qm9_local_hierarchy, qm9_neural_atom
from molgap.edge_state_adapter import build_edge_state_model, edge_state_metadata
from molgap.experiment_runner import run_experiment
from molgap.experiment_spec import ExperimentSpec, FAMILIES, SCHEMA_VERSION, TERMINAL_PROTOCOL


def _ref(name):
    return {"name": name, "version": "1", "sha256": "a" * 64}


def _arm(arm_id, addon=None):
    contract = FAMILIES[("edge_state_gps", "1")]
    return {
        "arm_id": arm_id,
        "scientific_role": "reference" if addon is None else "candidate",
        "family": {"name": "edge_state_gps", "version": "1"},
        "base": _ref("synthetic-base"),
        "initialization": {"kind": "random", "seed": 42, "state_sha256": "b" * 64},
        "data": {
            "dataset": _ref("pcqm4mv2"), "split": _ref("synthetic-split"),
            "roles": [
                {"role": role, "membership_sha256": "c" * 64,
                 "row_order_sha256": "d" * 64, "usage_sha256": "e" * 64}
                for role in contract.roles
            ],
            "feature_schema": contract.feature_schema, "feature_sha256": "f" * 64,
            "target": "pcqm4mv2-gap-eV-direct",
        },
        "training": {
            "recipe": _ref(contract.recipe), "overrides": {},
            "objective": _ref("normalized-gap-l1"),
            "sampler": _ref(contract.sampler), "transform": _ref(contract.transform),
        },
        "addons": [] if addon is None else [addon],
        "addon_semantics": "baseline" if addon is None else "ordered",
    }


def _addon(name, config=None):
    return {"name": name, "version": "1", "config": config or {},
            "source_sha256": "b" * 64}


@pytest.fixture
def payload():
    return {
        "schema_version": SCHEMA_VERSION, "experiment_id": "synthetic-edge-state",
        "logical_run_id": "model-only", "arms": [
            _arm("edge-state"),
            _arm("depth-6", _addon("edge_state_depth", {"num_layers": 6})),
            _arm("k1", _addon("neural_atom_k1")),
        ],
        "platform": {"name": "local", "accelerator": "cpu", "device_count": 3,
                     "cpu_cores": 1, "memory_gib": 4, "atomic_checkpoints": True,
                     "retrievable_chunks": True},
        "prospective": {"trajectory_id": "synthetic-edge-state",
                        "hypothesis": "Construction identity only",
                        "cheapest_falsifier": "Local synthetic test",
                        "stop_rule": "No training", "budget_sha256": "a" * 64},
        "evidence": {"policy": _ref("molgap-v5"), "required_artifacts": [
            "v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact",
        ]},
        "terminal_protocol": TERMINAL_PROTOCOL,
    }


def _same_state(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def test_family_and_variants_dispatch_to_existing_factories(payload):
    spec = ExperimentSpec(payload)
    assert ExperimentSpec.from_json(spec.to_json()).identity == spec.identity
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        baseline = build_edge_state_model(spec, "edge-state")
        torch.manual_seed(42)
        direct_baseline = qm9_local_hierarchy.make_encoder()
        torch.manual_seed(42)
        k1 = build_edge_state_model(spec, "k1")
        torch.manual_seed(42)
        direct_k1 = qm9_neural_atom.make_encoder("neural_atom_k1")
    _same_state(baseline.state_dict(), direct_baseline.state_dict())
    _same_state(k1.state_dict(), direct_k1.state_dict())
    assert edge_state_metadata(spec, "edge-state").num_layers == 9
    assert edge_state_metadata(spec, "depth-6").num_layers == 6
    assert edge_state_metadata(spec, "k1").addon == ("neural_atom_k1", "1")


@pytest.mark.parametrize("depth", [1, 6, 12, 16])
def test_depth_only_model_constructs_and_forwards(depth):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = edge_state_model_only_v1.make_encoder(depth)
    assert len(model.convs) == depth
    assert len(model.edge_updates) == depth
    model.eval()
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    with torch.no_grad():
        prediction = model(
            torch.zeros((3, 9), dtype=torch.long), edge_index,
            torch.zeros((4, 3), dtype=torch.long), torch.zeros(3, dtype=torch.long),
            torch.zeros((3, 16)),
        )
    assert prediction.shape == (1, 1)
    assert bool(torch.isfinite(prediction).all())


@pytest.mark.parametrize("value", [0, 9, 17, -1, 6.0, "6", True, None])
def test_depth_addon_rejects_invalid_or_duplicate_baseline(payload, value):
    payload["arms"] = [_arm("candidate", _addon("edge_state_depth", {"num_layers": value}))]
    with pytest.raises(ValueError, match="num_layers"):
        ExperimentSpec(payload)


@pytest.mark.parametrize("addons", [
    [_addon("edge_state_depth", {"num_layers": 6}), _addon("neural_atom_k1")],
    [_addon("neural_atom_k1"), _addon("neural_atom_k1")],
    [_addon("k1_pair_value")],
])
def test_incompatible_variants_rejected(payload, addons):
    payload["arms"] = [_arm("candidate", addons[0])]
    payload["arms"][0]["addons"] = addons
    with pytest.raises(ValueError, match="Incompatible"):
        ExperimentSpec(payload)


def test_depth_is_bound_to_spec_identity(payload):
    spec = ExperimentSpec(payload)
    changed = copy.deepcopy(payload)
    changed["arms"][1]["addons"][0]["config"]["num_layers"] = 12
    other = ExperimentSpec(changed)
    assert other.identity != spec.identity
    assert edge_state_metadata(other, "depth-6").arm_identity != (
        edge_state_metadata(spec, "depth-6").arm_identity
    )


def test_model_only_boundary_and_forged_snapshot(payload, tmp_path):
    spec = ExperimentSpec(payload)
    with pytest.raises(ValueError, match="does not implement initial-state loading"):
        build_edge_state_model(spec, "edge-state", initial_state_path=tmp_path / "state.pt")
    with pytest.raises(TypeError, match="Expected an ExperimentSpec"):
        edge_state_metadata(payload, "edge-state")
    forged = object.__new__(ExperimentSpec)
    altered = spec.to_dict()
    altered["arms"][1]["addons"][0]["config"]["num_layers"] = 0
    object.__setattr__(forged, "_canonical_json", json.dumps(altered))
    with pytest.raises(ValueError, match="num_layers"):
        build_edge_state_model(forged, "depth-6")


def test_runner_probe_uses_edge_state_adapter(payload, tmp_path):
    payload["arms"] = [payload["arms"][1]]
    payload["platform"]["device_count"] = 1
    summary = run_experiment(ExperimentSpec(payload), tmp_path / "probe", ["cpu"])
    assert summary["status"] == "SUCCEEDED"
    result = json.loads((tmp_path / "probe" / "arms" / "depth-6" / "arm_result.json").read_text())
    assert result["metadata"]["num_layers"] == 6
    assert result["parameter_count"] is None


def test_runner_constructs_depth_variant(payload, tmp_path):
    payload["arms"] = [payload["arms"][1]]
    payload["platform"]["device_count"] = 1
    summary = run_experiment(
        ExperimentSpec(payload), tmp_path / "construct", ["cpu"],
        worker="construct", timeout_seconds=120,
    )
    assert summary["status"] == "SUCCEEDED"
    result = json.loads(
        (tmp_path / "construct" / "arms" / "depth-6" / "arm_result.json").read_text()
    )
    assert result["parameter_count"] > 0
    assert result["metadata"]["num_layers"] == 6
