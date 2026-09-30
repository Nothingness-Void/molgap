"""Synthetic guards only: no factories, checkpoint weights or GPU execution."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from molgap.constants import REPO_ROOT
from molgap.research_memory.schemas import validate_cost_event, validate_trajectory
from molgap.research_memory.policy import validate_policy


@pytest.fixture
def addon():
    path = REPO_ROOT / "experiments/pcqm_scale_fit_retained/run_diagnostic.py"
    spec = importlib.util.spec_from_file_location("scale_fit_test_addon", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_schemas(addon):
    validate_trajectory(addon.read(addon.DIRECTORY / "trajectory_template.json"))
    validate_cost_event(addon.read(addon.DIRECTORY / "expected_cost.json"))
    addon.validate_inputs(addon.read(addon.DIRECTORY / "inputs.json"))


def test_policy_binds_actual_protocol_without_new_promotion_or_stop_gate(addon):
    policy = addon.read(REPO_ROOT / "research_memory/policies/pcqm-scale-fit-retained-diagnostic.1.json")
    validate_policy(policy)
    assert policy["created_from_source_digest"] == addon.file_digest(addon.DIRECTORY / "protocol.md")
    assert policy["promotion_rule"] is None
    assert policy["early_stop_rule"] is None


@pytest.mark.parametrize("mutation", ["historical_dev", "ema500k", "tf32", "train", "role"])
def test_frozen_scope_rejects_mutation(addon, mutation):
    data = copy.deepcopy(addon.read(addon.DIRECTORY / "inputs.json"))
    if mutation == "historical_dev":
        data["cohorts"]["development"]["start"] = 100000
    elif mutation == "ema500k":
        data["checkpoints"]["reference_500k"]["states"]["ema"] = "model"
    elif mutation == "tf32":
        data["execution"]["tf32_enabled"] = True
    elif mutation == "train":
        data["authorization"]["optimizer_updates_allowed"] = True
    else:
        data["authorization"]["official_roles_allowed"] = True
    with pytest.raises(ValueError):
        addon.validate_inputs(data)


def test_hash_change_fails_before_read(addon, tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"frozen")
    pin = addon.file_digest(path)
    addon.require_hash(path, pin)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Hash mismatch"):
        addon.require_hash(path, pin)


def test_review_is_explicit_and_hash_bound(addon):
    review = {"approved": True, "authority": "parent", "approved_at": "2026-10-01",
              "plan_sha256": "a", "trajectory_sha256": "b",
              "prospective_freeze_scope_approved": True}
    addon.check_review(review, "a", "b")
    for changed in ({}, dict(review, plan_sha256="changed"), dict(review, approved=False)):
        with pytest.raises(ValueError):
            addon.check_review(changed, "a", "b")


@pytest.mark.parametrize("field", ["approved", "authority", "approved_at", "plan_sha256",
                                   "trajectory_sha256", "prospective_freeze_scope_approved"])
def test_review_requires_every_frozen_authority_binding(addon, field):
    review = dict(approved=True, authority="synthetic authority", approved_at="2026-10-01",
                  plan_sha256="a" * 64, trajectory_sha256="b" * 64,
                  prospective_freeze_scope_approved=True)
    addon.check_review(review, "a" * 64, "b" * 64)
    review.pop(field)
    with pytest.raises(ValueError):
        addon.check_review(review, "a" * 64, "b" * 64)


@pytest.mark.parametrize("mutation", ["order", "count", "nan"])
def test_prediction_alignment_guard(addon, mutation):
    record = {"source_idx": torch.arange(500000, 505000),
              "target_eV": torch.zeros(5000), "prediction_eV": torch.zeros(5000)}
    addon.validate_prediction(record, "development")
    if mutation == "order":
        record["source_idx"] = record["source_idx"].flip(0)
    elif mutation == "count":
        record["prediction_eV"] = torch.zeros(4999)
    else:
        record["target_eV"][0] = float("nan")
    with pytest.raises(ValueError):
        addon.validate_prediction(record, "development")


def test_graph_guard(addon):
    from types import SimpleNamespace
    graph = SimpleNamespace(source_idx=torch.tensor([0]), x=torch.zeros(2, 9),
        edge_attr=torch.zeros(1, 3), edge_index=torch.tensor([[0], [1]]), y=torch.tensor([1.]))
    addon.validate_graph(graph, 0)
    graph.source_idx = torch.tensor([100000])
    with pytest.raises(ValueError):
        addon.validate_graph(graph, 0)


def test_analysis_does_not_promote(addon):
    records = {}
    for arm, semantics in addon.STATES.items():
        for state in semantics:
            for role, (start, stop) in addon.COHORTS.items():
                records[f"{arm}.{state}.{role}"] = {"source_idx": torch.arange(start, stop),
                    "target_eV": torch.ones(5000), "prediction_eV": torch.ones(5000)}
    result = addon.paired_analysis(records)
    assert len(result) == 6
    assert all(v["paired_gain_eV"] == 0 for v in result.values())
    assert all("material_3mev_point_gain" not in v for v in result.values())


def test_no_optimizer_or_platform_call():
    import ast
    path = REPO_ROOT / "experiments/pcqm_scale_fit_retained/run_diagnostic.py"
    tree = ast.parse(path.read_bytes())
    forbidden = {"backward", "step", "optimizer_for", "run_training", "rebuild"}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in forbidden for n in ast.walk(tree))
