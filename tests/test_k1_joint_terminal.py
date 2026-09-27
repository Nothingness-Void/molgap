"""Saved-artifact terminal translation; never train or instantiate a model."""
import ast
import importlib.util
import inspect
import json
from pathlib import Path
import shutil

import pytest

from molgap.k1_joint_terminal import RECIPES, split_stage_cost


def stage_fixture():
    execution = {"complete": True, "total_allocated_device_seconds": 240,
        "total_job_wall_seconds": 120,
        "workers": [{"mode": mode, "complete": True,
            "allocated_device_seconds": seconds, "worker_wall_seconds": seconds}
            for mode, seconds in zip(RECIPES, (100, 110))]}
    costs = {mode: {"complete": True, "device_count": 1,
        "wall_seconds": seconds, "audit_wall_seconds": audit}
        for mode, seconds, audit in zip(RECIPES, (99, 109), (10, 15))}
    return execution, costs


def test_parallel_audit_cost_excluded_once_from_each_training_arm():
    execution, costs = stage_fixture()
    result = split_stage_cost(execution, costs)
    assert result["audit_device_seconds"] == 25
    assert result["training"][RECIPES[0]]["device_seconds"] == 120
    assert result["training"][RECIPES[1]]["device_seconds"] == 95
    assert result["training"][RECIPES[0]]["allocated_overhead_seconds"] == 30
    assert result["total_allocated_device_seconds"] == 240
    assert "audit_wall_seconds" not in result
    assert execution["total_job_wall_seconds"] == 120


@pytest.mark.parametrize("field,value", [
    ("audit_wall_seconds", 0), ("audit_wall_seconds", -1),
    ("audit_wall_seconds", float("nan")), ("audit_wall_seconds", 100),
    ("wall_seconds", 101), ("complete", False), ("device_count", 2),
])
def test_invalid_stage_measurement_rejected(field, value):
    execution, costs = stage_fixture()
    costs[RECIPES[0]][field] = value
    with pytest.raises(ValueError, match="native stage"):
        split_stage_cost(execution, costs)


def test_legacy_terminal_defaults_unchanged_and_joint_uses_typed_objective():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("joint_terminal_test_adapter",
        root / "experiments/pcqm_k1_functional_group_token_100k/prepare_rml_terminal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    defaults = inspect.signature(module.main).parameters
    assert defaults["experiment_purpose"].default == "architecture_comparison"
    assert defaults["declared_intervention_fields"].default == ("architecture_config_identity",)
    assert defaults["observed_comparison_identity"].default is None
    source = (root / "src/molgap/k1_joint_terminal.py").read_text()
    tree = ast.parse(source)
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))}
    assert not calls.intersection({"_model", "_infer", "make_encoder", "backward", "step", "kernels_push"})
    assert 'experiment_purpose="training_objective_comparison"' in source
    assert 'declared_intervention_fields=("loss_identity",)' in source


def test_unfinished_parent_cost_rejected():
    execution, costs = stage_fixture()
    execution["complete"] = False
    with pytest.raises(ValueError, match="notebook cost"):
        split_stage_cost(execution, costs)


def test_no_train_stage_cost_override_does_not_invent_parallel_wall(tmp_path):
    from molgap.k1_relation_audit_records import prepare_no_train_terminal
    root = Path(__file__).resolve().parents[1]
    prefix = "experiments/pcqm_k1_relation_resolution_100k/audit"
    terminal = json.loads((root / prefix / "results/terminal.json").read_text())
    for ref in terminal["artifact_hashes"]:
        target = tmp_path / ref
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / ref, target)
    evidence = terminal["evidence"]
    measurements = {key: {"status": "measurement_missing", "value": None}
        for key in ("wall_hours", "cpu_hours", "queue_hours")}
    measurements["device_hours"] = {"status": "measured", "value": 25 / 3600}
    prepare_no_train_terminal(prefix=prefix,
        frozen=json.loads((tmp_path / prefix / "rml_plan/trajectory.json").read_text()),
        run_id=terminal["run_id"], evidence_id=evidence["evidence_id"],
        outcome=evidence["outcome"], scope=evidence["scope"],
        finalized_at=terminal["finalized_at"], acceptance_name="acceptance_summary",
        artifact_refs=[a["locator"] for a in evidence["artifacts"]],
        authority=evidence["authority"]["pointers"], repo_root=tmp_path,
        cost_measurement=measurements, attempt_id="v3", cost_semantics="observed audit device intervals only")
    costs = json.loads((tmp_path / prefix / "results/cost_records.json").read_text())
    assert costs["costs"][0]["measurement"] == measurements
    assert costs["costs"][0]["attempt_id"] == "v3"
    assert costs["semantics"] == "observed audit device intervals only"
