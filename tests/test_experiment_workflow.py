"""Composition tests; no real prospective publication, model or platform calls."""
import json
from unittest.mock import Mock

import pytest

from molgap import experiment_workflow as workflow
from molgap.experiment_spec import ExperimentSpec
from test_experiment_spec import payload


@pytest.fixture
def case(payload, tmp_path):
    payload["schema_version"] = "molgap-experiment-spec-v2"
    payload["prospective"] = {"arms": [
        {"arm_id": arm["arm_id"], "trajectory_id": f"TC-{i}",
         "plan_spec_ref": f"experiments/synthetic/plan{i}.json", "plan_spec_sha256": "a" * 64,
         "output": f"experiments/synthetic/rml{i}"} for i, arm in enumerate(payload["arms"])]}
    spec = ExperimentSpec(payload)
    config = {
        "format": workflow.WORKFLOW_FORMAT, "spec_identity": spec.identity,
        "source_paths": ["src/molgap/fixture.py", "recipe.json", "entry.py", "kernel.json"],
        "artifacts": {"initial.pt": {"path": "local/initial.pt", "sha256": "b" * 64}},
        "recipe_files": {"gptrans_t": "recipe.json", "neural_atom_k1": "recipe.json"},
        "initial_states": {"gptrans_t": "initial.pt", "neural_atom_k1": "initial.pt"},
        "required_modules": ["molgap.fixture"], "entry_template": "entry.py",
        "kernel_metadata": "kernel.json", "dataset_metadata": {"id": "synthetic/private"}, "pickle_inputs": [],
    }
    return spec, tmp_path, config, tmp_path / "workflow"


def test_one_entry_plans_before_staging(case, monkeypatch):
    spec, root, config, output = case
    ordered = []
    plan = Mock(side_effect=lambda *_: (ordered.append("plan") or {"status": "PLANNED"}, 0))
    stage = Mock(side_effect=lambda *_, **__: ordered.append("stage") or {"errors": [], "release_status": "VERIFIED"})
    monkeypatch.setattr(workflow, "plan_prospective", plan)
    monkeypatch.setattr(workflow, "stage_release_inputs", stage)
    result, code = workflow.prepare_experiment_release(spec, root, json.dumps(config), output)
    assert code == 0 and ordered == ["plan", "stage"]
    assert result["submitted"] is result["compute_released"] is False
    assert stage.call_args.kwargs["artifacts"]["initial.pt"].path == root / "local/initial.pt"
    assert stage.call_args.kwargs["output"] == output / "release"
    assert json.loads((output / "workflow.json").read_text()) == result
    with pytest.raises(FileExistsError):
        workflow.prepare_experiment_release(spec, root, json.dumps(config), output)
    assert plan.call_count == 1


def test_partial_plan_blocks_packaging_and_retains_receipt(case, monkeypatch):
    spec, root, config, output = case
    retained = {"status": "PARTIAL", "completed_records": ["one-arm"]}
    monkeypatch.setattr(workflow, "plan_prospective", lambda *_: (retained, 1))
    stage = Mock()
    monkeypatch.setattr(workflow, "stage_release_inputs", stage)
    result, code = workflow.prepare_experiment_release(spec, root, json.dumps(config), output)
    assert code == 1 and result["prospective"] == retained
    stage.assert_not_called()
    assert "RECONCILE" in result["status"]


def test_failed_release_does_not_become_authority(case, monkeypatch):
    spec, root, config, output = case
    monkeypatch.setattr(workflow, "plan_prospective", lambda *_: ({"status": "PLANNED"}, 0))
    monkeypatch.setattr(workflow, "stage_release_inputs", lambda *_, **__: {"errors": ["missing dependency"]})
    result, code = workflow.prepare_experiment_release(spec, root, json.dumps(config), output)
    assert code == 1 and result["status"] == "LOCAL_PREPARATION_BLOCKED"
    assert result["compute_released"] is False


def test_exception_keeps_published_plan_for_reconciliation(case, monkeypatch):
    spec, root, config, output = case
    monkeypatch.setattr(workflow, "plan_prospective", lambda *_: ({"status": "PLANNED"}, 0))
    monkeypatch.setattr(workflow, "stage_release_inputs", Mock(side_effect=ValueError("changed source")))
    with pytest.raises(ValueError, match="changed source"):
        workflow.prepare_experiment_release(spec, root, json.dumps(config), output)
    receipt = json.loads((output / "workflow.json").read_text())
    assert receipt["prospective"]["status"] == "PLANNED"
    assert "RECONCILE" in receipt["status"]


@pytest.mark.parametrize("mutation", ["spec", "unknown", "callback", "artifact", "empty", "path", "sha", "arm", "recipe"])
def test_invalid_config_never_plans(case, monkeypatch, mutation):
    spec, root, config, output = case
    if mutation == "spec":
        config["spec_identity"] = "0" * 64
    elif mutation == "unknown":
        config["submit"] = True
    elif mutation == "callback":
        config["trainer"] = "arbitrary.module"
    elif mutation == "artifact":
        config["artifacts"]["initial.pt"]["extra"] = True
    elif mutation == "empty":
        config["required_modules"] = []
    elif mutation == "sha":
        config["artifacts"]["initial.pt"]["sha256"] = "not-a-digest"
    elif mutation == "arm":
        config["initial_states"]["other-arm"] = "initial.pt"
    elif mutation == "recipe":
        config["recipe_files"]["gptrans_t"] = "missing.json"
    else:
        config["entry_template"] = "../escape.py"
    plan = Mock()
    monkeypatch.setattr(workflow, "plan_prospective", plan)
    with pytest.raises(ValueError):
        workflow.prepare_experiment_release(spec, root, json.dumps(config), output)
    plan.assert_not_called()
    assert not output.exists()


def test_duplicate_json_fields_rejected(case):
    spec, root, _, _ = case
    with pytest.raises(ValueError, match="Duplicate JSON"):
        workflow.release_workflow_inputs(spec, root, '{"format":"x","format":"y"}')
