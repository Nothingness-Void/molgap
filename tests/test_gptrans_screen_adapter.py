"""Argument wiring only; no trainer/model imports or accelerator operations."""
import copy

import pytest

from molgap.experiment_spec import ExperimentSpec
from molgap.gptrans_screen_adapter import gptrans_screen_arguments
from test_experiment_spec import payload, addon


@pytest.fixture
def screen(payload, tmp_path, monkeypatch):
    import molgap.gptrans_screen_adapter as module
    payload["schema_version"] = "molgap-experiment-spec-v2"
    degree = payload["arms"][0]
    degree["arm_id"] = "degree"
    degree["scientific_role"] = "candidate"
    degree["addons"] = [addon("degree_scale")]
    degree["addon_semantics"] = "ordered"
    path = copy.deepcopy(degree)
    path["arm_id"] = "path"
    path["addons"] = [addon("path_bond_mean")]
    payload["arms"] = [degree, path]
    payload["prospective"] = {"arms": [
        {"arm_id": name, "trajectory_id": "TC-" + name,
         "plan_spec_ref": f"experiments/synthetic/{name}/plan.json", "plan_spec_sha256": "a" * 64,
         "output": f"experiments/synthetic/{name}/rml"} for name in ("degree", "path")
    ]}
    spec = ExperimentSpec(payload)
    monkeypatch.setattr(module, "verify_experiment_source_package", lambda _: {
        "spec_identity": spec.identity, "archive_sha256": "b" * 64, "source_commit": "a" * 40,
    })
    return dict(spec=spec, package_dir=tmp_path / "package", dataset_root=tmp_path / "dataset",
                manifest_path=tmp_path / "dataset/manifest.json", initial_state_path=tmp_path / "initial.pt",
                target_transform_path=tmp_path / "transform.json", platform_id="synthetic-platform")


def test_independent_inputs_trace_and_run_id(screen, tmp_path):
    degree = gptrans_screen_arguments(**screen, arm_id="degree")
    path = gptrans_screen_arguments(**screen, arm_id="path", path_sidecar_root=tmp_path / "sidecar")
    assert degree["training"]["trajectory_id"] == "TC-degree"
    assert path["training"]["trajectory_id"] == "TC-path"
    assert degree["training"]["logical_run_id"] != path["training"]["logical_run_id"]
    assert degree["preflight"]["path_sidecar_root"] is None
    assert path["preflight"]["path_sidecar_root"] == tmp_path / "sidecar"
    assert degree["training"]["v5_audit"] is path["training"]["v5_audit"] is True
    assert degree["preflight"]["variant"] == "degree_scale"
    assert path["preflight"]["variant"] == "path_bond_mean"
    assert "preflight_path" not in degree["training"]
    assert "output" not in degree["training"]


@pytest.mark.parametrize("arm_id,sidecar", [("degree", "unexpected"), ("path", None)])
def test_sidecar_not_silently_consumed_by_another_arm(screen, arm_id, sidecar):
    with pytest.raises(ValueError, match="sidecar"):
        gptrans_screen_arguments(**screen, arm_id=arm_id, path_sidecar_root=sidecar)


def test_other_spec_package_rejected(screen, monkeypatch):
    monkeypatch.setattr("molgap.gptrans_screen_adapter.verify_experiment_source_package", lambda _: {
        "spec_identity": "0" * 64})
    with pytest.raises(ValueError, match="differs"):
        gptrans_screen_arguments(**screen, arm_id="degree")


def test_registration_does_not_authorize_unsupported_v5_training(screen):
    declaration = screen["spec"].to_dict()
    declaration["arms"][0]["addons"] = [addon("pair_prenorm")]
    screen["spec"] = ExperimentSpec(declaration)
    with pytest.raises(ValueError, match="no qualified"):
        gptrans_screen_arguments(**screen, arm_id="degree")


def test_existing_trainer_signatures_accept_arguments(screen):
    import ast
    from pathlib import Path
    source = Path(__file__).parents[1] / "src/molgap/pcqm_gptrans_v4.py"
    definitions = {node.name: node for node in ast.parse(source.read_bytes()).body if isinstance(node, ast.FunctionDef)}
    arguments = gptrans_screen_arguments(**screen, arm_id="degree")
    for stage, function in (("preflight", "run_preflight"), ("training", "run_training")):
        names = {arg.arg for arg in definitions[function].args.kwonlyargs}
        assert set(arguments[stage]) <= names


def test_non_v2_spec_cannot_lose_arm_trace_identity(payload, screen):
    payload["schema_version"] = "molgap-experiment-spec-v1"
    payload["prospective"] = {
        "trajectory_id": "TC-shared", "hypothesis": "synthetic", "cheapest_falsifier": "synthetic",
        "stop_rule": "stop", "budget_sha256": "a" * 64}
    screen["spec"] = ExperimentSpec(payload)
    with pytest.raises(ValueError, match="Unknown arm|Spec v2"):
        gptrans_screen_arguments(**screen, arm_id="degree")
