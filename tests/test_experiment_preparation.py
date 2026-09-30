"""Integrated read-only preparation checks; no remote jobs, models or real RML writes."""
import hashlib
import json
from unittest.mock import Mock

import pytest

from molgap import experiment_cli, experiment_workflow
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_staging import UploadArtifact
from molgap.screen_policy import canonical_fingerprint
from test_experiment_spec import payload
from test_research_memory_plan_batch import _item, planning_root


@pytest.fixture
def preparation(planning_root, payload, monkeypatch):
    root, _ = planning_root
    source = root / "src/molgap/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 42\n")
    (root / "entry.py").write_text('PIN = "__PIN_SOURCE_ARCHIVE_SHA256__"\n')
    (root / "kernel.json").write_text('{"synthetic":true}')
    artifact = root / "trusted.bin"
    artifact.write_bytes(b"arbitrary trusted bytes, not a tensor state")
    payload["schema_version"] = "molgap-experiment-spec-v2"
    payload["prospective"] = {"arms": []}
    recipes = {}
    for label, arm in zip(("a", "b"), payload["arms"]):
        arm["training"]["recipe"]["sha256"] = hashlib.sha256((root / f"contract-{label}.json").read_bytes()).hexdigest()
        item = _item(label)
        item["spec"]["trajectory"]["state_at_start"]["source_config_identity"] = canonical_fingerprint(arm)
        path = root / f"plan-{label}.json"
        path.write_text(json.dumps(item["spec"]))
        payload["prospective"]["arms"].append({"arm_id": arm["arm_id"], "trajectory_id": f"T-{label}",
            "plan_spec_ref": path.name, "plan_spec_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "output": item["output"]})
        recipes[arm["arm_id"]] = f"contract-{label}.json"
    spec = ExperimentSpec(payload)
    sources = ["src/molgap/fixture.py", "entry.py", "kernel.json", *recipes.values()]
    config = {"format": experiment_workflow.WORKFLOW_FORMAT, "spec_identity": spec.identity,
        "source_paths": sources, "recipe_files": recipes,
        "artifacts": {"trusted.bin": UploadArtifact.from_file(artifact).to_workflow()},
        "initial_states": {a["arm_id"]: "trusted.bin" for a in payload["arms"]},
        "required_modules": ["molgap.fixture"], "entry_template": "entry.py", "kernel_metadata": "kernel.json",
        "dataset_metadata": None, "pickle_inputs": []}
    monkeypatch.setattr(experiment_workflow, "_tracked_paths", lambda _: set(sources))
    monkeypatch.setattr(experiment_workflow, "_assert_clean_paths", lambda *_: "b" * 40)
    return root, spec, config


def test_cli_readonly_precheck_runs_actual_rml_schema(preparation, capsys):
    root, spec, config = preparation
    spec.write(root / "spec.json")
    (root / "workflow.json").write_text(json.dumps(config))
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    code = experiment_cli.main(["check-preparation", "--spec", str(root / "spec.json"),
        "--workflow", str(root / "workflow.json"), "--repo-root", str(root)])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "LOCAL_PREPARATION_CHECKED_ONLY"
    assert result["prospective"]["status"] == "PROSPECTIVE_VALIDATED_ONLY"
    assert result["published"] is result["compute_released"] is False
    assert result["release_report_required"] is True
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("fault", ["file_digest", "missing_manifest_source", "pin", "closed_context",
                                  "recipe", "missing_initial", "module", "syntax"])
def test_integrated_prepare_blocks_fault_before_any_plan(preparation, monkeypatch, fault):
    root, spec, config = preparation
    if fault == "file_digest":
        # A valid semantic SHA is not a substitute for the transport file SHA.
        config["artifacts"]["trusted.bin"]["sha256"] = "f" * 64
    elif fault == "missing_manifest_source":
        config["source_paths"].append("src/molgap/missing.py")
    elif fault == "pin":
        (root / "entry.py").write_text("NO_PIN = True\n")
    elif fault == "recipe":
        (root / "contract-a.json").write_text("changed recipe")
    elif fault == "missing_initial":
        config["initial_states"].clear()
    elif fault == "module":
        config["required_modules"] = ["molgap.absent"]
    elif fault == "syntax":
        (root / "src/molgap/fixture.py").write_text("def invalid(\n")
    else:
        path = root / "plan-b.json"
        plan = json.loads(path.read_text())
        plan["trajectory"]["hypothesis"]["related_closed_family_ids"] = []
        path.write_text(json.dumps(plan))
        declaration = spec.to_dict()
        declaration["prospective"]["arms"][1]["plan_spec_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        spec = ExperimentSpec(declaration)
        config["spec_identity"] = spec.identity
    publisher = Mock()
    monkeypatch.setattr(experiment_workflow, "plan_prospective", publisher)
    with pytest.raises((ValueError, RuntimeError, SyntaxError)):
        experiment_workflow.prepare_experiment_release(spec, root, json.dumps(config), root / "output")
    publisher.assert_not_called()
    assert not (root / "output").exists()
    assert not (root / "experiments/arm-a").exists()
    assert not (root / "experiments/arm-b").exists()


def test_artifact_helper_keeps_file_identity_separate(preparation):
    root, spec, config = preparation
    artifact = UploadArtifact.from_file(root / "trusted.bin")
    assert artifact.sha256 == hashlib.sha256((root / "trusted.bin").read_bytes()).hexdigest()
    assert artifact.sha256 != spec.to_dict()["arms"][0]["initialization"]["state_sha256"]
    assert config["artifacts"]["trusted.bin"] == artifact.to_workflow()
