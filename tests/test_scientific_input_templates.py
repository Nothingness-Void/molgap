"""Focused CPU checks for deriving an editable input from a frozen package."""
import json
from pathlib import Path

import pytest

from molgap.experiment_launch import canonical_json
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_presets import (
    build_scientific_input_template,
    export_scientific_input_template,
)
from test_experiment_workflow import _arm, _spec_payload


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value).encode("utf-8"))


@pytest.fixture
def frozen(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    run = tmp_path / "artifacts" / "runs" / "old-run" / "v1"
    source = {
        "format": "molgap-scientific-experiment-v1",
        "experiment_id": "old-experiment",
        "logical_run_id": "old-run",
        "attempt_id": "v1",
        "machine_profile": "old-machine.json",
        "monitor_paths": ["experiments/old/STATUS.md"],
        "decision_ref": "experiments/old/decision.md",
        "question": {"prompt": "Does the addon change the gap?"},
        "profile": {"platform": "CPU", "role": "screen"},
        "authority": {"user_authorized": True},
        "authority_ref": "experiments/old/authority.json",
        "role_ref": "experiments/old/roles.json",
    }
    spec_value = _spec_payload([_arm("neural_atom_k1", "3", "candidate")])
    spec_value.update(experiment_id="old-experiment", logical_run_id="old-run")
    spec = ExperimentSpec(spec_value)
    _write(run / "inputs/experiment_spec.json", spec.to_dict())
    _write(run / "inputs/workflow_plan.json", {
        "spec_identity": spec.identity,
        "configuration_source": {"path": "experiments/old/input.json"},
    })
    (run / "prepared/package").mkdir(parents=True)
    return repo, run, source, spec.identity


def test_template_copies_science_and_replaces_only_fresh_run_facts(frozen, monkeypatch, tmp_path):
    repo, run, source, spec_identity = frozen
    import molgap.experiment_package as package

    monkeypatch.setattr(
        package,
        "verify_experiment_source_package",
        lambda package_dir, *, _verification=None: {
            "spec_identity": spec_identity,
            "configuration_source": {"path": "experiments/old/input.json"}
        },
    )
    monkeypatch.setattr(
        package,
        "read_packaged_text",
        lambda package_dir, relative_path, *, _verification=None: canonical_json(source),
    )

    template = build_scientific_input_template(
        run, repo, "new-experiment", "new-run", machine_profile="new-machine.json"
    )
    assert template["experiment_id"] == "new-experiment"
    assert template["logical_run_id"] == "new-run"
    assert template["attempt_id"] == "v1"
    assert template["machine_profile"] == "new-machine.json"
    assert template["question"] == source["question"]
    assert template["profile"] == source["profile"]
    assert template["monitor_paths"] == ["experiments/new-experiment/STATUS.md"]
    assert template["decision_ref"] == "experiments/new-experiment/decision.md"
    for field in ("authority", "authority_ref", "role_ref"):
        assert field not in template

    output = tmp_path / "new-input.json"
    assert export_scientific_input_template(output, template, frozen_run=run) == output
    assert json.loads(output.read_bytes()) == template
    with pytest.raises(ValueError, match="fresh"):
        export_scientific_input_template(output, template, frozen_run=run)
    with pytest.raises(ValueError, match="outside the frozen run"):
        export_scientific_input_template(run / "inputs/new.json", template, frozen_run=run)


def test_template_requires_fresh_experiment_and_logical_ids(frozen, monkeypatch):
    repo, run, source, spec_identity = frozen
    import molgap.experiment_package as package

    monkeypatch.setattr(
        package,
        "verify_experiment_source_package",
        lambda package_dir, *, _verification=None: {
            "spec_identity": spec_identity,
            "configuration_source": {"path": "experiments/old/input.json"}
        },
    )
    monkeypatch.setattr(
        package,
        "read_packaged_text",
        lambda package_dir, relative_path, *, _verification=None: canonical_json(source),
    )
    with pytest.raises(ValueError, match="fresh"):
        build_scientific_input_template(run, repo, "new-experiment", "old-run")
