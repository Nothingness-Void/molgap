"""Thin local acceptance entry: explicit decisions and no hand-authored maps."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from molgap import experiment_cli as cli
from molgap import experiment_run as runs


def test_run_acceptance_cli_needs_no_outputs_or_locations(tmp_path, monkeypatch, capsys):
    owner = Mock(return_value={"status": "MECHANICALLY_VERIFIED", "arms": {}, "executed": False})
    monkeypatch.setattr(runs, "accept_run_workflow", owner)
    assert cli.main(["accept-workflow", "--run-id", "test", "--attempt-id", "v1",
        "--repo-root", str(tmp_path), "--profile", str(tmp_path / "profile.json"),
        "--retrieved-root", str(tmp_path / "retrieved")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "MECHANICALLY_VERIFIED"
    assert owner.call_args.kwargs["supplied"] is None
    assert owner.call_args.kwargs["locations"] is None
    assert owner.call_args.kwargs["retrieved_root"] == tmp_path / "retrieved"
    assert owner.call_args.kwargs["execute"] is False


def test_acceptance_script_selects_checkout_and_analysis_defaults(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    module_spec = importlib.util.spec_from_file_location("local_accept_run", root / "platforms/local/accept_run.py")
    entry = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(entry)
    owner = Mock(return_value=0)
    monkeypatch.setattr(cli, "main", owner)
    assert entry.main(["--run-id", "test", "--attempt-id", "v1"]) == 0
    arguments = owner.call_args.args[0]
    assert arguments[:3] == ["accept-workflow", "--diagnose", "--stage-terminal"]
    assert arguments[arguments.index("--repo-root") + 1] == str(root)
    assert "--execute" not in arguments


def test_diagnose_run_routes_json_output_and_defaults_no_bootstrap(tmp_path, monkeypatch, capsys):
    from molgap import experiment_attribution as attribution
    owner = Mock(return_value={"status": "DIAGNOSED", "arms": {}, "comparisons": {},
                              "scientific_acceptance": "NOT_EVALUATED"})
    monkeypatch.setattr(attribution, "diagnose_run", owner)
    assert cli.main(["diagnose", "--run-id", "test", "--attempt-id", "v1",
        "--profile", str(tmp_path / "machine.json"), "--repo-root", str(tmp_path),
        "--retrieved-root", str(tmp_path / "retrieved"), "--output", str(tmp_path / "diagnosis.json")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["scientific_acceptance"] == "NOT_EVALUATED"
    assert owner.call_args.kwargs["n_bootstrap"] == 0
    assert owner.call_args.kwargs["output"] == tmp_path / "diagnosis.json"


@pytest.mark.parametrize("extra", [["--control", "unused"], ["--threshold-ev", "0.1"]])
def test_run_diagnosis_rejects_ambiguous_or_unfrozen_policy(extra, tmp_path, capsys):
    assert cli.main(["diagnose", "--run-id", "test", "--attempt-id", "v1",
        "--profile", str(tmp_path / "machine.json"), *extra]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "ERROR"


def test_outcome_cannot_be_supplied_without_decision(tmp_path):
    with pytest.raises(ValueError, match="provided together"):
        runs.accept_run_workflow(None, tmp_path, "test", "v1", outcome="INCONCLUSIVE")


def test_workflow_inspection_does_not_build_a_terminal_package(tmp_path, monkeypatch):
    from molgap import experiment_family_workflow as family
    from molgap import experiment_workflow as workflow
    monkeypatch.setattr(workflow, "prepare_terminal_outputs", lambda *a, **k: {"arm": {}})
    monkeypatch.setattr(family, "inspect_terminal_output", lambda item: {"status": "MECHANICALLY_VERIFIED"})
    descriptor = Mock(side_effect=AssertionError("Mechanical inspection must not need terminal.json"))
    monkeypatch.setattr(workflow, "build_verified_terminal_descriptor", descriptor)
    result = workflow.accept_workflow(None, tmp_path, {}, receipt_path=None, package_dir=None,
        expected_package_identity="f" * 64)
    assert result["status"] == "MECHANICALLY_VERIFIED"
    assert result["scientific_outcome"] == "NOT_EVALUATED"
    descriptor.assert_not_called()
