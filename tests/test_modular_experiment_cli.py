"""New local commands delegate to the existing owners; no submission/model IO."""
from unittest.mock import Mock
import json

from molgap import experiment_cli as cli
from test_experiment_cli import invoke as _invoke, spec_file
from test_experiment_spec import payload


def test_dependency_command_is_read_only(capsys, spec_file, tmp_path, monkeypatch):
    import molgap.source_dependencies as dependencies
    report = dependencies.DependencyClosure(("src/molgap/owner.py",),
        ("src/molgap/owner.py", "src/molgap/helper.py"), ("torch",), (), ())
    inspect = Mock(return_value=report)
    monkeypatch.setattr(dependencies, "registered_dependencies", inspect)
    result = _invoke(capsys, ["source-dependencies", "--spec", spec_file,
        "--repo-root", tmp_path, "--extra-source", "src/molgap/addon.py", "--strict-dynamic"])
    assert result["submitted"] is False and result["execution_qualification"] == "NOT_EVALUATED"
    assert inspect.call_args.kwargs == {"require_static": True}
    assert inspect.call_args.args[2] == ["src/molgap/addon.py"]


def test_auto_package_delegates_existing_immutable_packager(capsys, spec_file, tmp_path, monkeypatch):
    import molgap.experiment_source_inventory as inventory
    selected = ["src/molgap/owner.py", "src/molgap/addon.py"]
    discover = Mock(return_value=selected)
    package = Mock(return_value={"status": "source_packaged"})
    monkeypatch.setattr(inventory, "registered_source_files", discover)
    monkeypatch.setattr(cli, "build_experiment_source_package", package)
    output = tmp_path / "package"
    _invoke(capsys, ["package", "--spec", spec_file, "--repo-root", tmp_path,
        "--output", output, "--auto-dependencies"])
    assert discover.call_args.kwargs == {"repo_root": tmp_path, "auto_dependencies": True}
    assert package.call_args.args[2:] == (selected, output)


def test_auto_family_preparation_reuses_existing_plan_and_entry(capsys, spec_file, tmp_path, monkeypatch):
    import molgap.experiment_source_inventory as inventory
    import molgap.experiment_workflow as workflow
    plan_path = tmp_path / "plan.json"
    plan = {"source_files": ["src/molgap/custom.py"], "arms": [], "format": "fixture"}
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    discover = Mock(return_value=["src/molgap/custom.py", "src/molgap/helper.py"])
    prepare = Mock(return_value={"status": "SOURCE_INPUTS_PREPARED"})
    monkeypatch.setattr(inventory, "registered_source_files", discover)
    monkeypatch.setattr(workflow, "prepare_family_training", prepare)
    _invoke(capsys, ["prepare-family-training", "--spec", spec_file, "--repo-root", tmp_path,
        "--plan", plan_path, "--output", tmp_path / "prepared", "--auto-dependencies"])
    assert discover.call_args.args[1] == plan["source_files"]
    assert prepare.call_args.args[2] == {**plan, "source_files": discover.return_value}


def test_auto_and_manual_package_inputs_are_exclusive(capsys, spec_file, tmp_path):
    code = cli.main(["package", "--spec", str(spec_file), "--repo-root", str(tmp_path),
        "--output", str(tmp_path / "out"), "--auto-dependencies", "--allowlist", "README.md"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "ERROR"


def test_sandbox_cli_delegates_without_dataset_or_platform_inputs(capsys, monkeypatch):
    import molgap.fast_sandbox as sandbox
    run = Mock(return_value={"status": "PASSED", "device": "cpu"})
    monkeypatch.setattr(sandbox, "run_family_sandbox", run)
    assert _invoke(capsys, ["sandbox", "--family", "gptrans_t"])["status"] == "PASSED"
    run.assert_called_once_with("gptrans_t")


def test_spec_sandbox_requires_arm(capsys, spec_file):
    assert cli.main(["sandbox", "--spec", str(spec_file)]) == 2
    assert "explicit --arm" in json.loads(capsys.readouterr().out)["error"]["message"]
