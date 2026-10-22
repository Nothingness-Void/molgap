"""Run-ID delegation over real synthetic packages/receipts; no remote access."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from molgap import experiment_cli as cli
from molgap import experiment_run as runs
from molgap.common.paths import MachinePaths
from molgap.experiment_input import locate_run
from molgap.experiment_launch import canonical_json
from molgap.experiment_package import build_experiment_source_package
from molgap.experiment_spec import ExperimentSpec
from molgap.screen_policy import canonical_fingerprint
from test_experiment_launch import build, envelope, known
from test_experiment_package import repo
from test_experiment_spec import payload


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value).encode())


@pytest.fixture
def frozen_run(tmp_path, repo, payload):
    profile = tmp_path / "machine.json"
    write(profile, {"format": "molgap-machine-paths-v1", "python": sys.executable,
        **{name: str(tmp_path / name) for name in ("data_root", "artifact_root", "scratch_root")}})
    spec = ExperimentSpec(payload)
    run = MachinePaths.from_file(profile).run_directory("synthetic-run", "v1")
    prepared = run / "prepared"
    package = prepared / "package"
    manifest = build_experiment_source_package(spec, repo, ["src/example.py", "README.md"], package)
    plan = {"spec_identity": spec.identity,
            "platform": {"source_dataset": "synthetic-owner/source"}}
    binding = {"format": "molgap-experiment-input-v1", "input_identity": "f" * 64,
        "spec_identity": spec.identity, "workflow_identity": canonical_fingerprint(plan),
        "authority_ref": "authority.json", "monitor_paths": ["experiments/question/STATUS.md"],
        "source_commit": manifest["source_commit"]}
    for name, value in (("input_binding.json", binding), ("experiment_spec.json", spec.to_dict()),
                        ("workflow_plan.json", plan)):
        write(run / "inputs" / name, value)
    write(prepared / "workflow_report.json", {"status": "PREPARED_FOR_PLATFORM",
        "spec_identity": spec.identity, "package_identity": manifest["package_identity"],
        "source_commit": manifest["source_commit"]})
    context = spec, package, manifest["package_identity"]
    response = envelope(context)
    response["platform_version"] = known("7")
    response["physical_runs"] = known([{"arm_ids": [a["arm_id"] for a in spec.to_dict()["arms"]],
        "run_identity": "kernel1/version7", "canonical_reference": "synthetic-owner/workload",
        "platform_version": known("7")}])
    receipt = build(context, response)
    write(run / "operations/launch_receipt.json", receipt)
    write(run / "operations/submission_state.json", {"phase": "BOUND", "binding": {
        "package_identity": manifest["package_identity"], "kernel": "synthetic-owner/workload",
        "dataset": "synthetic-owner/source", "authority_ref": binding["authority_ref"],
        "monitor_paths": binding["monitor_paths"]},
        "kernel_response": {"kernel": "synthetic-owner/workload", "version_number": 7}})
    return SimpleNamespace(profile=profile, repo=repo, run=run, prepared=prepared,
        package=package, spec=spec, manifest=manifest, receipt=receipt,
        original=(profile, repo, "synthetic-run", "v1"))


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_locate_run_reuses_validation_only_within_one_lookup(frozen_run, monkeypatch):
    from molgap import experiment_package as packages

    case = frozen_run
    full_validation = Mock(wraps=packages._verify_package_snapshot)
    monkeypatch.setattr(packages, "_verify_package_snapshot", full_validation)
    for expected_calls in (1, 2):
        assert locate_run(*case.original)["receipt"] == case.receipt
        assert full_validation.call_count == expected_calls

    archive = case.package / "source.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        locate_run(*case.original)
    assert full_validation.call_count == 3


def test_locate_run_rechecks_package_before_receipt_validation(frozen_run, monkeypatch):
    from molgap import experiment_launch as launch
    from molgap import experiment_package as packages

    case = frozen_run
    full_validation = Mock(wraps=packages._verify_package_snapshot)
    monkeypatch.setattr(packages, "_verify_package_snapshot", full_validation)
    original_read = launch.read_launch_receipt

    def changed_before_receipt(*args, **kwargs):
        assert full_validation.call_count == 1
        archive = case.package / "source.tar.gz"
        archive.write_bytes(archive.read_bytes() + b"tampered")
        return original_read(*args, **kwargs)

    monkeypatch.setattr(launch, "read_launch_receipt", changed_before_receipt)
    with pytest.raises(ValueError):
        locate_run(*case.original)
    assert full_validation.call_count == 2


def test_independent_receipt_reads_and_writes_keep_full_validation(frozen_run, tmp_path, monkeypatch):
    from molgap import experiment_launch as launch
    from molgap import experiment_package as packages

    case = frozen_run
    full_validation = Mock(wraps=packages._verify_package_snapshot)
    monkeypatch.setattr(packages, "_verify_package_snapshot", full_validation)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    for expected_calls in (2, 4):
        assert launch.read_launch_receipt(case.run / "operations/launch_receipt.json",
            case.spec, case.package, expected_package_identity=case.manifest["package_identity"]) == case.receipt
        launch.write_launch_receipt(canonical_json(case.receipt), receipts, case.spec,
            case.package, expected_package_identity=case.manifest["package_identity"])
        assert full_validation.call_count == expected_calls


def test_inspect_resolves_original_run_and_explicit_external_output(frozen_run, tmp_path, monkeypatch):
    case = frozen_run
    before = snapshot(case.run)
    locator = Mock(wraps=locate_run)
    monkeypatch.setattr(runs, "locate_run", locator)
    owner = Mock(return_value={"status": "MECHANICALLY_VERIFIED"})
    monkeypatch.setattr("molgap.experiment_family_workflow.inspect_output", owner)
    expected = {"frozen_expectations": True}
    output = tmp_path / "retrieved/gptrans"
    result = runs.inspect_run_output(*case.original, arm_id="gptrans_t",
        artifact_root=output, expectations=expected)
    assert result == owner.return_value
    locator.assert_called_once_with(*case.original)
    args, kwargs = owner.call_args
    assert args == (output,)
    assert kwargs["expected"] is expected
    assert kwargs["context"].source_commit == case.manifest["source_commit"]
    assert kwargs["context"].platform_version == "7"
    assert snapshot(case.run) == before


@pytest.mark.parametrize("operation", ["accept", "build", "prepare"])
def test_existing_owner_receives_original_bindings_and_explicit_plan(frozen_run, tmp_path, monkeypatch, operation):
    case = frozen_run
    before = snapshot(case.run)
    owner = Mock(return_value={"status": "OWNER_RESULT", "executed": False})
    if operation == "accept":
        monkeypatch.setattr("molgap.experiment_workflow.accept_workflow", owner)
        supplied, locations = {"original_outputs": True}, {"original_scientific_plan": True}
        result = runs.accept_run_workflow(*case.original, supplied=supplied, locations=locations)
        assert owner.call_args.kwargs["supplied"] is supplied
        assert owner.call_args.kwargs["locations"] is locations
        assert owner.call_args.kwargs["execute"] is False
    elif operation == "build":
        monkeypatch.setattr("molgap.experiment_workflow_resume.build_workflow_resume", owner)
        result = runs.build_run_resume(*case.original, arm_id="gptrans_t",
            source_output=tmp_path / "retrieved", output=tmp_path / "bundle")
        assert owner.call_args.kwargs["source_output"] == tmp_path / "retrieved"
        assert owner.call_args.kwargs["prepared"] == case.prepared
    else:
        monkeypatch.setattr("molgap.experiment_workflow_resume.prepare_resumed_workflow", owner)
        plan = {"original_resume_plan": True}
        result = runs.prepare_run_resume(*case.original, plan=plan, output=tmp_path / "recovery")
        assert owner.call_args.kwargs["plan"] is plan
        assert owner.call_args.kwargs["prepared"] == case.prepared
    assert result == owner.return_value
    assert owner.call_args.args[0] == case.spec
    assert owner.call_args.kwargs["package_dir"] == case.package
    assert owner.call_args.kwargs["expected_package_identity"] == case.manifest["package_identity"]
    assert owner.call_args.kwargs["receipt_path"] == case.run / "operations/launch_receipt.json"
    assert snapshot(case.run) == before


@pytest.mark.parametrize("phase,operation", [(phase, "inspect") for phase in
    ("SUBMIT_UNKNOWN", "KERNEL_SUBMITTING", "KERNEL_ACCEPTED", "SOURCE_UNKNOWN", "UNKNOWN")]
    + [("SUBMIT_UNKNOWN", operation) for operation in ("accept", "build", "prepare")])
def test_unknown_journal_never_calls_owner_or_mutates(frozen_run, tmp_path, monkeypatch, phase, operation):
    case = frozen_run
    path = case.run / "operations/submission_state.json"
    journal = json.loads(path.read_bytes())
    journal["phase"] = phase
    write(path, journal)
    owner = Mock(side_effect=AssertionError("Reconcile only"))
    for name in ("molgap.experiment_family_workflow.inspect_output",
                 "molgap.experiment_workflow.accept_workflow",
                 "molgap.experiment_workflow_resume.build_workflow_resume",
                 "molgap.experiment_workflow_resume.prepare_resumed_workflow"):
        monkeypatch.setattr(name, owner)
    actions = {
        "inspect": lambda: runs.inspect_run_output(*case.original, arm_id="gptrans_t",
            artifact_root=tmp_path / "retrieved", expectations={}),
        "accept": lambda: runs.accept_run_workflow(*case.original, supplied={}, locations={}),
        "build": lambda: runs.build_run_resume(*case.original, arm_id="gptrans_t",
            source_output=tmp_path / "retrieved", output=tmp_path / "bundle"),
        "prepare": lambda: runs.prepare_run_resume(*case.original, plan={}, output=tmp_path / "recovery"),
    }
    before = snapshot(case.run)
    with pytest.raises(runs.RunReconciliationRequired, match="original submission journal"):
        actions[operation]()
    owner.assert_not_called()
    assert snapshot(case.run) == before


@pytest.mark.parametrize("fault", ["spec", "workflow", "package", "receipt", "prepared", "source", "bound_source", "journal_binding", "journal_version"])
def test_tampered_original_fails_before_owner(frozen_run, monkeypatch, fault):
    case = frozen_run
    if fault == "package":
        path = case.package / "source.tar.gz"
        path.write_bytes(path.read_bytes() + b"tampered")
    else:
        name = {"spec": "inputs/experiment_spec.json", "workflow": "inputs/workflow_plan.json",
            "receipt": "operations/launch_receipt.json", "prepared": "prepared/workflow_report.json",
            "source": "prepared/workflow_report.json", "bound_source": "inputs/input_binding.json",
            "journal_binding": "operations/submission_state.json",
            "journal_version": "operations/submission_state.json"}[fault]
        path = case.run / name
        value = json.loads(path.read_bytes())
        if fault == "spec": value["logical_run_id"] = "other-run"
        elif fault == "workflow": value["spec_identity"] = "0" * 64
        elif fault == "receipt": value["receipt_identity"] = "0" * 64
        elif fault == "prepared": value["package_identity"] = "0" * 64
        elif fault in {"source", "bound_source"}: value["source_commit"] = "0" * 40
        elif fault == "journal_binding": value["binding"]["package_identity"] = "0" * 64
        else: value["kernel_response"]["version_number"] = 8
        write(path, value)
    before = snapshot(case.run)
    owner = Mock(side_effect=AssertionError("Tampered input cannot reach acceptance"))
    monkeypatch.setattr("molgap.experiment_workflow.accept_workflow", owner)
    with pytest.raises(ValueError):
        runs.accept_run_workflow(*case.original, supplied={}, locations={})
    owner.assert_not_called()
    assert snapshot(case.run) == before


@pytest.mark.parametrize("state", ["missing", "unknown"])
def test_missing_or_unreconciled_receipt_requires_reconciliation(frozen_run, monkeypatch, state):
    case = frozen_run
    path = case.run / "operations/launch_receipt.json"
    if state == "missing":
        path.unlink()
    else:
        response = envelope((case.spec, case.package, case.manifest["package_identity"]), "unknown")
        write(path, build((case.spec, case.package, case.manifest["package_identity"]), response))
    owner = Mock(side_effect=AssertionError("No original observed receipt"))
    monkeypatch.setattr("molgap.experiment_workflow.accept_workflow", owner)
    with pytest.raises(runs.RunReconciliationRequired, match="observed launch receipt"):
        runs.accept_run_workflow(*case.original, supplied={}, locations={})
    owner.assert_not_called()


@pytest.mark.parametrize("destination", ["checkout", "inputs", "prepared", "operations", "existing"])
@pytest.mark.parametrize("operation", ["build", "prepare"])
def test_resume_destination_cannot_change_checkout_or_frozen_records(frozen_run, tmp_path, monkeypatch, destination, operation):
    case = frozen_run
    output = {"checkout": case.repo / "resume", "inputs": case.run / "inputs/resume",
        "prepared": case.prepared / "resume", "operations": case.run / "operations/resume",
        "existing": tmp_path / "existing"}[destination]
    if destination == "existing": output.mkdir()
    owner = Mock(side_effect=AssertionError("Invalid destination"))
    monkeypatch.setattr("molgap.experiment_workflow_resume.build_workflow_resume", owner)
    monkeypatch.setattr("molgap.experiment_workflow_resume.prepare_resumed_workflow", owner)
    with pytest.raises(ValueError, match="outside|fresh"):
        if operation == "build":
            runs.build_run_resume(*case.original, arm_id="gptrans_t", source_output=tmp_path / "retrieved", output=output)
        else:
            runs.prepare_run_resume(*case.original, plan={}, output=output)
    owner.assert_not_called()


@pytest.mark.parametrize("command,status,owner_path", [
    ("inspect-output", "MECHANICALLY_VERIFIED", "molgap.experiment_family_workflow.inspect_output"),
    ("accept-workflow", "MECHANICALLY_VERIFIED", "molgap.experiment_workflow.accept_workflow"),
    ("build-resume", "RESUME_VERIFIED", "molgap.experiment_workflow_resume.build_workflow_resume"),
    ("prepare-resume", "PREPARED_FOR_PLATFORM", "molgap.experiment_workflow_resume.prepare_resumed_workflow"),
])
@pytest.mark.parametrize("identity", ["spec", "run"])
def test_cli_old_and_run_id_delegate_same_original_bindings(frozen_run, tmp_path, monkeypatch, capsys, command, status, owner_path, identity):
    case = frozen_run
    for name in ("expectations", "outputs", "locations", "resume_plan"):
        write(tmp_path / (name + ".json"), {})
    args = [command, "--repo-root", str(case.repo)]
    if identity == "run":
        args += ["--run-id", "synthetic-run", "--attempt-id", "v1", "--profile", str(case.profile)]
    else:
        args += ["--spec", str(case.run / "inputs/experiment_spec.json"), "--package", str(case.package),
            "--expected-package-identity", case.manifest["package_identity"],
            "--receipt", str(case.run / "operations/launch_receipt.json")]
        if command in {"build-resume", "prepare-resume"}: args += ["--prepared", str(case.prepared)]
    if command == "inspect-output":
        args += ["--arm", "gptrans_t", "--artifact-root", str(tmp_path / "retrieved"),
            "--expectations", str(tmp_path / "expectations.json")]
    elif command == "accept-workflow":
        args += ["--outputs", str(tmp_path / "outputs.json"), "--locations", str(tmp_path / "locations.json")]
    elif command == "build-resume":
        args += ["--arm", "gptrans_t", "--source-output", str(tmp_path / "retrieved"),
            "--output", str(tmp_path / "bundle")]
    else:
        args += ["--resume-plan", str(tmp_path / "resume_plan.json"), "--output", str(tmp_path / "recovery")]
    owner = Mock(return_value={"status": status})
    monkeypatch.setattr(owner_path, owner)
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out) == {"status": status}
    owner.assert_called_once()
    if command != "inspect-output":
        assert owner.call_args.kwargs["package_dir"] == case.package
        assert owner.call_args.kwargs["expected_package_identity"] == case.manifest["package_identity"]


def test_cli_unknown_journal_is_reconcile_only(frozen_run, tmp_path, monkeypatch, capsys):
    case = frozen_run
    path = case.run / "operations/submission_state.json"
    journal = json.loads(path.read_bytes())
    journal["phase"] = "SUBMIT_UNKNOWN"
    write(path, journal)
    write(tmp_path / "expectations.json", {})
    owner = Mock(side_effect=AssertionError("Uncertainty cannot be bypassed"))
    monkeypatch.setattr("molgap.experiment_family_workflow.inspect_output", owner)
    assert cli.main(["inspect-output", "--run-id", "synthetic-run", "--attempt-id", "v1",
        "--profile", str(case.profile), "--repo-root", str(case.repo), "--arm", "gptrans_t",
        "--artifact-root", str(tmp_path / "retrieved"), "--expectations", str(tmp_path / "expectations.json")]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "RECONCILIATION_REQUIRED"
    assert result["executed"] is False and result["network_accessed"] is False
    owner.assert_not_called()


@pytest.mark.parametrize("identity_args", [[], ["--spec", "SPEC"], ["--run-id", "synthetic-run"],
    ["--run-id", "synthetic-run", "--profile", "PROFILE", "--attempt-id", "v1", "--package", "PACKAGE"],
    ["--spec", "SPEC", "--profile", "PROFILE"]])
def test_cli_missing_or_conflicting_identity_arguments_fail(tmp_path, capsys, identity_args):
    args = [str(tmp_path / word.lower()) if word in {"SPEC", "PROFILE", "PACKAGE"} else word
            for word in identity_args]
    write(tmp_path / "expectations.json", {})
    assert cli.main(["inspect-output", *args, "--repo-root", str(tmp_path), "--arm", "gptrans_t",
        "--artifact-root", str(tmp_path / "retrieved"), "--expectations", str(tmp_path / "expectations.json")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "ERROR"
