"""Original run-ID retrieval CLI: synthetic packages, fake API, no remote access."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from molgap import experiment_run as runs
from molgap.experiment_family_workflow import ROLES, RunContext
from test_experiment_run import frozen_run, repo, snapshot, write
from test_experiment_spec import payload as base_payload
from test_kaggle_family_retrieval import (
    _Api, _File, _Http, _ResponsePage, _module,
)


@pytest.fixture
def payload(base_payload, request):
    # Select the platform before the shared fixture freezes any identity bytes.
    declaration = deepcopy(base_payload)
    if getattr(request, "param", "kaggle") == "kaggle":
        declaration["platform"].update(name="kaggle", accelerator="NvidiaTeslaT4", device_count=2)
    return declaration


@pytest.fixture
def retrieval(frozen_run, tmp_path, monkeypatch):
    case = frozen_run
    case.adapter = _module(monkeypatch)
    case.context = RunContext.from_launch(case.spec, case.run / "operations/launch_receipt.json",
        case.package, expected_package_identity=case.manifest["package_identity"], arm_id="gptrans_t")
    case.remote = "runs/version7/gptrans/output_manifest.json"
    case.destination = case.run.parents[2] / "retrieved/gptrans"
    case.local = tmp_path / "independently-pinned-manifest.json"
    case.output = {"format": "molgap-family-output-v1", "adapter": "gptrans-v1",
        "context": case.context.to_dict(), "artifacts": {}}
    case.payloads = {}
    case.remote_files = []
    for role in sorted(ROLES):
        data = ("synthetic-" + role).encode()
        relative = "nested/" + role + ".bin"
        case.output["artifacts"][role] = {"path": relative,
            "sha256": hashlib.sha256(data).hexdigest()}
        url = "https://signed.invalid/" + role + "?token=SECRET"
        case.remote_files.append(_File("runs/version7/gptrans/" + relative, url))
        case.payloads[url] = data
    write(case.local, case.output)
    manifest_url = "https://signed.invalid/manifest?token=SECRET"
    case.remote_files.append(_File(case.remote, manifest_url))
    case.payloads[manifest_url] = case.local.read_bytes()
    case.api = _Api(case.context.account, [_ResponsePage(case.remote_files)])
    case.http = _Http(case.payloads)
    case.auth = Mock(return_value=case.api)
    monkeypatch.setattr("molgap.kaggle_submission.connect_kaggle", case.auth)
    case.original_retrieve = case.adapter.retrieve_family_outputs
    case.retrieve = Mock(side_effect=lambda api, **kwargs:
        case.original_retrieve(api, http_session=case.http, **kwargs))
    monkeypatch.setattr(case.adapter, "retrieve_family_outputs", case.retrieve)
    return case


def arguments(case, *, execute=False):
    result = ["--run-id", "synthetic-run", "--attempt-id", "v1",
        "--profile", str(case.profile), "--repo-root", str(case.repo), "--arm", "gptrans_t",
        "--local-manifest", str(case.local), "--manifest-sha256",
        hashlib.sha256(case.local.read_bytes()).hexdigest(),
        "--remote-manifest-path", case.remote, "--destination", str(case.destination)]
    if execute:
        # This path never exists; only the fake authentication owner receives it.
        result += ["--execute", "--credentials", str(case.profile.parent / "unread-credentials.json")]
    return result


def test_no_execute_verifies_actual_original_context_without_auth_or_writes(retrieval, monkeypatch, capsys):
    case = retrieval
    assert case.spec.to_dict()["platform"]["name"] == case.context.platform == "kaggle"
    assert case.spec.to_dict()["platform"]["accelerator"] == "NvidiaTeslaT4"
    before = snapshot(case.run)
    original = Mock(wraps=runs._original_run)
    monkeypatch.setattr(runs, "_original_run", original)
    assert case.adapter.main(arguments(case)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "READY_FOR_RETRIEVAL"
    assert result["producer_version"] == "7"
    assert result["executed"] is result["network_accessed"] is result["scientific_acceptance"] is False
    original.assert_called_once_with(case.profile, case.repo, "synthetic-run", "v1")
    case.auth.assert_not_called()
    case.retrieve.assert_not_called()
    assert case.api.client_builds == 0 and case.http.urls == []
    assert not case.destination.exists()
    assert snapshot(case.run) == before


@pytest.mark.parametrize("payload", ["local"], indirect=True)
def test_local_platform_is_rejected_before_manifest_auth_or_writes(retrieval, monkeypatch, capsys):
    case = retrieval
    spec, _, _ = runs._original_run(*case.original)
    assert spec.to_dict()["platform"]["name"] == case.context.platform == "local"
    manifest_check = Mock(side_effect=AssertionError("Local runs cannot reach retrieval preflight"))
    monkeypatch.setattr(case.adapter, "_family_files", manifest_check)
    before = snapshot(case.run)
    assert case.adapter.main(arguments(case, execute=True)) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "BLOCKED" and result["stage"] == "original_run"
    manifest_check.assert_not_called()
    case.auth.assert_not_called()
    case.retrieve.assert_not_called()
    assert case.api.client_builds == 0 and case.http.urls == []
    assert not case.destination.exists() and snapshot(case.run) == before


def test_execute_delegates_original_auth_and_retriever_and_is_idempotent(retrieval, capsys):
    case = retrieval
    before = snapshot(case.run)
    assert case.adapter.main(arguments(case, execute=True)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "RETRIEVED" and result["executed"] is True
    assert result["producer_version"] == result["receipt_version"] == "7"
    assert result["scientific_acceptance"] is False
    assert result["platform_version_verified_by_output_api"] is False
    assert result["unselected_outputs_downloaded"] is False
    case.auth.assert_called_once_with(case.profile.parent / "unread-credentials.json")
    case.retrieve.assert_called_once()
    assert case.retrieve.call_args.kwargs["context"] == case.context
    assert len(case.http.urls) == len(ROLES) + 1
    retained = snapshot(case.destination)
    assert retained["output_manifest.json"] == case.local.read_bytes()
    case.api.client.kernels.pages.append(_ResponsePage(case.remote_files))
    assert case.adapter.main(arguments(case, execute=True)) == 0
    retry = json.loads(capsys.readouterr().out)
    assert all(item["state"] == "ALREADY_RETAINED" for item in retry["files"])
    assert len(case.http.urls) == len(ROLES) + 1
    assert snapshot(case.destination) == retained and snapshot(case.run) == before


def test_execute_without_credentials_fails_before_any_local_owner(retrieval, monkeypatch, capsys):
    case = retrieval
    original = Mock(side_effect=AssertionError("Must not resolve or prepare a run"))
    paths = Mock(side_effect=AssertionError("Must not read the profile"))
    monkeypatch.setattr(runs, "_original_run", original)
    monkeypatch.setattr("molgap.common.paths.MachinePaths.from_file", paths)
    assert case.adapter.main(arguments(case) + ["--execute"]) == 2
    assert json.loads(capsys.readouterr().out)["stage"] == "arguments"
    original.assert_not_called()
    paths.assert_not_called()
    case.auth.assert_not_called()
    case.retrieve.assert_not_called()
    assert not case.destination.exists()


def test_credentials_without_execute_are_rejected_without_read(retrieval, capsys):
    case = retrieval
    assert case.adapter.main(arguments(case) + ["--credentials", "unread.json"]) == 2
    assert json.loads(capsys.readouterr().out)["stage"] == "arguments"
    case.auth.assert_not_called()
    assert not case.destination.exists()


def test_abbreviated_execute_flag_cannot_resolve_run_or_authenticate(retrieval, monkeypatch, capsys):
    case = retrieval
    original = Mock(side_effect=AssertionError("Abbreviated flags cannot resolve a run"))
    monkeypatch.setattr(runs, "_original_run", original)
    args = arguments(case, execute=True)
    args[args.index("--execute")] = "--exe"
    assert case.adapter.main(args) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["stage"] == "arguments" and captured.err == ""
    original.assert_not_called()
    case.auth.assert_not_called()
    case.retrieve.assert_not_called()
    assert case.api.client_builds == 0 and case.http.urls == []
    assert not case.destination.exists()


@pytest.mark.parametrize("fault", ["hash", "version", "unknown_version", "package", "account",
    "arm", "roles", "format", "path", "local_alias", "remote_alias", "nonfinite"])
def test_invalid_manifest_is_blocked_before_auth_or_destination(retrieval, capsys, fault):
    case = retrieval
    if fault == "version": case.output["context"]["platform_version"] = "8"
    elif fault == "unknown_version": case.output["context"]["platform_version"] = None
    elif fault == "package": case.output["context"]["package_identity"] = "0" * 64
    elif fault == "account": case.output["context"]["account"] = "other-owner"
    elif fault == "arm": case.output["context"]["arm_id"] = "other-arm"
    elif fault == "roles": del case.output["artifacts"]["resume"]
    elif fault == "format": case.output["format"] = "historical-format"
    elif fault == "path": case.output["artifacts"]["resume"]["path"] = "../escape"
    elif fault == "local_alias":
        case.output["artifacts"]["resume"]["path"] = "output_manifest.json"
    elif fault == "remote_alias":
        case.remote = "runs/version7/gptrans/nested/resume.bin"
        case.output["artifacts"]["resume"]["path"] = "resume.bin"
    write(case.local, case.output)
    if fault == "nonfinite":
        case.local.write_bytes(case.local.read_bytes().replace(b'"artifacts":', b'"invalid":NaN,"artifacts":'))
    args = arguments(case, execute=True)
    if fault == "hash": args[args.index("--manifest-sha256") + 1] = "0" * 64
    assert case.adapter.main(args) == 2
    assert json.loads(capsys.readouterr().out)["stage"] == "manifest"
    case.auth.assert_not_called()
    case.retrieve.assert_not_called()
    assert not case.destination.exists()


@pytest.mark.parametrize("where", ["checkout", "frozen", "artifact_root", "drive_root", "outside", "existing", "file"])
def test_destination_constraints_are_checked_before_auth(retrieval, tmp_path, capsys, where):
    case = retrieval
    artifact_root = case.run.parents[2]
    destinations = {"checkout": case.repo / "retained", "frozen": case.run / "retained",
        "artifact_root": artifact_root, "drive_root": Path(tmp_path.anchor),
        "outside": tmp_path / "outside", "existing": artifact_root / "unrelated",
        "file": artifact_root / "file"}
    case.destination = destinations[where]
    if where == "existing": write(case.destination / "unrelated.json", {})
    if where == "file": case.destination.write_bytes(b"unrelated")
    before = snapshot(artifact_root)
    assert case.adapter.main(arguments(case, execute=True)) == 2
    assert json.loads(capsys.readouterr().out)["stage"] == "destination"
    case.auth.assert_not_called()
    case.retrieve.assert_not_called()
    assert snapshot(artifact_root) == before


def test_symlink_destination_fails_before_auth(retrieval, tmp_path, capsys):
    case = retrieval
    target = tmp_path / "external"
    target.mkdir()
    link = case.run.parents[2] / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit creating symlinks")
    case.destination = link / "retrieved"
    assert case.adapter.main(arguments(case, execute=True)) == 2
    assert json.loads(capsys.readouterr().out)["stage"] == "destination"
    case.auth.assert_not_called()
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("fault", ["journal_unknown", "receipt", "package"])
def test_original_run_tampering_or_uncertainty_cannot_authenticate(retrieval, capsys, fault):
    case = retrieval
    if fault == "journal_unknown":
        path = case.run / "operations/submission_state.json"
        value = json.loads(path.read_bytes())
        value["phase"] = "SUBMIT_UNKNOWN"
        write(path, value)
    elif fault == "receipt":
        path = case.run / "operations/launch_receipt.json"
        value = json.loads(path.read_bytes())
        value["receipt_identity"] = "0" * 64
        write(path, value)
    else:
        (case.package / "source.tar.gz").write_bytes(b"changed-package")
    before = snapshot(case.run)
    assert case.adapter.main(arguments(case, execute=True)) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["stage"] == "original_run"
    if fault == "journal_unknown": assert result["status"] == "RECONCILIATION_REQUIRED"
    case.auth.assert_not_called()
    assert not case.destination.exists() and snapshot(case.run) == before


def test_wrong_authenticated_account_reaches_original_gate_before_download(retrieval, capsys):
    case = retrieval
    case.api.username = "wrong-owner"
    assert case.adapter.main(arguments(case, execute=True)) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["stage"] == "platform_retrieval"
    case.auth.assert_called_once()
    case.retrieve.assert_called_once()
    assert case.api.client_builds == 0 and case.http.urls == []
    assert not case.destination.exists()


def test_retained_artifact_conflict_fails_before_auth(retrieval, capsys):
    case = retrieval
    write(case.destination / "output_manifest.json", case.output)
    relative = case.output["artifacts"]["resume"]["path"]
    target = case.destination / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"conflicting-retained-output")
    before = snapshot(case.destination)
    assert case.adapter.main(arguments(case, execute=True)) == 2
    assert json.loads(capsys.readouterr().out)["stage"] == "destination"
    case.auth.assert_not_called()
    case.retrieve.assert_not_called()
    assert snapshot(case.destination) == before


def test_original_transport_rejects_bytes_from_another_attempt(retrieval, capsys):
    case = retrieval
    # Even a matching latest-session slug cannot replace the independent pin.
    case.payloads[case.remote_files[0].url] = b"another-attempt"
    assert case.adapter.main(arguments(case, execute=True)) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["stage"] == "platform_retrieval"
    assert "SECRET" not in captured.out and captured.err == ""
    case.retrieve.assert_called_once()
    assert not (case.destination / "output_manifest.json").exists()
    assert not any(case.destination.rglob("*.bin"))


def test_platform_diagnostics_and_exception_secrets_never_escape(retrieval, capsys):
    case = retrieval
    def fail(*args):
        import sys
        print("credential-SECRET")
        print("signed-SECRET", file=sys.stderr)
        raise RuntimeError("https://signed.invalid/?key=exception-SECRET")
    case.auth.side_effect = fail
    assert case.adapter.main(arguments(case, execute=True)) == 2
    captured = capsys.readouterr()
    assert "SECRET" not in captured.out and captured.err == ""
    assert json.loads(captured.out)["stage"] == "platform_retrieval"
    case.retrieve.assert_not_called()
    assert not case.destination.exists()


def test_parser_errors_are_finite_json_without_echoing_input(retrieval, capsys):
    assert retrieval.adapter.main(["--unknown=SECRET"]) == 2
    captured = capsys.readouterr()
    assert "SECRET" not in captured.out and captured.err == ""
    assert json.loads(captured.out)["stage"] == "arguments"
