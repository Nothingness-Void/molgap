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
from test_experiment_launch import build, unknown
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


@pytest.fixture
def unknown_producer(retrieval):
    case = retrieval
    # Exercise the actual training producer, not a hand-built null context.
    case.producer = RunContext.for_training(case.spec, case.package,
        expected_package_identity=case.manifest["package_identity"], arm_id="gptrans_t",
        account=case.context.account, run_reference=case.context.run_reference)
    assert case.producer.platform_version is None
    assert case.producer.to_dict() == {**case.context.to_dict(), "platform_version": None}
    case.output["context"] = case.producer.to_dict()
    write(case.local, case.output)
    case.manifest_url = next(item.url for item in case.remote_files if item.file_name == case.remote)
    case.payloads[case.manifest_url] = case.local.read_bytes()
    return case


@pytest.mark.parametrize("execute", [False, True])
@pytest.mark.parametrize("continuation", [False, True])
def test_normal_unknown_producer_binds_exact_receipt_without_backfill(unknown_producer, capsys, execute, continuation):
    case = unknown_producer
    original = snapshot(case.run)
    pinned = case.local.read_bytes()
    args = arguments(case, execute=execute)
    if continuation:
        response = deepcopy(case.receipt["response"])
        response["platform_version"]["value"] = "8"
        for physical in response["physical_runs"]["value"]:
            physical["platform_version"]["value"] = "8"
        receipt = build((case.spec, case.package, case.manifest["package_identity"]), response)
        successor = case.profile.parent / "continuation-receipt.json"
        write(successor, receipt)
        case.context = RunContext.from_launch(case.spec, successor, case.package,
            expected_package_identity=case.manifest["package_identity"], arm_id="gptrans_t")
        args += ["--continuation-receipt", str(successor)]
    assert case.context.platform_version == ("8" if continuation else "7")
    assert case.adapter.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["producer_version"] is None
    assert result["manifest_sha256"] == hashlib.sha256(pinned).hexdigest()
    assert result["scientific_acceptance"] is False
    if execute:
        assert result["status"] == "RETRIEVED" and result["receipt_version"] == case.context.platform_version
        assert result["platform_version_verified_by_output_api"] is True
        assert result["unselected_outputs_downloaded"] is False
        assert case.retrieve.call_args.kwargs["context"] == case.context
        retained = (case.destination / "output_manifest.json").read_bytes()
        assert retained == pinned
        assert json.loads(retained)["context"]["platform_version"] is None
        assert len(case.http.urls) == len(ROLES) + 1
        assert all(request.version_number == int(case.context.platform_version)
                   for request in case.api.client.kernels.exact_requests)
    else:
        assert result["status"] == "READY_FOR_RETRIEVAL"
        case.auth.assert_not_called()
        case.retrieve.assert_not_called()
        assert case.http.urls == [] and not case.destination.exists()
    assert case.local.read_bytes() == pinned
    retained_run = snapshot(case.run)
    if execute:
        mapping = "operations/retrieved_outputs/" + ("8/" if continuation else "") + "gptrans_t.json"
        record = json.loads(retained_run.pop(mapping))
        assert record["context"] == case.context.to_dict()
        assert record["manifest_sha256"] == result["manifest_sha256"]
        assert record["output_dir"] == str(case.destination)
    assert retained_run == original


def test_all_arms_retrieves_every_pinned_manifest_from_the_run_directory(retrieval, monkeypatch, capsys):
    from types import SimpleNamespace
    from molgap.experiment_allocation import AllocationLedger
    from molgap.experiment_retention import seal_execution_retention
    import requests
    import sys

    # This flow uses the installed SDK's request types, not the permissive
    # placeholder modules used by legacy transport unit tests.
    for name, module in list(sys.modules.items()):
        if (name == "kagglesdk" or name.startswith("kagglesdk.")) and not getattr(module, "__file__", None):
            monkeypatch.delitem(sys.modules, name)
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiGetKernelSessionStatusResponse
    from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus

    case = retrieval
    response = deepcopy(case.receipt["response"])
    response["platform_version"]["value"] = "8"
    for physical in response["physical_runs"]["value"]:
        physical["platform_version"]["value"] = "8"
    successor = case.profile.parent / "continuation-receipt.json"
    write(successor, build((case.spec, case.package, case.manifest["package_identity"]), response))
    resume_prepared = case.profile.parent / "resume-prepared"
    remote_root = case.profile.parent / "remote"
    resume_bindings, manifests, arm_roots = {}, {}, {}
    for arm in case.spec.to_dict()["arms"]:
        arm_id = arm["arm_id"]
        context = RunContext.from_launch(case.spec, successor, case.package,
            expected_package_identity=case.manifest["package_identity"], arm_id=arm_id)
        output = deepcopy(case.output)
        output.update(context=context.to_dict(), runtime={"runtime_certificate_id": "a" * 64})
        output["adapter"] = "gptrans-v1" if arm_id == "gptrans_t" else "k1-v1"
        arm_root = remote_root / arm_id
        arm_roots[arm_id] = arm_root
        write(arm_root / "output_manifest.json", output)
        manifests[arm_id] = (arm_root / "output_manifest.json").read_bytes()
        for item in output["artifacts"].values():
            role = Path(item["path"]).stem
            path = arm_root / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic-" + role).encode())
        bundle = case.profile.parent / arm_id / "resume_bundle.json"
        write(bundle, {"artifacts": {"checkpoint": {"sha256": "b" * 64}}})
        resume_bindings[arm_id] = {"resume_bundle": bundle}
    assert len(manifests) == 2
    clock = [0.0]
    accelerator = case.spec.to_dict()["platform"]["accelerator"]
    ledger = AllocationLedger(spec_identity=case.spec.identity, hardware=[accelerator] * 2,
        assignments={arm: device for device, arm in enumerate(arm_roots)},
        started=0.0, clock=lambda: clock[0])
    clock[0] = 3.0
    ledger.write(remote_root, "complete", arm_roots=list(arm_roots.values()))
    write(remote_root / "execution_report.json", {"status": "complete"})
    seal_execution_retention(remote_root, case.spec, package_identity=case.context.package_identity,
        source_commit=case.context.source_commit, source_archive_sha256=case.context.source_archive_sha256,
        arm_roots=arm_roots)
    files, payloads = [], {}
    for path in remote_root.rglob("*"):
        if path.is_file():
            name = "runs/version8/" + path.relative_to(remote_root).as_posix()
            url = "https://signed.invalid/" + name + "?token=SECRET"
            files.append(_File(name, url))
            payloads[url] = path.read_bytes()
    case.api = _Api(case.context.account, [_ResponsePage(files)])
    status_response = ApiGetKernelSessionStatusResponse()
    status_response.status = KernelWorkerStatus.COMPLETE
    case.api.kernels_status = Mock(return_value=status_response)
    remote_source = SimpleNamespace(metadata=SimpleNamespace(current_version_number=8))
    def get_kernel(request):
        assert isinstance(request, ApiGetKernelRequest)
        assert request.user_name + "/" + request.kernel_slug == case.context.run_reference
        return remote_source
    case.api.client.kernels.get_kernel = get_kernel
    case.http = _Http(payloads)
    case.auth.return_value = case.api
    monkeypatch.setattr(requests, "Session", lambda: case.http)
    # Only the native/resume proof owners are substituted: this tests forwarding,
    # not proof qualification. Manifest, cost-ledger and transport checks stay real.
    resolve_resume = Mock(return_value=resume_bindings)
    monkeypatch.setattr("molgap.experiment_workflow_resume.resolve_terminal_resume", resolve_resume)
    native = Mock(return_value={"status": "DEPENDENCIES_VERIFIED", "network_accessed": False})
    monkeypatch.setattr(case.adapter, "retrieve_native_dependencies", native)
    base = ["--all-arms", "--run-id", "synthetic-run", "--attempt-id", "v1",
            "--profile", str(case.profile), "--repo-root", str(case.repo), "--strict",
            "--continuation-receipt", str(successor), "--resume-prepared", str(resume_prepared),
            "--remote-root", "runs/version8"]
    original = snapshot(case.run)
    credentials = str(case.profile.parent / "unread-credentials.json")
    assert case.adapter.main([*base, "--status", "--credentials", credentials]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["status"] == "OBSERVED" and observed["scheduler"] == "complete"
    assert observed["version"] == "8" and observed["downloaded_files"] == 0
    assert observed["scheduler_scope"] == "latest" and observed["observed_latest_version"] == 8
    case.api.kernels_status.assert_called_once_with(case.context.run_reference)
    assert case.http.urls == [] and snapshot(case.run) == original
    case.auth.reset_mock()
    assert case.adapter.main(base) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "READY_FOR_RETRIEVAL"
    assert set(plan["missing_manifests"]) == {
        *(f"runs/version8/{arm}/output_manifest.json" for arm in manifests),
        "runs/version8/execution_retention.json"}
    case.auth.assert_not_called()
    native.assert_not_called()
    resolve_resume.assert_not_called()
    assert case.http.urls == [] and snapshot(case.run) == original
    case.api.kernels_status.return_value.status = KernelWorkerStatus.RUNNING
    assert case.adapter.main([*base, "--execute", "--credentials", credentials]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "WAIT_REMOTE_TERMINAL"
    assert case.http.urls == [] and snapshot(case.run) == original
    # A newer running version cannot block retrieval of the pinned completed v8.
    remote_source.metadata.current_version_number = 9
    assert case.adapter.main([*base, "--execute", "--credentials", credentials]) == 0
    done = json.loads(capsys.readouterr().out)
    assert done["status"] == "RETRIEVED" and set(done["arms"]) == set(manifests)
    assert native.call_count == 2
    assert resolve_resume.call_count == 2
    assert all(call.args[1:] == (resume_prepared, None) for call in resolve_resume.call_args_list)
    assert {call.kwargs["context"].arm_id for call in native.call_args_list} == set(manifests)
    for call in native.call_args_list:
        assert call.kwargs["context"].platform_version == "8"
        assert call.kwargs["resume_checkpoint_sha256"] == "b" * 64
    pinned_root = case.run / "operations/remote_manifest/version8/runs/version8"
    destination = case.run.parents[2] / "retrieved/synthetic-run/v1/version8"
    for arm_id, manifest_bytes in manifests.items():
        assert done["arms"][arm_id]["status"] == "RETRIEVED"
        assert (pinned_root / arm_id / "output_manifest.json").read_bytes() == manifest_bytes
        assert (destination / arm_id / "output_manifest.json").read_bytes() == manifest_bytes
        mapping = case.run / "operations/retrieved_outputs/8" / (arm_id + ".json")
        assert json.loads(mapping.read_bytes())["context"]["platform_version"] == "8"
    execution_root = destination / "_execution"
    assert done["execution_root"] == str(execution_root)
    assert done["execution_retention"]["execution_retention"]["status"] == "EXECUTION_RETENTION_VERIFIED"
    retained_ledger = json.loads((execution_root / "allocation_ledger.json").read_bytes())
    assert retained_ledger["wall_seconds"] == 3.0
    assert retained_ledger["allocated_device_seconds"] == 6.0
    for arm_id in manifests:
        assert (execution_root / arm_id / "allocation_ledger.json").read_bytes() == (
            remote_root / arm_id / "allocation_ledger.json").read_bytes()
    requests_made = case.api.client.kernels.exact_requests
    assert {request.file_path for request in requests_made} == {file.file_name for file in files}
    assert all(request.version_number == 8 for request in requests_made)
    assert not case.api.client.kernels.requests
    assert set(case.http.urls) == set(payloads)
    for arm_id in manifests:
        url = f"https://signed.invalid/runs/version8/{arm_id}/output_manifest.json?token=SECRET"
        assert case.http.urls.count(url) == 1
    downloaded = list(case.http.urls)
    assert case.adapter.main([*base, "--execute", "--credentials", credentials]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "RETRIEVED"
    assert case.http.urls == downloaded


@pytest.mark.parametrize("missing", ["physical", "all"])
def test_unknown_producer_cannot_replace_missing_receipt_version(unknown_producer, capsys, missing):
    case = unknown_producer
    response = deepcopy(case.receipt["response"])
    for physical in response["physical_runs"]["value"]:
        physical["platform_version"] = unknown()
    if missing == "all":
        response["platform_version"] = unknown()
    # Rebuild a valid receipt so rejection is about missing observation, not tampering.
    receipt = build((case.spec, case.package, case.manifest["package_identity"]), response)
    write(case.run / "operations/launch_receipt.json", receipt)
    original = snapshot(case.run)
    assert case.adapter.main(arguments(case, execute=True)) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "BLOCKED" and result["stage"] == "original_run"
    case.auth.assert_not_called()
    case.retrieve.assert_not_called()
    assert case.http.urls == [] and not case.destination.exists()
    assert snapshot(case.run) == original


@pytest.mark.parametrize("changed", ["local", "remote"])
def test_unknown_producer_still_requires_unchanged_pinned_manifest(unknown_producer, capsys, changed):
    case = unknown_producer
    args = arguments(case, execute=True)
    original = snapshot(case.run)
    pinned = case.local.read_bytes()
    altered = deepcopy(case.output)
    altered["context"]["platform_version"] = "8"
    if changed == "local":
        write(case.local, altered)
    else:
        case.payloads[case.manifest_url] = json.dumps(altered).encode()
    assert case.adapter.main(args) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "BLOCKED"
    if changed == "local":
        assert result["stage"] == "manifest"
        case.auth.assert_not_called()
        case.retrieve.assert_not_called()
        assert case.http.urls == [] and not case.destination.exists()
    else:
        assert result["stage"] == "platform_retrieval"
        case.retrieve.assert_called_once()
        assert case.local.read_bytes() == pinned
        assert not (case.destination / "output_manifest.json").exists()
    assert snapshot(case.run) == original


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
    from molgap.experiment_package import _PackageVerification
    original.assert_called_once()
    assert original.call_args.args == (case.profile, case.repo, "synthetic-run", "v1")
    assert set(original.call_args.kwargs) == {"_verification"}
    assert isinstance(original.call_args.kwargs["_verification"], _PackageVerification)
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


def test_execute_delegates_original_auth_and_retriever_and_is_idempotent(
    retrieval, monkeypatch, capsys
):
    case = retrieval
    before = snapshot(case.run)
    _, _, kernel_slug = case.context.run_reference.partition("/")
    from molgap import experiment_package as packages
    verified = Mock(wraps=packages._verify_package_snapshot)
    monkeypatch.setattr(packages, "_verify_package_snapshot", verified)
    failed_role = sorted(ROLES)[-1]
    failed_relative = case.output["artifacts"][failed_role]["path"]
    failed_remote = "runs/version7/gptrans/" + failed_relative
    failed_url = next(item.url for item in case.remote_files
                      if item.file_name == failed_remote)
    case.http.fail_once[failed_url] = RuntimeError("one interrupted artifact stream")

    # The first attempt must publish the pinned manifest before payloads, so a
    # partially retained destination remains safely retryable by this CLI.
    assert case.adapter.main(arguments(case, execute=True)) == 2
    assert verified.call_count == 1
    interrupted = json.loads(capsys.readouterr().out)
    assert interrupted["status"] == "BLOCKED"
    assert interrupted["stage"] == "platform_retrieval"
    assert (case.destination / "output_manifest.json").read_bytes() == case.local.read_bytes()
    partial = snapshot(case.destination)
    assert failed_relative not in partial
    assert len(partial) > 1
    assert case.api.client.kernels.requests == []
    assert case.api.client.kernels.exact_requests
    assert all(request.version_number == 7
               for request in case.api.client.kernels.exact_requests)
    assert all(request.owner_slug == case.context.account
               and request.kernel_slug == kernel_slug
               for request in case.api.client.kernels.exact_requests)

    retained_before_retry = len(case.http.urls)
    assert case.adapter.main(arguments(case, execute=True)) == 0
    assert verified.call_count == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "RETRIEVED" and result["executed"] is True
    assert result["producer_version"] == result["receipt_version"] == "7"
    assert result["scientific_acceptance"] is False
    assert result["platform_version_verified_by_output_api"] is True
    assert result["unselected_outputs_downloaded"] is False
    credentials = case.profile.parent / "unread-credentials.json"
    assert case.auth.call_count == 2
    assert [call.args for call in case.auth.call_args_list] == [
        (credentials,), (credentials,)
    ]
    assert case.retrieve.call_count == 2
    assert case.retrieve.call_args.kwargs["context"] == case.context
    assert len(case.http.urls) == retained_before_retry + 1
    assert snapshot(case.destination)["output_manifest.json"] == case.local.read_bytes()
    assert failed_relative in snapshot(case.destination)
    assert all(case.http.urls.count(item.url) == (2 if item.url == failed_url else 1)
               for item in case.remote_files if item.file_name != case.remote)
    exact_requests = case.api.client.kernels.exact_requests
    assert case.api.client.kernels.requests == []
    assert len(exact_requests) == len(ROLES) + 2
    assert all(request.version_number == 7 for request in exact_requests)
    assert all(request.owner_slug == case.context.account
               and request.kernel_slug == kernel_slug
               for request in exact_requests)
    requested_paths = [request.file_path for request in exact_requests]
    assert requested_paths.count(failed_remote) == 2
    assert case.remote in requested_paths
    assert set(requested_paths) <= {item.file_name for item in case.remote_files}
    retained = snapshot(case.destination)
    assert retained["output_manifest.json"] == case.local.read_bytes()
    from molgap import kaggle_output_retrieval, experiment_run_acceptance
    from molgap.common import transport
    reads = []
    real_digest = kaggle_output_retrieval.file_digest
    def counted_digest(path):
        reads.append(Path(path))
        return real_digest(path)
    for owner in (case.adapter, kaggle_output_retrieval, experiment_run_acceptance, transport):
        monkeypatch.setattr(owner, "file_digest", counted_digest)
    assert case.adapter.main(arguments(case, execute=True)) == 0
    assert verified.call_count == 3
    retry = json.loads(capsys.readouterr().out)
    assert all(item["state"] == "ALREADY_RETAINED" for item in retry["files"])
    assert len(case.http.urls) == retained_before_retry + 1
    assert snapshot(case.destination) == retained
    assert all(reads.count(case.destination / item["path"]) == 1
               for item in case.output["artifacts"].values())
    retained_run = snapshot(case.run)
    mapping = json.loads(retained_run.pop("operations/retrieved_outputs/gptrans_t.json"))
    assert mapping["context"] == case.context.to_dict()
    assert mapping["output_dir"] == str(case.destination)
    assert retained_run == before
    # A new invocation must still reject changed retained bytes before auth.
    damaged = case.destination / next(iter(case.output["artifacts"].values()))["path"]
    damaged.write_bytes(b"changed after the completed call")
    auth_calls = case.auth.call_count
    assert case.adapter.main(arguments(case, execute=True)) == 2
    assert json.loads(capsys.readouterr().out)["stage"] == "destination"
    assert case.auth.call_count == auth_calls and len(case.http.urls) == retained_before_retry + 1


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


@pytest.mark.parametrize("fault", ["hash", "version", "package", "account",
    "arm", "roles", "format", "path", "local_alias", "remote_alias", "nonfinite"])
def test_invalid_manifest_is_blocked_before_auth_or_destination(retrieval, capsys, fault):
    case = retrieval
    if fault == "version": case.output["context"]["platform_version"] = "8"
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
    # The transport announces its finite retrieval plan on stderr; no signed URL may appear.
    assert "SECRET" not in captured.out + captured.err
    assert json.loads(captured.err.splitlines()[0])["retrieval_plan"]["files"]
    case.retrieve.assert_called_once()
    assert (case.destination / "output_manifest.json").read_bytes() == case.local.read_bytes()
    assert not any(case.destination.rglob("*.bin"))
    assert not (case.run / "operations/retrieved_outputs").exists()


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
