"""Bounded OFFLINE platform roundtrip, never a native K1/GPTrans resume gate.

The existing fake Kaggle API supplies observations; shared journal, package,
release, receipt, retrieval and recovery owners run unchanged. Family validation
uses the explicitly synthetic owner from the lifecycle/resume fixtures. Tiny
CPU tensors are evidence fixtures, not model execution or scientific results.
All scratch files use pytest temporary paths; this module performs no real authentication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from types import SimpleNamespace as NS

import pytest

from molgap import experiment_launch as launch
from molgap import experiment_run as runs
from molgap import experiment_workflow as workflow
from molgap import kaggle_submission as submission
from molgap.common.paths import MachinePaths
from molgap.experiment_family_workflow import RunContext
from molgap.experiment_package import verify_experiment_source_package
from molgap.experiment_resume import (
    RESUME_BUNDLE_FILE, create_resume_bundle, restore_resume_bundle,
    validate_resume_bundle,
)
from molgap.experiment_workflow_resume import RESUME_PLAN_FORMAT
from molgap.kaggle_output_retrieval import retrieve_selected_kaggle_files
from molgap.screen_policy import canonical_fingerprint
from test_experiment_lifecycle_integration import (
    _install_synthetic_owner, _prepare_reference_inputs, _rewrite_case,
    _write_family_output,
)
from test_experiment_run import snapshot, write
from test_experiment_workflow import _git, workflow_case
from test_experiment_workflow_resume import (
    _install_resume_owner, _prior_allocation_ledger, _write_incomplete_output,
)
from test_kaggle_family_retrieval import _Api, _File, _Http, _ResponsePage, _module
from test_kaggle_submission import release as fake_release


MONITORS = ["experiments/example/STATUS.md"]


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def daily_case(tmp_path, workflow_case, monkeypatch):
    # Block accidental fallback to a real HTTP client, including authentication.
    def offline(*args, **kwargs):
        raise AssertionError("This regression has no network authority")

    monkeypatch.setattr("requests.sessions.Session.request", offline)
    monkeypatch.setattr(submission, "connect_kaggle", offline)
    repo = workflow_case["repo"]
    _prepare_reference_inputs(repo)
    module = _install_synthetic_owner(repo, monkeypatch)
    authority = {"user_authorized": True, "allocated_t4_hours_ceiling": 14,
        "arms": [arm["arm_id"] for arm in workflow_case["spec"].to_dict()["arms"]],
        "seeds": [42]}
    write(repo / "authority.json", authority)
    # Existing fixture helpers commit only within this temporary synthetic repo.
    _git(repo, "add", "--", "authority.json")
    _install_resume_owner(module)
    spec, plan, expected, recipes = _rewrite_case(workflow_case, repo, module)
    plan["source_files"].append("authority.json")
    plan["kaggle"].update(account="owner", kernel="owner/new-kernel",
        title="New Kernel", source_dataset="owner/new-source",
        datasets=["owner/new-source", "owner/fixed-data"])

    profile = tmp_path / "machine.json"
    write(profile, {"format": "molgap-machine-paths-v1", "python": sys.executable,
        **{name: str(tmp_path / name) for name in ("data_root", "artifact_root", "scratch_root")}})
    paths = MachinePaths.from_file(profile)
    run_id = spec.to_dict()["logical_run_id"]
    run = paths.run_directory(run_id, "v1")
    prepared = run / "prepared"
    report = workflow.prepare_workflow(spec, repo, plan, prepared)
    assert report["status"] == "PREPARED_FOR_PLATFORM", report
    manifest = verify_experiment_source_package(prepared / "package")
    write(run / "inputs/experiment_spec.json", spec.to_dict())
    write(run / "inputs/workflow_plan.json", plan)
    write(run / "inputs/input_binding.json", {
        "format": "molgap-experiment-input-v1", "input_identity": "f" * 64,
        "spec_identity": spec.identity, "workflow_identity": canonical_fingerprint(plan),
        "source_commit": manifest["source_commit"], "authority_ref": "authority.json",
        "monitor_paths": MONITORS})

    transport_root = tmp_path / "fake-platform"
    transport_root.mkdir()
    # Extract the existing fake transport, without inheriting that fixture's
    # package/release/receipt stubs. All those owners below remain real.
    with monkeypatch.context() as stubs:
        transport = fake_release.__wrapped__(transport_root, stubs,
            staged_source=prepared / "source_dataset")
        fake_push = submission.push_kernel_with_accelerator
    shutil.copyfile(prepared / "kernel/run.py", transport.kernel / "run.py")
    monkeypatch.setattr(submission, "_kernel_source", lambda *_: transport.api.remote)
    monkeypatch.setattr(submission, "_dataset_details", lambda *_: NS(current_version_number=1))

    def checked_fake_push(**kwargs):
        assert kwargs["package_dir"] == prepared / "kernel"
        assert kwargs["credential_path"] == transport.credentials
        assert kwargs["accelerator"] == "NvidiaTeslaT4"
        submission._verify_release_report(kwargs["release_report_path"], prepared / "kernel/run.py")
        return fake_push(**kwargs)
    monkeypatch.setattr(submission, "push_kernel_with_accelerator", checked_fake_push)

    def submit():
        return submission.submit_prepared_workflow(prepared, transport.credentials,
            run / "operations", authority_ref="authority.json", monitor_paths=MONITORS,
            execute=True, api=transport.api, http_session=transport.http)

    return NS(repo=repo, spec=spec, expected=expected, recipes=recipes,
        profile=profile, paths=paths, run=run, prepared=prepared,
        package=prepared / "package", manifest=manifest, transport=transport,
        submit=submit, push=checked_fake_push, original=(profile, repo, run_id, "v1"),
        receipt=run / "operations/launch_receipt.json")


def _bound(case):
    result = case.submit()
    assert result["status"] == "SUBMISSION_BOUND", result
    receipt = launch.read_launch_receipt(case.receipt, case.spec, case.package,
        expected_package_identity=case.manifest["package_identity"])
    assert receipt["submission_state"] == "ACCEPTED"
    assert receipt["canonical_platform_reference"]["value"] == "owner/new-kernel"
    assert receipt["platform_version"]["value"] == "1"
    physical, = receipt["physical_runs"]["value"]
    assert physical == {"arm_ids": [a["arm_id"] for a in case.spec.to_dict()["arms"]],
        "canonical_reference": "owner/new-kernel", "platform_version": {
            "value": "1", "missing_reason": None}, "run_identity": "kernel77/version1"}
    assert receipt["binding"]["source_commit"] == case.manifest["source_commit"]
    assert receipt["binding"]["source_archive_sha256"] == case.manifest["archive_sha256"]
    assert receipt["binding"]["package_identity"] == case.manifest["package_identity"]
    proof = json.loads((case.run / "operations/source_verification.json").read_bytes())
    assert (proof["kernel_id"], proof["version"], proof["entry_bytes_equal"]) == (77, 1, True)
    assert proof["entry_sha256"] == _sha((case.prepared / "kernel/run.py").read_bytes())
    assert result["native_qualification"] == "NOT_EVALUATED"
    return receipt


def _observed_context(case, arm_id):
    return RunContext.from_launch(case.spec, case.receipt, case.package,
        expected_package_identity=case.manifest["package_identity"], arm_id=arm_id)


def _remote_files(context, root, files, remote_manifest):
    """Feed existing fake listing/HTTP classes; files is a manifest allowlist."""
    payloads, listing = {}, []
    for remote, (relative, _) in files.items():
        url = "https://signed.invalid/" + remote + "?token=SYNTHETIC_SECRET"
        payloads[url] = (root / relative).read_bytes()
        listing.append(_File(remote, url))
    extra = "https://signed.invalid/unselected?token=SYNTHETIC_SECRET"
    listing.append(_File("unselected/large-debug.bin", extra))
    payloads[extra] = b"must never be retrieved"
    api = _Api(context.account, [_ResponsePage(listing)])
    http = _Http(payloads)
    return NS(api=api, http=http, files=files, extra=extra,
        remote_manifest=remote_manifest)


def _resume_remote(case, arm_id, tmp_path):
    context = _observed_context(case, arm_id)
    producer = RunContext.for_training(case.spec, case.package,
        expected_package_identity=case.manifest["package_identity"], arm_id=arm_id,
        account=context.account, run_reference=context.run_reference)
    source = _write_incomplete_output(case.repo, producer, arm_id)
    contexts = {arm["arm_id"]: _observed_context(case, arm["arm_id"])
        for arm in case.spec.to_dict()["arms"]}
    write(source / "allocation_ledger.json", _prior_allocation_ledger(case.spec, contexts))
    trajectory = case.prepared / "source_dataset/prospective" / arm_id / "trajectory.json"
    bundle = tmp_path / "durable-remote" / arm_id
    manifest = create_resume_bundle(source, producer, trajectory, bundle)
    checked = validate_resume_bundle(bundle, case.spec, arm_id=arm_id,
        context=producer, trajectory=trajectory)
    assert checked["owner"]["owner"] == "synthetic"
    remote_manifest = "runs/kernel77/version1/" + arm_id + "/" + RESUME_BUNDLE_FILE
    prefix = PurePosixPath(remote_manifest).parent
    files = {remote_manifest: (RESUME_BUNDLE_FILE, _sha((bundle / RESUME_BUNDLE_FILE).read_bytes()))}
    pointers = [p for p in manifest["artifacts"].values() if p is not None]
    for group in ("runtime", "provenance", "context", "cost_segments"):
        pointers.extend(manifest[group])
    for pointer in pointers:
        files[(prefix / pointer["path"]).as_posix()] = (pointer["path"], pointer["sha256"])
    remote = _remote_files(context, bundle, files, remote_manifest)
    remote.context, remote.producer, remote.trajectory = context, producer, trajectory
    remote.manifest, remote.bundle = manifest, bundle
    return remote


@pytest.mark.parametrize("lost_response", [False, True], ids=["observed", "lost-response"])
def test_offline_daily_roundtrip_preserves_original_run(daily_case, tmp_path, monkeypatch, lost_response):
    case = daily_case
    immutable = snapshot(case.prepared)
    if lost_response:
        def lost(**kwargs):
            case.push(**kwargs)
            return {"status": "submission_unknown", "reconciliation_required": True}
        monkeypatch.setattr(submission, "push_kernel_with_accelerator", lost)
        assert case.submit()["status"] == "RECONCILIATION_REQUIRED"
        assert json.loads((case.run / "operations/submission_state.json").read_bytes())["phase"] == "SUBMIT_UNKNOWN"
        assert not case.receipt.exists()
    _bound(case)
    retained_receipt = case.receipt.read_bytes()
    _bound(case)
    assert case.receipt.read_bytes() == retained_receipt
    assert case.transport.api.created == case.transport.api.pushed == 1
    upload = json.loads((case.run / "operations/source_upload_verification.json").read_bytes())
    assert upload["status"] == "REMOTE_SOURCE_BYTES_VERIFIED" and upload["version"] == 1
    state = json.loads((case.run / "operations/submission_state.json").read_bytes())
    retirement = state["source_copy_retirement"]
    assert retirement["status"] == "REMOVED"
    roundtrip = {item["path"]: {key: item[key] for key in ("sha256", "bytes")}
        for item in upload["files"]}
    assert retirement["roundtrip"] == roundtrip
    retired_uploads = {path: record for path, record in roundtrip.items()
        if path == "source_payload.bin" or path.startswith("initial_states/")}
    assert retirement["uploaded"] == retired_uploads
    retired_prepared_paths = {"source_dataset/" + path for path in retired_uploads}
    prepared_after = snapshot(case.prepared)
    assert set(immutable) - set(prepared_after) == retired_prepared_paths
    assert prepared_after == {path: payload for path, payload in immutable.items()
        if path not in retired_prepared_paths}
    assert not (case.run / "operations/source_roundtrip").exists()
    frozen = snapshot(case.run)
    _module(monkeypatch)  # Existing offline SDK request type only.

    bundles = {}
    missing_owner_checked = False
    for arm in case.spec.to_dict()["arms"]:
        arm_id = arm["arm_id"]
        remote = _resume_remote(case, arm_id, tmp_path)
        retained = tmp_path / "retrieved" / arm_id
        def retrieve():
            return retrieve_selected_kaggle_files(remote.api, context=remote.context,
                remote_manifest_path=remote.remote_manifest, files=remote.files,
                destination=retained, http_session=remote.http)
        result = retrieve()
        assert result["receipt_version"] == "1"
        assert result["platform_version_verified_by_output_api"] is False
        assert result["unselected_outputs_downloaded"] is False
        assert remote.extra not in remote.http.urls
        assert len(remote.http.urls) == len(remote.files)
        assert snapshot(retained) == snapshot(remote.bundle)
        remote.api.client.kernels.pages.append(remote.api.client.kernels.pages[0])
        assert all(item["state"] == "ALREADY_RETAINED" for item in retrieve()["files"])
        assert len(remote.http.urls) == len(remote.files)
        restored = tmp_path / "restored-prefix" / arm_id
        restored_report = restore_resume_bundle(retained, restored, case.spec,
            arm_id=arm_id, context=remote.producer, trajectory=remote.trajectory)
        assert restored_report["owner"]["owner"] == "synthetic"
        if not missing_owner_checked:
            frozen_plan = json.loads((case.run / "inputs/workflow_plan.json").read_bytes())
            initial_pointer = next(entry["initial_state"] for entry in frozen_plan["arms"]
                if entry["arm_id"] == arm_id)
            initial_owner = Path(initial_pointer)
            if not initial_owner.is_absolute():
                initial_owner = case.repo / initial_owner
            initial_owner = initial_owner.absolute()
            owner_bytes = initial_owner.read_bytes()
            moved_owner = initial_owner.with_name(initial_owner.name + ".missing-owner-control")
            assert not moved_owner.exists()
            missing_output = tmp_path / "missing-owner-recovery" / arm_id
            assert not missing_output.exists()
            before_run = snapshot(case.run)
            before_prepared = snapshot(case.prepared)
            assert not list(case.run.glob(".resume-inputs-*"))
            initial_owner.rename(moved_owner)
            try:
                with pytest.raises(ValueError, match="Original frozen recovery input is missing or changed"):
                    runs.build_run_resume(*case.original, arm_id=arm_id,
                        source_output=restored, output=missing_output)
                assert not missing_output.exists()
                assert snapshot(case.run) == before_run
                assert snapshot(case.prepared) == before_prepared
                assert not list(case.run.glob(".resume-inputs-*"))
            finally:
                moved_owner.rename(initial_owner)
            assert initial_owner.read_bytes() == owner_bytes
            missing_owner_checked = True
        exported = runs.build_run_resume(*case.original, arm_id=arm_id,
            source_output=restored, output=tmp_path / "validated-bundles" / arm_id)
        assert exported["status"] == "RESUME_VERIFIED"
        assert exported["cursor"] == remote.manifest["cursor"]
        exported_manifest = json.loads(Path(exported["manifest"]).read_bytes())
        assert exported_manifest["run"] == remote.manifest["run"]
        assert exported_manifest["source"] == remote.manifest["source"]
        action, = json.loads(remote.trajectory.read_bytes())["actions"]
        assert exported_manifest["run"] == {"logical_run_id": case.original[2],
            "action_id": action["action_id"], "run_id": action["run_ids"][0],
            "attempt_id": action["attempt_ids"][0]}
        bundles[arm_id] = exported["manifest"]

    assert missing_owner_checked
    assert not list(case.run.glob(".resume-inputs-*"))

    def no_new_scientific_plan(*args, **kwargs):
        raise AssertionError("Recovery must reuse the original prospective action")
    monkeypatch.setattr(workflow, "plan_prospective", no_new_scientific_plan)
    attempt2 = case.paths.run_directory(case.original[2], "v2") / "prepared"
    recovery = runs.prepare_run_resume(*case.original, output=attempt2, plan={
        "format": RESUME_PLAN_FORMAT, "spec_identity": case.spec.identity, "arms": bundles})
    assert recovery["status"] == "RESUME_INPUTS_PREPARED", recovery
    assert "owning continuation adapter" in recovery["next_action"]
    # This payload retains the original producer identity. A physical successor
    # is a separate owning-platform operation, not another fresh-release POST.
    for execute in (False, True):
        with pytest.raises(ValueError, match="owning continuation adapter"):
            submission.submit_prepared_workflow(attempt2, tmp_path / "absent-credentials.json",
                attempt2.parent / "unsupported-operations", authority_ref="authority.json",
                monitor_paths=MONITORS, execute=execute,
                api=case.transport.api, http_session=case.transport.http)
    assert not (attempt2.parent / "unsupported-operations").exists()
    assert recovery["prospective_published"] is recovery["submitted"] is False
    assert recovery["spec_identity"] == case.spec.identity
    assert recovery["package_identity"] == case.manifest["package_identity"]
    for name in ("release_report.json", "workflow_report.json"):
        persisted_report = (attempt2 / name).read_bytes()
        assert b".resume-inputs-" not in persisted_report, name
    recovered_launch = json.loads((attempt2 / "source_dataset/experiment_launch.json").read_bytes())
    assert set(recovered_launch["resume"]) == set(bundles)
    original_launch = json.loads((case.prepared / "source_dataset/experiment_launch.json").read_bytes())
    assert {k: v for k, v in recovered_launch.items() if k != "resume"} == original_launch
    for arm_id in bundles:
        relative = Path("source_dataset/prospective") / arm_id / "trajectory.json"
        assert (attempt2 / relative).read_bytes() == (case.prepared / relative).read_bytes()
    # v2 is local recovery staging, not an observed second physical job or a
    # newly published run. Original prospective attempt IDs remain provenance.
    assert not (attempt2.parent / "operations").exists()
    assert not (attempt2.parent / "inputs").exists()
    assert snapshot(case.run) == frozen
    assert case.transport.api.created == case.transport.api.pushed == 1


def test_unobserved_lost_response_stays_reconcile_only(daily_case, monkeypatch, tmp_path):
    case = daily_case
    def lost(**kwargs):
        kwargs["_before_post"]()
        case.transport.api.pushed += 1
        raise TimeoutError("synthetic lost response")
    monkeypatch.setattr(submission, "push_kernel_with_accelerator", lost)
    for _ in range(3):
        assert case.submit()["status"] == "RECONCILIATION_REQUIRED"
    assert case.transport.api.created == case.transport.api.pushed == 1
    assert not case.receipt.exists()
    with pytest.raises(runs.RunReconciliationRequired, match="original submission journal"):
        runs.prepare_run_resume(*case.original, plan={}, output=tmp_path / "forbidden-recovery")
    assert not (tmp_path / "forbidden-recovery").exists()


@pytest.mark.parametrize("drift", ["version", "source"])
def test_observed_identity_drift_cannot_rebind_or_repost(daily_case, drift):
    case = daily_case
    _bound(case)
    receipt = case.receipt.read_bytes()
    if drift == "version":
        case.transport.api.remote.metadata.current_version_number = 2
    else:
        case.transport.api.remote.blob.source += "# changed remote source\n"
    with pytest.raises(ValueError, match="binding differs"):
        case.submit()
    assert case.receipt.read_bytes() == receipt
    assert case.transport.api.created == case.transport.api.pushed == 1


@pytest.mark.parametrize("corruption", ["checkpoint", "manifest"])
def test_pinned_resume_bytes_from_another_attempt_are_rejected(daily_case, tmp_path, monkeypatch, corruption):
    case = daily_case
    _bound(case)
    _module(monkeypatch)
    arm_id = case.spec.to_dict()["arms"][0]["arm_id"]
    remote = _resume_remote(case, arm_id, tmp_path)
    relative = (remote.manifest["artifacts"]["checkpoint"]["path"]
        if corruption == "checkpoint" else RESUME_BUNDLE_FILE)
    url = next(url for url in remote.http.payloads if relative in url)
    remote.http.payloads[url] = b"different physical attempt"
    destination = tmp_path / "rejected-retrieval"
    with pytest.raises(ValueError, match="hash disagrees with pinned attempt"):
        retrieve_selected_kaggle_files(remote.api, context=remote.context,
            remote_manifest_path=remote.remote_manifest, files=remote.files,
            destination=destination, http_session=remote.http)
    assert not (destination / relative).exists()
    assert not list(destination.rglob("*.tmp"))
    assert case.transport.api.created == case.transport.api.pushed == 1


def test_completed_family_retrieval_is_separate_synthetic_evidence(daily_case, tmp_path, monkeypatch):
    case = daily_case
    _bound(case)
    adapter = _module(monkeypatch)
    arm_id = "gptrans_candidate"
    source = _write_family_output(case.repo, case.spec, case.package,
        case.manifest["package_identity"], case.receipt, case.expected,
        arm_id, case.recipes[arm_id])
    context = _observed_context(case, arm_id)
    local_manifest = source / "output_manifest.json"
    manifest = json.loads(local_manifest.read_bytes())
    remote_manifest = "runs/kernel77/version1/completed/output_manifest.json"
    _, files = adapter._family_files(context, remote_manifest, local_manifest,
        _sha(local_manifest.read_bytes()))
    remote = _remote_files(context, source, files, remote_manifest)
    destination = tmp_path / "completed-family"
    result = adapter.retrieve_family_outputs(remote.api, context=context,
        remote_manifest_path=remote_manifest, local_manifest=local_manifest,
        manifest_sha256=_sha(local_manifest.read_bytes()), destination=destination,
        http_session=remote.http)
    assert result["producer_version"] == result["receipt_version"] == "1"
    assert result["platform_version_verified_by_output_api"] is True
    assert result["unselected_outputs_downloaded"] is False
    assert remote.extra not in remote.http.urls
    assert snapshot(destination) == {relative: (source / relative).read_bytes()
        for relative, _ in files.values()}
    assert manifest["context"] == context.to_dict()
    inspected = runs.inspect_run_output(*case.original, arm_id=arm_id,
        artifact_root=destination, expectations=case.expected)
    assert inspected["status"] == "MECHANICALLY_VERIFIED", inspected
    with pytest.raises(ValueError, match="Completed family output"):
        create_resume_bundle(destination, context, {}, tmp_path / "invalid-resume")
