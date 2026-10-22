"""Platform state machine with synthetic files and fake APIs; no network/job."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from molgap import kaggle_submission as owner
from molgap.experiment_launch import canonical_json
from molgap.experiment_spec import ExperimentSpec
from molgap.training_reproducibility import atomic_json
from test_scientific_input import scientific


class Missing(Exception):
    status = 404


class Response:
    def __init__(self, payload=None, content=b""):
        self.payload, self.content = payload, content
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload

    def iter_content(self, _size):
        yield self.content

    def iter_lines(self, **kwargs):
        yield 'data: {"data":"synthetic startup"}'
        yield "data: END_OF_LOG"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _dispatch_before_post(kwargs):
    callback = kwargs.get("_before_post")
    if not callable(callback):
        raise AssertionError("Fake transport must mark the durable POST boundary")
    callback()


@pytest.fixture
def release(tmp_path, monkeypatch, staged_source=None):
    prepared, output = tmp_path / "prepared", tmp_path / "operations"
    kernel, staged = prepared / "kernel", prepared / "source_dataset"
    kernel.mkdir(parents=True)
    (prepared / "package").mkdir()
    from test_experiment_workflow import _arm, _spec_payload
    spec = ExperimentSpec(_spec_payload([_arm("neural_atom_k1", "2", name)
                                        for name in ("original", "optimized")], platform="kaggle"))
    (prepared / "package/experiment_spec.json").write_text(spec.to_json())
    manifest = {"package_identity": "a" * 64}
    atomic_json(prepared / "workflow_report.json", {"status": "PREPARED_FOR_PLATFORM", **manifest})
    if staged_source is None:
        metadata = {"id": "owner/new-kernel", "is_private": True,
                    "dataset_sources": ["owner/new-source", "owner/fixed-data"]}
        dataset_ref = "owner/new-source"
    else:
        metadata = json.loads((Path(staged_source).parent / "kernel/kernel-metadata.json").read_bytes())
        dataset_ref = json.loads((Path(staged_source) / "dataset-metadata.json").read_bytes())["id"]
    atomic_json(kernel / "kernel-metadata.json", metadata)
    (kernel / "run.py").write_bytes(b"print('synthetic')\n")
    if staged_source is not None:
        shutil.copytree(staged_source, staged)
    else:
        staged.mkdir()
        atomic_json(staged / "dataset-metadata.json", {"id": "owner/new-source"})
        atomic_json(staged / "experiment_launch.json", {"account": "owner", "run_reference": metadata["id"], "accelerator": "NvidiaTeslaT4",
            "jobs": [{"arm_id": binding["arm_id"], "device": device,
                      "recipe": f"recipes/{binding['arm_id']}.json",
                      "initial_state": f"initial_states/{binding['arm_id']}.pt",
                      "trajectory_id": binding["trajectory_id"]}
                     for device, binding in enumerate(spec.to_dict()["prospective"]["arms"])]})
        from molgap.experiment_package import SIDECARS
        for name in SIDECARS:
            (staged / ("source_payload.bin" if name == "source.tar.gz" else name)).write_bytes(b"synthetic")
        for arm in ("original", "optimized"):
            (staged / "initial_states").mkdir(exist_ok=True)
            (staged / "initial_states" / (arm + ".pt")).write_bytes(b"synthetic tensor bytes")
            path = staged / "prospective" / arm
            path.mkdir(parents=True)
            (path / "trajectory.json").write_bytes(b"{}")
    atomic_json(prepared / "release_report.json", {"inputs": {"recipe_files": {}, "required_modules": [],
        "initial_states": {"original": "a", "optimized": "b"}}})
    credentials = tmp_path / "outside-key.json"
    atomic_json(credentials, {"username": "owner", "key": "do-not-print"})
    monkeypatch.setattr(owner, "verify_experiment_source_package", lambda _, **kw: manifest)
    monkeypatch.setattr(owner, "read_packaged_text", lambda *_, **kw: json.dumps({"user_authorized": True,
        "allocated_t4_hours_ceiling": 14, "arms": ["original", "optimized"], "seeds": [42]}))
    monkeypatch.setattr(owner, "_verify_release_report", lambda *_, **kw: {})
    import molgap.experiment_preflight as preflight
    monkeypatch.setattr(preflight, "check_release_inputs", lambda *a, **k: {"errors": []})
    monkeypatch.setattr(owner, "build_launch_receipt", lambda *a, **k: {"binding": {"synthetic": True}})
    def reconcile(_spec, _package, raw, **kwargs):
        envelope = json.loads(raw)
        assert envelope["physical_runs"]["value"][0]["run_identity"] == "kernel77/version1"
        assert envelope["canonical_platform_reference"]["value"] == "owner/new-kernel"
        return {"submission_state": "ACCEPTED", "binding": envelope["binding"],
                "canonical_platform_reference": envelope["canonical_platform_reference"],
                "platform_version": envelope["platform_version"], "physical_runs": envelope["physical_runs"]}
    monkeypatch.setattr(owner, "reconcile_platform_response", reconcile)
    import molgap.experiment_launch as launch
    monkeypatch.setattr(launch, "read_launch_receipt", lambda path, *a, **k: json.loads(path.read_bytes()))

    class Api:
        CONFIG_NAME_USER = "user"
        created = 0
        pushed = 0
        dataset = False
        ready = True
        remote = None

        def get_config_value(self, _name):
            return metadata["id"].split("/", 1)[0]

        def dataset_status(self, ref):
            if not self.dataset:
                raise Missing()
            return "ready" if self.ready else "processing"

        def dataset_create_new(self, path, **kwargs):
            assert kwargs == dict(public=False, quiet=True, convert_to_csv=False, dir_mode="zip")
            self.created += 1
            self.dataset = True
            return NS(ref="/datasets/" + dataset_ref, error="")

        def dataset_list_files(self, ref, page_size):
            return NS(files=[NS(name=p.relative_to(staged).as_posix(), total_bytes=p.stat().st_size)
                             for p in staged.rglob("*") if p.is_file() and p.name != "dataset-metadata.json"])

        def kernels_status(self, ref):
            assert ref == metadata["id"] + "/1"
            return NS(status="RUNNING", failure_message="")

    api = Api()
    monkeypatch.setattr(owner, "_dataset_details", lambda _api, ref:
                        NS(ref=ref, is_private=True, current_version_number=1))
    monkeypatch.setattr(owner, "_kernel_source", lambda _api, _ref: api.remote)
    def accepted():
        api.remote = NS(metadata=NS(id=77, current_version_number=1, is_private=True,
            enable_gpu=True, machine_shape="NvidiaTeslaT4", dataset_data_sources=metadata["dataset_sources"]),
            blob=NS(source=(kernel / "run.py").read_text()))
        return {"status": "submitted", "kernel": metadata["id"], "kernel_id": 77,
                "version_number": 1, "reconciliation_required": False}
    def push(**kwargs):
        _dispatch_before_post(kwargs)
        api.pushed += 1
        return accepted()
    monkeypatch.setattr(owner, "push_kernel_with_accelerator", push)

    class Http:
        auth = None
        downloads = []
        quota_hours = 30

        def get(self, url, **kwargs):
            if url.endswith("/quota"):
                seconds = lambda n: {"seconds": n * 3600, "nanos": 0}
                return Response({"gpuQuota": {"totalTimeAllowed": seconds(self.quota_hours),
                    "timeUsed": seconds(0), "timeReserved": seconds(0)}})
            if "/logs/stream/" in url:
                return Response()
            from urllib.parse import unquote
            assert "datasetVersionNumber=1" in url
            dataset_path = "/datasets/download/" + dataset_ref + "/"
            name = unquote(url.split(dataset_path, 1)[1].split("?", 1)[0])
            self.downloads.append(name)
            return Response(content=(staged / name).read_bytes())
    http = Http()
    def run(execute=True):
        return owner.submit_prepared_workflow(prepared, credentials, output, authority_ref="authority.json",
            monitor_paths=["experiments/example/STATUS.md"], execute=execute, api=api, http_session=http)
    return NS(run=run, api=api, http=http, output=output, staged=staged, kernel=kernel,
              prepared=prepared, accepted=accepted, credentials=credentials)


def test_check_has_no_network_or_filesystem_publication(release):
    assert release.run(False)["status"] == "READY_FOR_PLATFORM_OPERATIONS"
    assert not release.output.exists()
    assert release.api.created == release.api.pushed == 0


def test_full_submission_and_repeated_call_never_duplicate(release):
    result = release.run()
    assert result["status"] == "SUBMISSION_BOUND" and result["version"] == 1
    receipt = (release.output / "launch_receipt.json").read_bytes()
    assert release.api.created == release.api.pushed == 1
    assert len(release.http.downloads) == 11
    assert release.run()["status"] == "SUBMISSION_BOUND"
    assert release.api.created == release.api.pushed == 1
    assert (release.output / "launch_receipt.json").read_bytes() == receipt
    assert "do-not-print" not in ''.join(p.read_text(errors="replace") for p in release.output.glob("*.json"))


def test_source_processing_resumes_without_second_create(release):
    release.api.ready = False
    assert release.run()["status"] == "WAIT_SOURCE_READY"
    assert release.api.created == 1 and release.api.pushed == 0
    release.api.ready = True
    assert release.run()["status"] == "SUBMISSION_BOUND"
    assert release.api.created == release.api.pushed == 1


def test_lost_kernel_response_reconciles_matching_source_without_repost(release, monkeypatch):
    def lost(**kwargs):
        _dispatch_before_post(kwargs)
        release.api.pushed += 1
        release.accepted()
        return {"status": "submission_unknown", "reconciliation_required": True}
    monkeypatch.setattr(owner, "push_kernel_with_accelerator", lost)
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    assert release.run()["status"] == "SUBMISSION_BOUND"
    assert release.api.pushed == 1


def test_unobserved_kernel_timeout_never_retries_post(release, monkeypatch):
    def local_failure(**kwargs):
        raise OSError("synthetic local failure before POST dispatch")
    monkeypatch.setattr(owner, "push_kernel_with_accelerator", local_failure)
    blocked = release.run()
    assert blocked["status"] == "BLOCKED"
    assert blocked["stage"] == "kernel_preparation" and blocked["submitted"] is False
    state = json.loads((release.output / "submission_state.json").read_bytes())
    assert state["phase"] == "SOURCE_PUBLISHED"
    assert release.api.created == 1 and release.api.pushed == 0

    def lost(**kwargs):
        _dispatch_before_post(kwargs)
        release.api.pushed += 1
        raise TimeoutError("secret-bearing transport error")
    monkeypatch.setattr(owner, "push_kernel_with_accelerator", lost)
    unknown = release.run()
    assert unknown["status"] == "RECONCILIATION_REQUIRED"
    assert unknown["submitted"] == "UNKNOWN"
    state = json.loads((release.output / "submission_state.json").read_bytes())
    assert state["phase"] == "SUBMIT_UNKNOWN"
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    assert release.api.created == 1 and release.api.pushed == 1


def test_unobserved_dataset_timeout_never_retries_create(release, monkeypatch):
    def lost(*args, **kwargs):
        release.api.created += 1
        raise TimeoutError()
    monkeypatch.setattr(release.api, "dataset_create_new", lost)
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    assert release.api.created == 1 and release.api.pushed == 0


def test_source_timeout_with_created_dataset_reconciles(release, monkeypatch):
    real_create = release.api.dataset_create_new
    def lost(*args, **kwargs):
        real_create(*args, **kwargs)
        raise TimeoutError()
    monkeypatch.setattr(release.api, "dataset_create_new", lost)
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    assert release.run()["status"] == "SUBMISSION_BOUND"
    assert release.api.created == 1


def test_quota_shortfall_stops_before_publication(release):
    release.http.quota_hours = 13
    with pytest.raises(ValueError, match="quota"):
        release.run()
    assert release.api.created == release.api.pushed == 0


def test_extra_upload_file_is_rejected_before_publication(release):
    (release.staged / "accidental-secret.json").write_text("secret")
    with pytest.raises(ValueError, match="unbound files"):
        release.run()
    assert release.api.created == 0


def test_remote_source_corruption_blocks_kernel_post(release, monkeypatch):
    real_get = release.http.get
    def corrupt(url, **kwargs):
        response = real_get(url, **kwargs)
        if "/datasets/download/" in url:
            response.content = b"corrupt"
        return response
    monkeypatch.setattr(release.http, "get", corrupt)
    with pytest.raises(ValueError, match="hash"):
        release.run()
    assert release.api.pushed == 0


def test_parallel_transport_starts_after_all_parent_directories_exist(release, monkeypatch):
    import molgap.kaggle_output_retrieval as transport
    real_download = transport._download_selected
    def checked(**kwargs):
        root = kwargs["destination"]
        assert all((root / path).is_dir() for path in
                   ("initial_states", "prospective/original", "prospective/optimized"))
        return real_download(**kwargs)
    monkeypatch.setattr(transport, "_download_selected", checked)
    assert release.run()["status"] == "SUBMISSION_BOUND"


def test_unknown_remote_identity_is_not_absence(release, monkeypatch):
    def unknown(*args):
        raise RuntimeError("API uncertainty")
    monkeypatch.setattr(owner, "_kernel_source", unknown)
    with pytest.raises(RuntimeError):
        release.run()
    assert release.api.created == release.api.pushed == 0


def test_remote_version_drift_blocks_binding_and_never_reposts(release):
    assert release.run()["status"] == "SUBMISSION_BOUND"
    release.api.remote.metadata.current_version_number = 2
    with pytest.raises(ValueError, match="binding differs"):
        release.run()
    assert release.api.pushed == 1


def test_other_process_lock_blocks_all_mutation(release):
    release.output.mkdir()
    (release.output / ".submission.lock").write_text("another process")
    with pytest.raises(ValueError, match="lock exists"):
        release.run()
    assert release.api.created == release.api.pushed == 0


def test_frozen_authority_cannot_expand_arms_or_seeds(release, monkeypatch):
    monkeypatch.setattr(owner, "read_packaged_text", lambda *_: json.dumps({"user_authorized": True,
        "allocated_t4_hours_ceiling": 14, "arms": ["original", "optimized"], "seeds": [42, 43]}))
    with pytest.raises(ValueError, match="exactly these arms and seeds"):
        release.run()
    assert release.api.created == release.api.pushed == 0


def test_changed_monitor_binding_cannot_reset_a_journal(release, monkeypatch):
    release.api.ready = False
    assert release.run()["status"] == "WAIT_SOURCE_READY"
    state = json.loads((release.output / "submission_state.json").read_bytes())
    state["binding"]["kernel"] = "owner/another-kernel"
    atomic_json(release.output / "submission_state.json", state)
    with pytest.raises(ValueError, match="inputs or authority changed"):
        release.run()
    assert release.api.created == 1 and release.api.pushed == 0


def test_desktop_acceptance_attachments_use_the_staging_owners_allowlist(release, monkeypatch):
    import molgap.kaggle_workflow as staging
    launch_path = release.staged / "experiment_launch.json"
    launch = json.loads(launch_path.read_bytes())
    plan = release.staged / "acceptance/experiments/synthetic/acceptance.json"
    plan.parent.mkdir(parents=True)
    plan.write_bytes(b"{}")
    pin = owner.sha256_file(plan)
    launch["target_identity"] = {"plan_path": "experiments/synthetic/acceptance.json", "plan_sha256": pin}
    atomic_json(launch_path, launch)
    calls = []
    def pins(spec, root, path):
        calls.append((root, path))
        return {path: pin}
    monkeypatch.setattr(staging, "acceptance_input_pins", pins)
    assert release.run(False)["status"] == "READY_FOR_PLATFORM_OPERATIONS"
    assert calls == [(release.staged / "acceptance", "experiments/synthetic/acceptance.json")]
    (plan.parent / "extra.json").write_bytes(b"secret")
    with pytest.raises(ValueError, match="unbound files"):
        release.run(False)
    assert release.api.created == release.api.pushed == 0


def test_changed_physical_slug_cannot_bind_the_requested_journal(release):
    response = release.accepted()
    response["kernel"] = "owner/another-kernel"
    metadata = json.loads((release.kernel / "kernel-metadata.json").read_bytes())
    with pytest.raises(ValueError, match="binding differs"):
        owner._verify_remote_kernel(release.api.remote, response, metadata, (release.kernel / "run.py").read_bytes())


def test_connector_reuses_explicit_credentials_without_global_auth(monkeypatch, tmp_path):
    import molgap.kaggle_credentials as credentials
    path = tmp_path / "key.json"
    atomic_json(path, {"username": "owner", "key": "synthetic"})
    seen = []
    api = object()
    def connect(actual):
        seen.append(actual)
        return api
    monkeypatch.setattr(credentials, "api_for_credentials", connect)
    assert owner.connect_kaggle(path) is api
    assert seen == [path]


def test_dataset_status_matches_installed_sdk_signature(release):
    release.api.dataset = True
    assert owner._dataset_status(release.api, "owner/new-source") == {"status": "ready"}
    release.api.ready = False
    assert owner._dataset_status(release.api, "owner/new-source") == {"status": "processing"}


@pytest.mark.parametrize("ref,private", [("owner/new-source", True), ("other/source", True), ("owner/new-source", False)])
def test_dataset_details_uses_sdk_request_and_checks_identity(ref, private):
    from contextlib import nullcontext
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    def get_dataset(request):
        assert isinstance(request, ApiGetDatasetRequest)
        assert (request.owner_slug, request.dataset_slug) == ("owner", "new-source")
        return NS(ref=ref, is_private=private, current_version_number=1)
    client = NS(datasets=NS(dataset_api_client=NS(get_dataset=get_dataset)))
    api = NS(build_kaggle_client=lambda: nullcontext(client))
    if ref == "owner/new-source" and private:
        assert owner._dataset_details(api, ref).current_version_number == 1
    else:
        with pytest.raises(ValueError, match="identity/private"):
            owner._dataset_details(api, "owner/new-source")


def test_source_version_drift_blocks_post(release, monkeypatch):
    monkeypatch.setattr(owner, "_dataset_details", lambda _api, ref:
                        NS(ref=ref, is_private=True, current_version_number=2))
    with pytest.raises(ValueError, match="version1"):
        release.run()
    assert release.api.created == 1 and release.api.pushed == 0


def test_remote_release_keeps_dataset_and_pickle_bindings(release, monkeypatch):
    import molgap.experiment_preflight as preflight
    prior = json.loads((release.prepared / "release_report.json").read_bytes())
    prior["inputs"].update(dataset_manifests={"original": "bound-manifest.json"},
                           pickle_inputs=["bound-serialization-input.pt"])
    atomic_json(release.prepared / "release_report.json", prior)
    observed = []
    def check(*args, **kwargs):
        observed.append(kwargs)
        return {"errors": []}
    monkeypatch.setattr(preflight, "check_release_inputs", check)
    assert release.run()["status"] == "SUBMISSION_BOUND"
    assert observed[0]["dataset_manifests"] == prior["inputs"]["dataset_manifests"]
    assert observed[0]["pickle_inputs"] == prior["inputs"]["pickle_inputs"]


@pytest.mark.parametrize("copies", [1, 2])
def test_packaged_authority_reader_requires_one_regular_member(tmp_path, copies):
    import io
    import tarfile
    from molgap.experiment_package import read_packaged_text
    with tarfile.open(tmp_path / "source.tar.gz", "w:gz") as archive:
        for _ in range(copies):
            member = tarfile.TarInfo("experiments/synthetic/authority.json")
            member.size = 2
            archive.addfile(member, io.BytesIO(b"{}"))
    if copies == 1:
        assert read_packaged_text(tmp_path, "experiments/synthetic/authority.json") == "{}"
    else:
        with pytest.raises(ValueError, match="one regular"):
            read_packaged_text(tmp_path, "experiments/synthetic/authority.json")


def test_bound_upload_retires_only_transient_copies_and_retains_receipts(release):
    result = release.run()
    assert result["source_copy_retirement"]["status"] == "REMOVED"
    assert not (release.output / "source_roundtrip").exists()
    assert not (release.staged / "source_payload.bin").exists()
    assert not (release.staged / "initial_states").exists()
    assert (release.staged / "experiment_launch.json").is_file()
    assert (release.output / "source_upload_verification.json").is_file()
    assert (release.prepared / "package/experiment_spec.json").is_file()
    assert release.run()["status"] == "SUBMISSION_BOUND"
    assert release.run(execute=False)["status"] == "SUBMISSION_BOUND"
    assert release.api.created == release.api.pushed == 1


def test_bound_cleanup_dry_check_validates_receipt_without_resubmitting(release, monkeypatch):
    submitted = release.run()
    assert submitted["status"] == "SUBMISSION_BOUND"
    assert not (release.staged / "source_payload.bin").exists()
    assert not (release.staged / "initial_states").exists()
    before = (release.api.created, release.api.pushed, len(release.http.downloads))
    monkeypatch.setattr(owner, "_verify_release_report",
                        lambda *a, **k: pytest.fail("retired release paths must not be re-read"))

    observed = release.run(execute=False)

    assert observed["status"] == "SUBMISSION_BOUND" and observed["submitted"] is True
    assert observed["kernel"] == "owner/new-kernel" and observed["version"] == 1
    assert observed["source_copy_retirement"]["status"] == "REMOVED"
    assert (release.api.created, release.api.pushed, len(release.http.downloads)) == before

    journal = release.output / "submission_state.json"
    state = json.loads(journal.read_bytes())
    state["kernel_response"]["version_number"] = 2
    atomic_json(journal, state)
    with pytest.raises(ValueError, match="launch receipt differs"):
        release.run(execute=False)


def test_uncertain_upload_or_kernel_keeps_all_recovery_copies(release, monkeypatch):
    def unknown(**kwargs):
        _dispatch_before_post(kwargs)
        return {"status": "submission_unknown", "reconciliation_required": True}
    monkeypatch.setattr(owner, "push_kernel_with_accelerator", unknown)
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    assert (release.staged / "source_payload.bin").is_file()
    assert (release.output / "source_roundtrip/source_payload.bin").is_file()


def test_crash_during_bound_source_retirement_recovers_without_reposting(release, monkeypatch):
    from molgap import local_storage
    remove = local_storage._remove_verified_files
    crashed = []
    def interrupted(root, expected, **kw):
        if not crashed:
            crashed.append(True)
            first = next(iter(expected))
            (root / first).unlink()
            raise KeyboardInterrupt("synthetic power loss")
        return remove(root, expected, **kw)
    monkeypatch.setattr(local_storage, "_remove_verified_files", interrupted)
    with pytest.raises(KeyboardInterrupt):
        release.run()
    state = json.loads((release.output / "submission_state.json").read_bytes())
    assert state["phase"] == "BOUND" and state["source_copy_retirement"]["status"] == "PENDING"
    assert release.run()["source_copy_retirement"]["status"] == "REMOVED"
    assert release.api.created == release.api.pushed == 1


def _packaged_release(case, tmp_path, monkeypatch):
    """Run the compiled editable input through the platform entry with HTTP-only transport."""
    import requests
    from molgap.common.paths import MachinePaths

    paths = MachinePaths.from_file(case.value["machine_profile"])
    run = paths.run_directory(case.value["logical_run_id"], case.value["attempt_id"])
    prepared, output = run / "prepared", run / "operations"
    platform = json.loads((case.repo / case.value["platform_ref"]).read_bytes())
    account = platform["kaggle"]["account"]
    credentials = tmp_path / "outside-key.json"
    atomic_json(credentials, {"username": account, "key": "fixture-key"})

    entry_path = case.repo / "platforms/kaggle/submit_workflow.py"
    entry_spec = importlib.util.spec_from_file_location("joined_kaggle_entry", entry_path)
    entry = importlib.util.module_from_spec(entry_spec)
    entry_spec.loader.exec_module(entry)
    transport_root = tmp_path / "fake-platform"
    transport_root.mkdir()
    transport_state = {"transport": None, "tamper_before_push": True,
                       "tampered_roundtrip": None, "lose_next_post": False}

    def transport():
        if transport_state["transport"] is None:
            # Preparation has produced its genuine staged source before
            # submit_prepared_workflow asks for the first platform client.
            with monkeypatch.context() as stubs:
                transport_state["transport"] = release.__wrapped__(
                    transport_root, stubs, staged_source=prepared / "source_dataset")
            shutil.copyfile(prepared / "kernel/run.py", transport_state["transport"].kernel / "run.py")
        return transport_state["transport"]

    def offline(*args, **kwargs):
        raise AssertionError("This regression has no network authority")

    monkeypatch.setattr("requests.sessions.Session.request", offline)
    monkeypatch.setattr(owner, "connect_kaggle", lambda *_: transport().api)
    monkeypatch.setattr(owner, "_kernel_source", lambda *_: transport().api.remote)
    monkeypatch.setattr(owner, "_dataset_details", lambda _api, ref:
                        NS(ref=ref, is_private=True, current_version_number=1))
    monkeypatch.setattr(requests, "Session", lambda: transport().http)

    real_push = owner.push_kernel_with_accelerator

    def mutate_roundtrip_before_recheck(**kwargs):
        if transport_state["tamper_before_push"]:
            path = output / "source_roundtrip/experiment_launch.json"
            original = path.read_bytes()
            value = json.loads(original)
            value["run_reference"] = "changed/after-roundtrip"
            path.write_bytes(canonical_json(value).encode())
            transport_state["tampered_roundtrip"] = (path, original)
            transport_state["tamper_before_push"] = False
        return real_push(**kwargs)

    monkeypatch.setattr(owner, "push_kernel_with_accelerator", mutate_roundtrip_before_recheck)

    def post(url, *, auth, json, timeout):
        current = transport()
        metadata = json_module.loads((prepared / "kernel/kernel-metadata.json").read_bytes())
        assert url == "https://www.kaggle.com/api/v1/kernels/push"
        assert json["slug"] == metadata["id"]
        assert json["machineShape"] == "NvidiaTeslaT4"
        assert json["text"].encode() == (prepared / "kernel/run.py").read_bytes()
        current.api.pushed += 1
        current.accepted()
        if transport_state["lose_next_post"]:
            transport_state["lose_next_post"] = False
            raise requests.Timeout("synthetic response loss after accepted POST")
        return Response({"ref": metadata["id"], "kernelId": 77, "versionNumber": 1})

    import json as json_module
    monkeypatch.setattr("requests.post", post)

    def run():
        from contextlib import redirect_stdout
        from io import StringIO
        captured = StringIO()
        with redirect_stdout(captured):
            code = entry.main(["--experiment", str(case.path), "--repo-root", str(case.repo),
                "--credentials", str(credentials), "--execute", "--verbose"])
        return code, json.loads(captured.getvalue())

    def restore_roundtrip():
        path, original = transport_state["tampered_roundtrip"]
        path.write_bytes(original)

    def lose_next_post():
        transport_state["lose_next_post"] = True

    return NS(run=run, repo=case.repo, value=case.value, input_path=case.path, prepared=prepared,
              output=output, credentials=credentials, transport=transport,
              tampered_roundtrip=lambda: transport_state["tampered_roundtrip"],
              restore_roundtrip=restore_roundtrip, lose_next_post=lose_next_post)


def _assert_real_package_release_post_receipt_reuses_only_exact_content(case, monkeypatch):
    from molgap import experiment_package as package_owner
    from molgap.experiment_launch import read_launch_receipt
    from molgap.experiment_spec import ExperimentSpec
    verifications = []
    verify = package_owner._verify_package_snapshot
    def counted(*args, **kwargs):
        verifications.append(True)
        return verify(*args, **kwargs)
    monkeypatch.setattr(package_owner, "_verify_package_snapshot", counted)
    verification_counts = []
    code, first = case.run()
    assert code == 1
    assert first["status"] == "BLOCKED", first
    assert first["stage"] == "release_recheck", json.dumps(first, indent=2)
    assert len(verifications) == 1
    verification_counts.append(len(verifications))
    assert first["submitted"] is False
    assert first["release_recheck"]["category"] in {"binding_mismatch", "check_failed"}
    assert "workflow" in first["release_recheck"]["changed_checks"]
    state = json.loads((case.output / "submission_state.json").read_bytes())
    assert state["phase"] == "SOURCE_PUBLISHED"
    recheck = state["release_recheck"]
    assert recheck == first["release_recheck"]
    assert case.tampered_roundtrip() is not None
    assert not (case.output / "launch_receipt.json").exists()
    assert case.transport().api.created == 1 and case.transport().api.pushed == 0
    source = case.prepared / "source_dataset"
    launch = json.loads((source / "experiment_launch.json").read_bytes())
    assert launch["allocation_wall_limit_seconds"] == 32400
    assert launch["cleanup_reserve_seconds"] == 180
    assert len(launch["jobs"]) == 2
    initial_paths = {job["initial_state"] for job in launch["jobs"]}
    assert len(initial_paths) == 1
    assert len(list((source / "initial_states").glob("*.pt"))) == 1
    shared_path = initial_paths.pop()
    assert (case.output / "source_roundtrip" / shared_path).read_bytes() == (source / shared_path).read_bytes()
    verifications.clear()
    case.restore_roundtrip()
    case.lose_next_post()
    code, lost = case.run()
    assert code == 1
    assert lost["status"] == "RECONCILIATION_REQUIRED", lost
    assert lost["submitted"] == "UNKNOWN"
    assert not (case.output / "launch_receipt.json").exists()
    assert len(verifications) == 1
    verification_counts.append(len(verifications))
    assert case.transport().api.created == case.transport().api.pushed == 1
    verifications.clear()
    code, result = case.run()
    assert code == 0
    assert result["status"] == "SUBMISSION_BOUND", result
    assert len(verifications) == 1
    verification_counts.append(len(verifications))
    assert case.transport().api.created == case.transport().api.pushed == 1
    verifications.clear()
    receipt_path = case.output / "launch_receipt.json"
    manifest = json.loads((case.prepared / "package/package_manifest.json").read_bytes())
    spec = ExperimentSpec.from_json((case.prepared / "package/experiment_spec.json").read_text())
    receipt = read_launch_receipt(receipt_path, spec, case.prepared / "package",
        expected_package_identity=manifest["package_identity"])
    assert receipt["submission_state"] == "ACCEPTED"
    assert receipt["physical_runs"]["value"][0]["run_identity"] == "kernel77/version1"
    assert receipt["binding"]["source_archive_sha256"] == manifest["archive_sha256"]
    assert manifest["configuration_source"]["path"] == case.input_path.relative_to(case.repo).as_posix()
    assert (case.output / "submission_state.json").is_file()
    retained = receipt_path.read_bytes()
    verifications.clear()
    code, repeated = case.run()
    assert code == 0
    assert repeated["status"] == "SUBMISSION_BOUND"
    assert len(verifications) == 1
    verification_counts.append(len(verifications))
    verifications.clear()
    with monkeypatch.context() as settled:
        settled.setattr(owner, "_kernel_source", lambda *a, **k:
            pytest.fail("a bound submission must not query the latest remote version"))
        code, settled_result = case.run()
        assert code == 0
        assert settled_result["status"] == "SUBMISSION_BOUND"
        assert len(verifications) == 1
        verification_counts.append(len(verifications))
    assert receipt_path.read_bytes() == retained
    assert case.transport().api.created == case.transport().api.pushed == 1
    # An independent call must fully validate again, even if only one sidecar
    # changed after this transaction's successful content checks.
    (case.prepared / "package/SOURCE_COMMIT.txt").write_text("0" * 40 + "\n")
    with pytest.raises(ValueError):
        owner.submit_prepared_workflow(case.prepared, case.credentials, case.output,
            authority_ref=case.value["authority_ref"],
            monitor_paths=case.value["monitor_paths"], execute=False)
    assert case.transport().api.created == case.transport().api.pushed == 1
    return {"status": result["status"], "family": case.value["arms"][0]["model"]["family"],
            "verification_counts": verification_counts, "api_created": case.transport().api.created,
            "api_pushed": case.transport().api.pushed}


def test_real_package_release_post_receipt_reuses_only_exact_content(scientific, tmp_path):
    case = scientific
    from copy import deepcopy
    from test_scientific_input import _git
    value = deepcopy(case.value)
    second = deepcopy(value["arms"][0])
    second["arm_id"] = "candidate_two"
    value["arms"].append(second)
    atomic_json(case.path, value)
    authority_path = case.repo / value["authority_ref"]
    authority = json.loads(authority_path.read_bytes())
    authority["arms"].append("candidate_two")
    atomic_json(authority_path, authority)
    platform_path = case.repo / value["platform_ref"]
    platform = json.loads(platform_path.read_bytes())
    platform["kaggle"].update(allocation_wall_limit_seconds=32400, cleanup_reserve_seconds=180)
    atomic_json(platform_path, platform)
    _git(case.repo, "add", "--", case.path.relative_to(case.repo).as_posix(),
         value["authority_ref"], value["platform_ref"])
    _git(case.repo, "-c", "commit.gpgsign=false", "commit", "-m", "Explicit paired fixture with shared initial state and launch budget")
    child = r'''
import json, pathlib, sys
from types import SimpleNamespace as NS
repo, experiment, temp, tests = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(repo / "src"))
sys.path.insert(1, str(tests))
import pytest
import test_kaggle_submission as journey
case = NS(repo=repo, path=experiment, value=json.loads(experiment.read_bytes()), tmp=experiment.parents[3])
patch = pytest.MonkeyPatch()
try:
    release = journey._packaged_release(case, temp, patch)
    summary = journey._assert_real_package_release_post_receipt_reuses_only_exact_content(release, patch)
finally:
    patch.undo()
print(json.dumps(summary))
'''
    completed = subprocess.run([sys.executable, "-I", "-c", child,
        str(case.repo), str(case.path), str(tmp_path), str(Path(__file__).resolve().parent)],
        cwd=case.repo, text=True, capture_output=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    observation = json.loads(completed.stdout.splitlines()[-1])
    assert observation["status"] == "SUBMISSION_BOUND"
    assert observation["verification_counts"] == [1, 1, 1, 1, 1]
    assert observation["api_created"] == observation["api_pushed"] == 1


def test_same_prepared_release_cannot_switch_journals_after_unknown(release, monkeypatch, tmp_path):
    def lost(**kwargs):
        _dispatch_before_post(kwargs)
        release.api.pushed += 1
        raise TimeoutError()
    monkeypatch.setattr(owner, "push_kernel_with_accelerator", lost)
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    # Windows spelling changes do not create another journal identity.
    import os
    if os.name == "nt":
        state = json.loads((release.output / "submission_state.json").read_bytes())
        owner._bind_operation_directory(Path(str(release.prepared).swapcase()),
            Path(str(release.output).swapcase()), state["binding"])
    for execute in (False, True):
        with pytest.raises(ValueError, match="owns another submission journal"):
            owner.submit_prepared_workflow(release.prepared, release.credentials, tmp_path / "other-journal",
                authority_ref="authority.json", monitor_paths=["experiments/example/STATUS.md"],
                execute=execute, api=release.api, http_session=release.http)
    assert release.api.created == release.api.pushed == 1


def test_concurrent_other_journal_is_blocked_before_first_publication(release, monkeypatch, tmp_path):
    create = release.api.dataset_create_new
    def interleaved(*args, **kwargs):
        with pytest.raises(ValueError, match="owns another submission journal"):
            owner.submit_prepared_workflow(release.prepared, release.credentials, tmp_path / "other-journal",
                authority_ref="authority.json", monitor_paths=["experiments/example/STATUS.md"],
                execute=True, api=release.api, http_session=release.http)
        return create(*args, **kwargs)
    monkeypatch.setattr(release.api, "dataset_create_new", interleaved)
    assert release.run()["status"] == "SUBMISSION_BOUND"
    assert release.api.created == release.api.pushed == 1
