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
def release(tmp_path, monkeypatch, request, staged_source=None):
    continuation = getattr(request, "param", False)
    version = 2 if continuation else 1
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
    recovery = {}
    if continuation and staged_source is None:
        from molgap.training_reproducibility import sha256_file
        original = tmp_path / "original"
        original.mkdir()
        shutil.move(prepared / "package", original / "package")
        receipt = original / "launch_receipt.json"
        atomic_json(receipt, {"submission_state": "ACCEPTED",
            "canonical_platform_reference": {"value": metadata["id"], "missing_reason": None}})
        atomic_json(prepared / "workflow_report.json", {"format": "molgap-workflow-recovery-preparation-v1",
            "status": "RESUME_INPUTS_PREPARED", **manifest,
            "original_prepared": str(original), "original_receipt": str(receipt)})
        launch_path = staged / "experiment_launch.json"
        config = json.loads(launch_path.read_bytes())
        config.update(allocation_wall_limit_seconds=18000, cleanup_reserve_seconds=180,
                      device_count=spec.to_dict()["platform"]["device_count"])
        config["resume"] = {}
        for arm in ("original", "optimized"):
            bundle_path = staged / "resume" / arm / "resume_bundle.json"
            bundle_path.parent.mkdir(parents=True)
            atomic_json(bundle_path, {"artifacts": {}, "runtime": [], "provenance": [], "context": [], "cost_segments": []})
            config["resume"][arm] = {"manifest": bundle_path.relative_to(staged).as_posix(),
                                     "sha256": sha256_file(bundle_path)}
        atomic_json(launch_path, config)
        authorization = tmp_path / "recovery_authority.md"
        authorization.write_text("Explicit synthetic continuation authority; original arms and ceiling retained.")
        recovery = dict(recovery_authority=authorization,
                        recovery_authority_sha256=sha256_file(authorization),
                        source_version=version, kernel_version=version)
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
        assert envelope["physical_runs"]["value"][0]["run_identity"] == f"kernel77/version{version}"
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
        dataset_version = 1
        predecessor_status = "RUNNING"

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

        def dataset_create_version(self, path, **kwargs):
            assert continuation and self.dataset_version == version - 1
            assert kwargs == dict(version_notes="Authorized frozen checkpoint continuation",
                quiet=True, convert_to_csv=False, delete_old_versions=False, dir_mode="zip")
            self.created += 1
            self.dataset_version = version
            return NS(ref="/datasets/" + dataset_ref, error="")

        def dataset_list_files(self, ref, page_size):
            return NS(files=[NS(name=p.relative_to(staged).as_posix(), total_bytes=p.stat().st_size)
                             for p in staged.rglob("*") if p.is_file() and p.name != "dataset-metadata.json"])

        def kernels_status(self, ref):
            if continuation and ref == metadata["id"] + "/1":
                return NS(status=self.predecessor_status, failure_message="")
            assert ref == metadata["id"] + "/" + str(version)
            return NS(status="RUNNING", failure_message="")

    api = Api()
    monkeypatch.setattr(owner, "_dataset_details", lambda _api, ref:
                        NS(ref=ref, is_private=True, current_version_number=api.dataset_version))
    monkeypatch.setattr(owner, "_kernel_source", lambda _api, _ref: api.remote)
    def accepted():
        api.remote = NS(metadata=NS(id=77, current_version_number=version, is_private=True,
            enable_gpu=True, machine_shape="NvidiaTeslaT4", dataset_data_sources=metadata["dataset_sources"]),
            blob=NS(source=(kernel / "run.py").read_text()))
        return {"status": "submitted", "kernel": metadata["id"], "kernel_id": 77,
                "version_number": version, "reconciliation_required": False}
    if continuation:
        accepted()
        api.remote.metadata.current_version_number = version - 1
        api.dataset = True
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
            assert f"datasetVersionNumber={version}" in url
            dataset_path = "/datasets/download/" + dataset_ref + "/"
            name = unquote(url.split(dataset_path, 1)[1].split("?", 1)[0])
            self.downloads.append(name)
            return Response(content=(staged / name).read_bytes())
    http = Http()
    def run(execute=True):
        return owner.submit_prepared_workflow(prepared, credentials, output, authority_ref="authority.json",
            monitor_paths=["experiments/example/STATUS.md"], execute=execute, api=api, http_session=http, **recovery)
    return NS(run=run, api=api, http=http, output=output, staged=staged, kernel=kernel,
              prepared=prepared, accepted=accepted, credentials=credentials)


def test_check_has_no_network_or_filesystem_publication(release):
    assert release.run(False)["status"] == "READY_FOR_PLATFORM_OPERATIONS"
    assert not release.output.exists()
    assert release.api.created == release.api.pushed == 0


@pytest.mark.parametrize("release", [False, True], indirect=True, ids=["fresh", "continuation"])
def test_full_submission_and_repeated_call_never_duplicate(release):
    continuation = release.api.remote is not None
    if continuation:
        assert type(json.loads((release.prepared / "workflow_report.json").read_bytes())["original_receipt"]) is str
        with pytest.raises(ValueError, match="authoritatively terminal"):
            release.run()
        assert release.api.created == release.api.pushed == 0
        release.api.predecessor_status = "ERROR"
    result = release.run()
    assert result["status"] == "SUBMISSION_BOUND" and result["version"] == (2 if continuation else 1)
    receipt = (release.output / "launch_receipt.json").read_bytes()
    assert release.api.created == release.api.pushed == 1
    assert len(release.http.downloads) == (13 if continuation else 11)
    if continuation:
        binding = json.loads((release.output / "submission_state.json").read_bytes())["binding"]["continuation"]
        assert binding["source_version"] == binding["kernel_version"] == 2
        from molgap.training_reproducibility import sha256_file
        assert binding["authority_sha256"] == sha256_file(Path(binding["authority_path"]))
        release.api.remote.metadata.current_version_number = 3
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
    import requests
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
        response = requests.Response()
        response.status_code = 403
        response.url = "https://www.kaggle.com/api/v1/kernels/push?token=do-not-print"
        response.headers["X-Request-Id"] = "kernel-request-42"
        raise requests.HTTPError("secret-bearing transport error", response=response)
    monkeypatch.setattr(owner, "push_kernel_with_accelerator", lost)
    unknown = release.run()
    assert unknown["status"] == "RECONCILIATION_REQUIRED"
    assert unknown["submitted"] == "UNKNOWN"
    state = json.loads((release.output / "submission_state.json").read_bytes())
    assert state["phase"] == "SUBMIT_UNKNOWN"
    assert state["platform_error"] == unknown["platform_error"] == {
        "error_type": "HTTPError", "stage": "kernel_publication",
        "endpoint_class": "kaggle_api", "http_status": 403, "request_id": "kernel-request-42"}
    assert "do-not-print" not in canonical_json(state)
    assert "secret-bearing" not in canonical_json(state)
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    assert release.api.created == 1 and release.api.pushed == 1


def test_unobserved_dataset_timeout_never_retries_create(release, monkeypatch):
    def lost(*args, **kwargs):
        release.api.created += 1
        raise TimeoutError("secret-bearing transport error")
    monkeypatch.setattr(release.api, "dataset_create_new", lost)
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    assert release.api.created == 1 and release.api.pushed == 0
    state = json.loads((release.output / "submission_state.json").read_bytes())
    assert state["platform_error"] == {"error_type": "TimeoutError",
        "stage": "source_publication", "endpoint_class": "dataset_create_new"}
    assert "secret-bearing" not in canonical_json(state)


def test_source_timeout_with_created_dataset_reconciles(release, monkeypatch, capsys):
    import requests
    real_create = release.api.dataset_create_new
    def lost(*args, **kwargs):
        real_create(*args, **kwargs)
        response = requests.Response()
        response.status_code = 503
        response.url = "https://storage.googleapis.com/private-upload?token=do-not-print"
        response.headers.update({"X-GUploader-UploadID": "upload-request-42", "Set-Cookie": "do-not-print"})
        response._content = b'{"message":"do-not-print"}'
        raise requests.HTTPError("do-not-print", response=response)
    monkeypatch.setattr(release.api, "dataset_create_new", lost)
    # Use the actual CLI serializer as well as the journal: neither default
    # compact output nor retained evidence may discard or leak the HTTP fact.
    entry_spec = importlib.util.spec_from_file_location("submission_http_fixture",
        Path(__file__).resolve().parents[1] / "platforms/kaggle/submit_workflow.py")
    entry = importlib.util.module_from_spec(entry_spec)
    entry_spec.loader.exec_module(entry)
    monkeypatch.setattr(owner, "connect_kaggle", lambda _: release.api)
    monkeypatch.setattr(requests, "Session", lambda: release.http)
    assert entry.main(["--prepared", str(release.prepared), "--credentials", str(release.credentials),
        "--output", str(release.output), "--authority", "authority.json",
        "--monitor-path", "experiments/example/STATUS.md", "--execute"]) == 1
    printed = capsys.readouterr().out
    compact = json.loads(printed)
    assert compact["status"] == "RECONCILIATION_REQUIRED" and compact["phase"] == "SOURCE_UNKNOWN"
    detail = {"error_type": "HTTPError", "stage": "source_publication",
        "endpoint_class": "google_storage", "http_status": 503, "request_id": "upload-request-42"}
    state = json.loads((release.output / "submission_state.json").read_bytes())
    assert state["platform_error"] == compact["platform_error"] == detail
    assert "do-not-print" not in printed + canonical_json(state)
    assert release.api.created == 1 and release.api.pushed == 0
    assert release.run()["status"] == "SUBMISSION_BOUND"
    assert release.api.created == release.api.pushed == 1
    assert json.loads((release.output / "submission_state.json").read_bytes())["platform_error"] == detail


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


def test_dataset_status_matches_installed_sdk_signature(release, monkeypatch):
    release.api.dataset = True
    assert owner._dataset_status(release.api, "owner/new-source") == {"status": "ready"}
    release.api.ready = False
    assert owner._dataset_status(release.api, "owner/new-source") == {"status": "processing"}
    import requests
    response = requests.Response()
    response.status_code = 200  # The installed SDK can raise HTTPError for JSON code.
    response.url = "https://www.kaggle.com/api/v1/datasets/status?token=do-not-print"
    response.headers.update({"X-Request-Id": "unsafe\nheader", "X-Kaggle-Request-Id": "status-request-42"})
    response._content = b'{"code":429,"message":"do-not-print"}'
    def denied(_):
        raise requests.HTTPError("do-not-print", response=response)
    monkeypatch.setattr(release.api, "dataset_status", denied)
    with pytest.raises(RuntimeError, match="Dataset status is unknown") as raised:
        owner._dataset_status(release.api, "owner/new-source")
    from molgap.common.kaggle import platform_error_details
    details = platform_error_details(raised.value, stage="platform_operations", endpoint_class="other_http")
    assert details == {"error_type": "HTTPError", "stage": "dataset_status",
        "endpoint_class": "kaggle_api", "http_status": 200, "api_status": 429,
        "request_id": "status-request-42"}
    assert "do-not-print" not in canonical_json(details) + str(raised.value)


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


def _packaged_release(case, tmp_path, monkeypatch, *, resume_prepared=None, recovery_authority=None):
    """Run the compiled editable input through the platform entry with HTTP-only transport."""
    import requests
    from molgap.common.paths import MachinePaths

    paths = MachinePaths.from_file(case.value["machine_profile"])
    run = paths.run_directory(case.value["logical_run_id"], case.value["attempt_id"])
    continuation = resume_prepared is not None
    version = 2 if continuation else 1
    prepared = Path(resume_prepared) if continuation else run / "prepared"
    output = prepared.parent / "operations" if continuation else run / "operations"
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
    transport_state = {"transport": None, "tamper_before_push": not continuation,
                       "tampered_roundtrip": None, "lose_next_post": False}

    def transport():
        if transport_state["transport"] is None:
            # Preparation has produced its genuine staged source before
            # submit_prepared_workflow asks for the first platform client.
            with monkeypatch.context() as stubs:
                transport_state["transport"] = release.__wrapped__(
                    transport_root, stubs, NS(param=continuation), staged_source=prepared / "source_dataset")
            if continuation:
                transport_state["transport"].api.predecessor_status = "ERROR"
            shutil.copyfile(prepared / "kernel/run.py", transport_state["transport"].kernel / "run.py")
        return transport_state["transport"]

    def offline(*args, **kwargs):
        raise AssertionError("This regression has no network authority")

    monkeypatch.setattr("requests.sessions.Session.request", offline)
    monkeypatch.setattr(owner, "connect_kaggle", lambda *_: transport().api)
    monkeypatch.setattr(owner, "_kernel_source", lambda *_: transport().api.remote)
    monkeypatch.setattr(owner, "_dataset_details", lambda _api, ref:
                        NS(ref=ref, is_private=True, current_version_number=transport().api.dataset_version))
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
        return Response({"ref": metadata["id"], "kernelId": 77, "versionNumber": version})

    import json as json_module
    monkeypatch.setattr("requests.post", post)

    def run():
        from contextlib import redirect_stdout
        from io import StringIO
        captured = StringIO()
        with redirect_stdout(captured):
            if continuation:
                from molgap.training_reproducibility import sha256_file
                args = ["--prepared", str(prepared), "--output", str(output),
                    "--authority", case.value["authority_ref"],
                    "--recovery-authority", str(recovery_authority),
                    "--recovery-authority-sha256", sha256_file(recovery_authority),
                    "--source-version", str(version), "--kernel-version", str(version)]
                for path in case.value["monitor_paths"]:
                    args.extend(["--monitor-path", path])
            else:
                args = ["--experiment", str(case.path)]
            code = entry.main([*args, "--repo-root", str(case.repo),
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


def _train_retrieve_and_close_frozen_addon(case, spec, manifest, monkeypatch):
    """Pause this submitted run, submit its actual bundles, restore and close it."""
    import time
    import torch
    from contextlib import redirect_stdout
    from io import StringIO
    from molgap.common.paths import MachinePaths
    from molgap.experiment_allocation import AllocationLedger
    from molgap.experiment_family_workflow import RunContext
    from molgap.experiment_package import read_packaged_text
    from molgap.experiment_retention import seal_execution_retention
    from molgap.experiment_run import accept_run_workflow, build_run_resume, prepare_run_resume
    from molgap.experiment_resume import restore_resume_bundle
    from molgap.family_training_data import load_training_data
    from molgap.family_training_kernel import FamilyKernel
    from molgap.family_training_loaders import FamilyLoaders
    from molgap.family_training_outputs import SessionOutputs
    from molgap.family_training_runner import _recipe_configuration, PROFILES
    from molgap.family_training_session import FamilyTrainingSession
    from molgap.gptrans_execution_tuning import ExecutionTuning
    from molgap.research_memory.compiler import frozen_differences
    from molgap.training_reproducibility import configure_fp32_determinism, sha256_file
    from test_kaggle_family_retrieval import _module, _Api, _File, _ResponsePage

    paths = MachinePaths.from_file(case.value["machine_profile"])
    run_id, attempt = case.value["logical_run_id"], case.value["attempt_id"]
    package = case.prepared / "package"
    frozen = json.loads((case.prepared.parent / "inputs/workflow_plan.json").read_bytes())
    launch = json.loads((case.prepared / "source_dataset/experiment_launch.json").read_bytes())
    origin = paths.artifact_root / "offline-worker-v1"
    remote = paths.artifact_root / "offline-worker-v2"
    origin.mkdir()
    remote.mkdir()
    contexts, decisions, audits, downloads, inputs, prefixes, bundles = {}, {}, {}, [], {}, {}, {}
    prospective = {binding["arm_id"]: (case.repo / binding["output"] / "trajectory.json").read_bytes()
                   for binding in spec.to_dict()["prospective"]["arms"]}
    for job in launch["jobs"]:
        arm_id = job["arm_id"]
        # The worker cannot observe its eventual receipt version. Keep that
        # producer field unknown instead of manufacturing the reconciled value.
        context = RunContext.for_training(spec, package, expected_package_identity=manifest["package_identity"],
            arm_id=arm_id, account=launch["account"], run_reference=launch["run_reference"])
        contract = paths.scratch_root / (arm_id + "-contract.json")
        contract.parent.mkdir(parents=True, exist_ok=True)
        contract.write_text(read_packaged_text(package, job["recipe"]), encoding="utf-8")
        record = json.loads(contract.read_bytes())
        recipe = _recipe_configuration(record)
        data = load_training_data(paths.data_root, recipe.dataset)
        initial = next(item["initial_state"] for item in frozen["arms"] if item["arm_id"] == arm_id)
        initial = Path(initial) if Path(initial).is_absolute() else case.repo / initial
        configure_fp32_determinism(recipe.seed)
        trajectory = json.loads(prospective[arm_id])
        output = SessionOutputs(origin / arm_id, context, contract=contract, adapter=PROFILES[recipe.family],
            trajectory_id=job["trajectory_id"], train_role_identity=record["train_role_identity"],
            resume_trajectory=trajectory)
        session = FamilyTrainingSession(FamilyKernel(recipe, initial, device="cpu"),
            FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0, development_workers=0)),
            checkpoint=output.checkpoint, selected=output.selected, pass_finished=output.pass_finished)
        session.batch_finished = output.batch_finished
        try:
            progress = session.run(should_pause=lambda: session.progress.optimizer_step >= 2)
            assert progress.status == "paused" and progress.optimizer_step == 2
            checkpoint = torch.load(output.root / "last_checkpoint.pt", map_location="cpu", weights_only=True)
            assert checkpoint["optimizer"]["state"]
            prefixes[arm_id] = json.loads((output.root / "canonical_trace.json").read_bytes())["observations"]
        finally:
            session.close()
        bundle = paths.artifact_root / "continuation" / "bundles" / arm_id
        exported = build_run_resume(case.value["machine_profile"], case.repo, run_id, attempt,
            arm_id=arm_id, source_output=origin / arm_id, output=bundle)
        assert exported["status"] == "RESUME_VERIFIED", exported
        assert exported["prospective_published"] is exported["submitted"] is False
        bundles[arm_id] = str(bundle)
        inputs[arm_id] = (context, contract, record, recipe, data, initial, trajectory)

    resumed_prepared = paths.artifact_root / "continuation" / "prepared"
    with monkeypatch.context() as preparation:
        def no_new_question(*args, **kwargs):
            pytest.fail("Checkpoint continuation must reuse the original prospective publication")
        preparation.setattr("molgap.experiment_workflow.plan_prospective", no_new_question)
        resumed = prepare_run_resume(case.value["machine_profile"], case.repo, run_id, attempt,
            plan={"format": "molgap-workflow-resume-v1", "spec_identity": spec.identity, "arms": bundles,
                  "allocation_wall_limit_seconds": 120, "cleanup_reserve_seconds": 5}, output=resumed_prepared)
    assert resumed["status"] == "RESUME_INPUTS_PREPARED", resumed
    assert resumed["prospective_published"] is False
    authority = paths.artifact_root / "continuation" / "authority.md"
    authority.write_text("Explicit synthetic continuation authority; local transport and CPU fixture only.\n")
    transport_tmp = paths.scratch_root / "continuation-transport"
    transport_tmp.mkdir()
    with monkeypatch.context() as transport:
        continued = _packaged_release(NS(repo=case.repo, path=case.input_path, value=case.value),
            transport_tmp, transport, resume_prepared=resumed_prepared, recovery_authority=authority)
        code, submitted = continued.run()
        assert code == 0 and submitted["status"] == "SUBMISSION_BOUND", submitted
        continued_receipt = continued.output / "launch_receipt.json"
        retained_receipt = continued_receipt.read_bytes()
        code, repeated = continued.run()
        assert code == 0 and repeated["status"] == "SUBMISSION_BOUND", repeated
        assert continued_receipt.read_bytes() == retained_receipt
        assert continued.transport().api.created == continued.transport().api.pushed == 1

    resumed_launch = json.loads((resumed_prepared / "source_dataset/experiment_launch.json").read_bytes())
    for arm_id, (context, contract, record, recipe, data, initial, trajectory) in inputs.items():
        bundle = resumed_prepared / "source_dataset" / resumed_launch["resume"][arm_id]["manifest"]
        restore_resume_bundle(bundle, remote / arm_id, spec, context=context, trajectory=trajectory)
        output = SessionOutputs(remote / arm_id, context, contract=contract, adapter=PROFILES[recipe.family],
            trajectory_id=trajectory["trajectory_id"], train_role_identity=record["train_role_identity"],
            resume_trajectory=trajectory)
        configure_fp32_determinism(recipe.seed)
        session = FamilyTrainingSession(FamilyKernel(recipe, initial, device="cpu"),
            FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0, development_workers=0)),
            checkpoint=output.checkpoint, selected=output.selected, pass_finished=output.pass_finished)
        session.batch_finished = output.batch_finished
        session.load_state_dict(output.retained_state())
        assert session.progress.optimizer_step == 2
        started = time.perf_counter()
        try:
            progress = session.run()
            checked = output.complete(progress, runtime={"platform": context.platform, "account": context.account,
                "precision": "fp32", "source_commit": context.source_commit,
                "source_archive_sha256": context.source_archive_sha256}, costs=[{
                    "metric": "wall_seconds", "unit": "seconds", "value": time.perf_counter() - started,
                    "status": "measured", "semantics": "process_wall", "hardware": "synthetic-cpu"}])
            assert checked["status"] == "MECHANICALLY_VERIFIED", checked
            assert progress.optimizer_step == recipe.optimizer_steps
            trace = json.loads((output.root / "canonical_trace.json").read_bytes())
            assert trace["observations"][:len(prefixes[arm_id])] == prefixes[arm_id]
        finally:
            session.close()
        context = RunContext.from_launch(spec, continued_receipt, package,
            expected_package_identity=manifest["package_identity"], arm_id=arm_id)
        assert context.platform_version == "2"
        contexts[arm_id] = context
        decision = case.repo / "experiments/question" / (arm_id + "-disposition.md")
        decision.write_text("Synthetic infrastructure exercise only; no scientific conclusion.\n", encoding="utf-8")
        decisions[arm_id] = decision.relative_to(case.repo).as_posix()
        audit = case.repo / "audits" / (arm_id + ".json")
        atomic_json(audit, {"context": context.to_dict(), "roles": [],
            "role_use": {role: "untouched" for role in ("official_validation", "test_dev", "test_challenge")}})
        audits[arm_id] = audit.relative_to(case.repo).as_posix()

    # A deterministic fixture clock describes a simulated allocation, not measured GPU time.
    clock = [0.]
    ledger = AllocationLedger(spec_identity=spec.identity, hardware=["Tesla T4", "Tesla T4"],
        assignments={job["arm_id"]: job["device"] for job in launch["jobs"]}, started=0., clock=lambda: clock[0])
    clock[0] = 1.
    ledger.write(remote, "complete", arm_roots=[remote / arm for arm in contexts])
    seal_execution_retention(remote, spec, package_identity=manifest["package_identity"],
        source_commit=manifest["source_commit"], source_archive_sha256=manifest["archive_sha256"],
        arm_roots={arm: remote / arm for arm in contexts})
    with monkeypatch.context() as transport:
        retriever = _module(transport)
        for arm_id, context in contexts.items():
            pinned = remote / arm_id / "output_manifest.json"
            value = json.loads(pinned.read_bytes())
            files = {"output_manifest.json", *(item["path"] for item in value["artifacts"].values())}
            prefix = f"runs/kernel77/version2/{arm_id}"
            payloads = {"https://offline.invalid/" + prefix + "/" + name: remote / arm_id / name for name in files}
            api = _Api(context.account, [_ResponsePage([_File(prefix + "/" + name, url)
                for name, url in ((name, "https://offline.invalid/" + prefix + "/" + name) for name in files)])])
            class Http:
                def get(self, url, **kwargs):
                    downloads.append(url)
                    return Response(content=payloads[url].read_bytes())
            transport.setattr(owner, "connect_kaggle", lambda *_: api)
            transport.setattr("requests.Session", Http)
            captured = StringIO()
            args = ["--profile", case.value["machine_profile"], "--repo-root", str(case.repo),
                    "--run-id", run_id, "--attempt-id", attempt, "--arm", arm_id,
                    "--remote-manifest-path", prefix + "/output_manifest.json", "--local-manifest", str(pinned),
                    "--manifest-sha256", sha256_file(pinned), "--destination", str(paths.artifact_root / "retrieved" / arm_id),
                    "--continuation-receipt", str(continued_receipt),
                    "--credentials", str(case.credentials), "--execute"]
            with redirect_stdout(captured):
                code = retriever.main(args)
            assert code == 0, captured.getvalue()
            assert json.loads(captured.getvalue())["status"] == "RETRIEVED"
            downloaded = len(downloads)
            captured = StringIO()
            with redirect_stdout(captured):
                code = retriever.main(args)
            assert code == 0, captured.getvalue()
            assert len(downloads) == downloaded
            # CPU transport exercises cannot produce native GPU qualification.
            # Retain explicit missing-qualification observations at that boundary.
            native = paths.artifact_root / "retrieved/_resume_preflight" / arm_id
            native.mkdir(parents=True)
            for name in ("runtime_certificate.json", "actual_resume_certificate.json"):
                atomic_json(native / name, {"status": "NOT_QUALIFIED", "reason": "Synthetic CPU journey only"})
    options = dict(
        diagnose=True, stage_terminal=True, execute=True, outcome="INCONCLUSIVE",
        decision_ref=decisions, role_audit=audits, execution_root=remote,
        continuation_receipt=continued_receipt, resume_prepared=resumed_prepared, resume_authority=authority)
    result = accept_run_workflow(case.value["machine_profile"], case.repo, run_id, attempt, **options)
    assert result["status"] == "COMPLETE", result
    assert result["diagnosis"]["status"] == "DIAGNOSED", result["diagnosis"]
    assert result["diagnosis"]["reference_arm_id"] == "baseline"
    assert not frozen_differences(case.repo)
    finalizations = {}
    for binding in spec.to_dict()["prospective"]["arms"]:
        final = case.repo / binding["output"] / "rml_finalized"
        assert json.loads((final / "trajectory.json").read_bytes())["decision"]["outcome"] == "INCONCLUSIVE"
        assert (final / "finalization.json").is_file()
        finalizations[final] = {path.relative_to(final).as_posix(): path.read_bytes()
                                for path in final.rglob("*") if path.is_file()}
        assert (case.prepared / "source_dataset/prospective" / binding["arm_id"] / "trajectory.json").read_bytes() == prospective[binding["arm_id"]]
    repeated = accept_run_workflow(case.value["machine_profile"], case.repo, run_id, attempt, **options)
    assert repeated["status"] == "COMPLETE", repeated
    for final, original in finalizations.items():
        assert {path.relative_to(final).as_posix(): path.read_bytes()
                for path in final.rglob("*") if path.is_file()} == original
    return {"status": result["status"], "diagnosis": result["diagnosis"]["status"],
            "downloaded_files": len(downloads), "source_identity": spec.identity,
            "continuation_version": "2", "paused_step": 2, "completed_step": progress.optimizer_step,
            "repeat_downloads": 0, "repeat_finalization_changes": 0,
            "native_qualification": "NOT_QUALIFIED"}


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
    assert code == 1, first
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
    lifecycle = _train_retrieve_and_close_frozen_addon(case, spec, manifest, monkeypatch)
    # An independent call must fully validate again, even if only one sidecar
    # changed after this transaction's successful content checks.
    (case.prepared / "package/SOURCE_COMMIT.txt").write_text("0" * 40 + "\n")
    with pytest.raises(ValueError):
        owner.submit_prepared_workflow(case.prepared, case.credentials, case.output,
            authority_ref=case.value["authority_ref"],
            monitor_paths=case.value["monitor_paths"], execute=False)
    assert case.transport().api.created == case.transport().api.pushed == 1
    return {"status": result["status"], "family": spec.to_dict()["arms"][0]["family"]["name"], "lifecycle": lifecycle,
            "verification_counts": verification_counts, "api_created": case.transport().api.created,
            "api_pushed": case.transport().api.pushed}


def test_real_package_release_post_receipt_reuses_only_exact_content(scientific, tmp_path, record_testsuite_property):
    case = scientific
    import torch
    from molgap.family_training_data import load_training_data
    from molgap.experiment_family_workflow import tensor_digest
    from molgap.family_training_runner import family_arm_bindings
    from test_family_training_data import write_pinned_data
    from test_scientific_pair_input import _pair, _commit
    from test_experiment_workflow import _write_json
    from molgap.common.paths import MachinePaths
    paths = MachinePaths.from_file(case.value["machine_profile"])
    _, selection, _ = write_pinned_data(paths.data_root, train_rows=256, development_rows=3)
    data = load_training_data(paths.data_root, selection)
    recipe_path = case.repo / case.value["arms"][0]["recipe_ref"]
    record = json.loads(recipe_path.read_bytes())
    record.update(dataset=selection.__dict__, target_mean_eV=data.target_mean_eV, target_std_eV=data.target_std_eV)
    record["acceptance_requirements"].update(
        source_idx_sha256=tensor_digest(torch.cat([g.source_idx for g in data.development]), role="source_idx"),
        target_sha256=tensor_digest(torch.cat([g.y for g in data.development]), role="target"))
    _write_json(recipe_path, record)
    data_path = case.repo / case.value["arms"][0]["data_ref"]["spec_ref"]
    declaration = json.loads(data_path.read_bytes())
    base = declaration["arms"][0]
    base["data"]["dataset"]["sha256"] = selection.manifest_sha256
    base["training"]["recipe"]["sha256"] = __import__("hashlib").sha256(recipe_path.read_bytes()).hexdigest()
    from molgap.family_training_runner import _recipe_configuration
    bindings = family_arm_bindings(_recipe_configuration(record))
    base["training"].update({key: bindings[key] for key in ("sampler", "transform", "objective")})
    _write_json(data_path, declaration)
    value, authority, input_path = _pair(case)
    value["shared"].update(training={}, purpose="architecture_comparison")
    model = dict(value["shared"]["model"])
    name = "k1_readout_scale" if model["family"] == "neural_atom_k1" else "gptrans_readout_scale"
    model["addons"] = [name]
    value["arms"][1]["model"] = model
    target_path = case.repo / value["same_run"]["target_manifest_ref"]
    target = json.loads(target_path.read_bytes())
    target["target_encoding"] = "little-endian-float64-contiguous-raw-bytes"
    _write_json(target_path, target)
    # This fixture's unavailable-history case is exercised by prepare; this
    # independent lifecycle fixture needs a self-contained canonical RML corpus.
    (case.repo / "experiments/history/comparison_readiness.json").unlink()
    platform_path = case.repo / value["platform_ref"]
    platform = json.loads(platform_path.read_bytes())
    platform["kaggle"].update(allocation_wall_limit_seconds=32400, cleanup_reserve_seconds=180)
    _write_json(platform_path, platform)
    _commit(case, value, authority, input_path)
    child = r'''
import json, pathlib, sys, torch
from types import SimpleNamespace as NS
repo, experiment, temp, tests = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(repo / "src"))
sys.path.insert(1, str(tests))
import pytest
import test_kaggle_submission as journey
case = NS(repo=repo, path=experiment, value=json.loads(experiment.read_bytes()), tmp=experiment.parents[3])
patch = pytest.MonkeyPatch()
torch.set_num_threads(1)
try:
    release = journey._packaged_release(case, temp, patch)
    summary = journey._assert_real_package_release_post_receipt_reuses_only_exact_content(release, patch)
finally:
    patch.undo()
print(json.dumps(summary))
'''
    completed = subprocess.run([sys.executable, "-I", "-c", child,
        str(case.repo), str(input_path), str(tmp_path), str(Path(__file__).resolve().parent)],
        cwd=case.repo, text=True, capture_output=True, timeout=360)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    observation = json.loads(completed.stdout.splitlines()[-1])
    assert observation["status"] == "SUBMISSION_BOUND"
    assert observation["verification_counts"] == [1, 1, 1, 1, 1]
    assert observation["api_created"] == observation["api_pushed"] == 1
    assert observation["lifecycle"]["continuation_version"] == "2"
    assert observation["lifecycle"]["paused_step"] == 2
    assert observation["lifecycle"]["completed_step"] == 4
    record_testsuite_property("journey_" + observation["family"], json.dumps(observation, sort_keys=True))
    assert observation["lifecycle"]["status"] == "COMPLETE"


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
