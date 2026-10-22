"""Platform state machine with synthetic files and fake APIs; no network/job."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from requests import HTTPError, Timeout

from molgap import kaggle_submission as owner
from molgap.experiment_launch import canonical_json
from molgap.experiment_spec import ExperimentSpec
from molgap.training_reproducibility import atomic_json


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


def _sdk_quota_observation():
    return NS(quota_refresh_time="2030-01-01T00:00:00Z", gpu_quota=NS(
        total_time_allowed=timedelta(hours=48), time_used=timedelta(hours=30),
        time_reserved=timedelta(hours=5.6165)))


def _session_quota_authority(quota_hours=9):
    return {"user_authorized": True, "arms": ["original", "optimized"], "seeds": [42],
        "allocated_t4_hours_ceiling": 18, "kaggle_quota_hours_ceiling": quota_hours,
        "gpu": {"accelerator": "NvidiaTeslaT4", "allocated_devices": 2,
            "wall_ceiling_seconds": 32400}}


@pytest.fixture
def release(tmp_path, monkeypatch):
    prepared, output = tmp_path / "prepared", tmp_path / "operations"
    kernel, staged = prepared / "kernel", prepared / "source_dataset"
    kernel.mkdir(parents=True)
    staged.mkdir()
    (prepared / "package").mkdir()
    root = Path(__file__).resolve().parents[1]
    spec = ExperimentSpec.from_json((root / "experiments/gptrans_infrastructure_t4_100k/experiment_spec.json").read_text())
    (prepared / "package/experiment_spec.json").write_text(spec.to_json())
    manifest = {"package_identity": "a" * 64}
    atomic_json(prepared / "workflow_report.json", {"status": "PREPARED_FOR_PLATFORM", **manifest})
    metadata = {"id": "owner/new-kernel", "is_private": True, "dataset_sources": ["owner/new-source", "owner/fixed-data"]}
    atomic_json(kernel / "kernel-metadata.json", metadata)
    (kernel / "run.py").write_bytes(b"print('synthetic')\n")
    atomic_json(staged / "dataset-metadata.json", {"id": "owner/new-source"})
    atomic_json(staged / "experiment_launch.json", {"account": "owner", "run_reference": metadata["id"], "accelerator": "NvidiaTeslaT4"})
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
    monkeypatch.setattr(owner, "verify_experiment_source_package", lambda _: manifest)
    monkeypatch.setattr(owner, "read_packaged_text", lambda *_: json.dumps({"user_authorized": True,
        "allocated_t4_hours_ceiling": 14, "arms": ["original", "optimized"], "seeds": [42]}))
    monkeypatch.setattr(owner, "_verify_release_report", lambda *_: {})
    import molgap.experiment_preflight as preflight
    monkeypatch.setattr(preflight, "check_release_inputs", lambda *a, **k: {"errors": []})
    monkeypatch.setattr(owner, "build_launch_receipt", lambda *a, **k: {"binding": {"synthetic": True}})
    def reconcile(_spec, _package, raw, **kwargs):
        envelope = json.loads(raw)
        assert envelope["physical_runs"]["value"][0]["run_identity"] == "kernel77/version1"
        assert envelope["canonical_platform_reference"]["value"] == "owner/new-kernel"
        return {"submission_state": "ACCEPTED", "canonical_platform_reference": envelope["canonical_platform_reference"],
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
        owned_kernels = []
        kernel_list_calls = 0

        def get_config_value(self, _name):
            return "owner"

        def dataset_status(self, ref, /):
            if not self.dataset:
                raise Missing()
            return "ready" if self.ready else "processing"

        def dataset_create_new(self, path, **kwargs):
            assert kwargs == dict(public=False, quiet=True, convert_to_csv=False, dir_mode="zip")
            self.created += 1
            self.dataset = True
            return NS(ref="/datasets/owner/new-source", error="")

        def dataset_list_files(self, ref, page_size):
            return NS(files=[NS(name=p.relative_to(staged).as_posix(), total_bytes=p.stat().st_size)
                             for p in staged.rglob("*") if p.is_file() and p.name != "dataset-metadata.json"])

        def kernels_status(self, ref):
            assert ref == "owner/new-kernel/1"
            return NS(status="RUNNING", failure_message="")

        def kernels_list(self, *, mine, user, page, page_size):
            assert (mine, user, page, page_size) == (True, "owner", 1, 20)
            self.kernel_list_calls += 1
            return self.owned_kernels

    api = Api()
    sdk = NS(requests=[], responses=[], client_exits=[])
    signed_payloads = {}

    class SDKResponse:
        status_code = 200

        def __init__(self, url):
            self.url = url
            self.closed = False

        def close(self):
            self.closed = True

    class DatasetAPI:
        def download_dataset(self, request):
            from urllib.parse import quote
            sdk.requests.append(request)
            url = "https://storage.example/" + quote(request.file_name, safe="")
            signed_payloads[url] = (staged / request.file_name).read_bytes()
            response = SDKResponse(url)
            sdk.responses.append(response)
            return response

    class SDKClient:
        datasets = NS(dataset_api_client=DatasetAPI())

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            sdk.client_exits.append(True)

    api.build_kaggle_client = lambda: SDKClient()
    monkeypatch.setattr(owner, "_dataset_details", lambda api, ref: NS(current_version_number=1))
    monkeypatch.setattr(owner, "_kernel_source", lambda _api, _ref: api.remote)
    def accepted():
        api.remote = NS(metadata=NS(id=77, current_version_number=1, is_private=True,
            enable_gpu=True, machine_shape="NvidiaTeslaT4", dataset_data_sources=metadata["dataset_sources"]),
            blob=NS(source=(kernel / "run.py").read_text()))
        return {"status": "submitted", "kernel": metadata["id"], "kernel_id": 77,
                "version_number": 1, "reconciliation_required": False}
    def push(**kwargs):
        api.pushed += 1
        return accepted()
    monkeypatch.setattr(owner, "push_kernel_with_accelerator", push)

    class Http:
        auth = None
        downloads = []
        heads = []
        head_status = 404
        quota_hours = 30

        def head(self, url, **kwargs):
            self.heads.append({"url": url, **kwargs})
            response = Response()
            response.status_code = self.head_status
            return response

        def get(self, url, **kwargs):
            if url.endswith("/quota"):
                seconds = lambda n: {"seconds": n * 3600, "nanos": 0}
                return Response({"gpuQuota": {"totalTimeAllowed": seconds(self.quota_hours),
                    "timeUsed": seconds(0), "timeReserved": seconds(0)}})
            if "/logs/stream/" in url:
                return Response()
            from urllib.parse import unquote
            assert "datasetVersionNumber=1" in url
            name = unquote(url.split("/new-source/", 1)[1].split("?", 1)[0])
            self.downloads.append(name)
            return Response(content=(staged / name).read_bytes())
    http = Http()
    anonymous = NS(sessions=[], gets=[], corrupt=False)

    class AnonymousSession:
        def __init__(self):
            self.auth = None
            self.closed = False
            anonymous.sessions.append(self)

        def get(self, url, **kwargs):
            anonymous.gets.append({"url": url, "session": self, "auth": self.auth,
                "request_auth": kwargs.get("auth"), "request_auth_supplied": "auth" in kwargs})
            content = signed_payloads[url]
            if anonymous.corrupt:
                content = b"corrupt"
            return Response(content=content)

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    import requests
    monkeypatch.setattr(requests, "Session", AnonymousSession)

    def run(execute=True, *, reconcile_retired_push=False):
        return owner.submit_prepared_workflow(prepared, credentials, output, authority_ref="authority.json",
            monitor_paths=["experiments/example/STATUS.md"], execute=execute, api=api, http_session=http,
            reconcile_retired_push=reconcile_retired_push)
    return NS(run=run, api=api, http=http, output=output, staged=staged, kernel=kernel,
              prepared=prepared, accepted=accepted, credentials=credentials, sdk=sdk,
              anonymous=anonymous)


def test_check_has_no_network_or_filesystem_publication(release):
    assert release.run(False)["status"] == "READY_FOR_PLATFORM_OPERATIONS"
    assert not release.output.exists()
    assert release.api.created == release.api.pushed == 0


def test_full_submission_and_repeated_call_never_duplicate(release):
    result = release.run()
    assert result["status"] == "SUBMISSION_BOUND" and result["version"] == 1
    receipt = (release.output / "launch_receipt.json").read_bytes()
    assert release.api.created == release.api.pushed == 1
    expected_names = {p.relative_to(release.staged).as_posix() for p in release.staged.rglob("*")
                      if p.is_file() and p.name != "dataset-metadata.json"}
    assert len(release.sdk.requests) == len(expected_names)
    assert {(r.owner_slug, r.dataset_slug, r.dataset_version_number, r.file_name, r.raw)
            for r in release.sdk.requests} == {
                ("owner", "new-source", 1, name, True) for name in expected_names}
    assert len(release.sdk.responses) == len(expected_names)
    assert all(response.closed for response in release.sdk.responses)
    assert len(release.sdk.client_exits) == len(expected_names)
    assert len(release.anonymous.gets) == len(expected_names)
    assert all(call["session"].auth is None and call["auth"] is None
               and call["request_auth"] is None and not call["request_auth_supplied"]
               for call in release.anonymous.gets)
    assert all(session.closed for session in release.anonymous.sessions)
    assert release.http.auth == ("owner", "do-not-print")
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
        release.api.pushed += 1
        release.accepted()
        return {"status": "submission_unknown", "reconciliation_required": True}
    monkeypatch.setattr(owner, "push_kernel_with_accelerator", lost)
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    assert release.run()["status"] == "SUBMISSION_BOUND"
    assert release.api.pushed == 1


def test_retired_route_http_error_recovers_once_only_after_confirmed_absence(release, monkeypatch):
    legacy_url = "https://www.kaggle.com/api/v1/kernels/push"
    rpc_url = "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel"
    pushes = []

    def retired_route(**kwargs):
        pushes.append("legacy")
        raise HTTPError("retired route")

    monkeypatch.setattr(owner, "push_kernel_with_accelerator", retired_route)
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    state_path = release.output / "submission_state.json"
    prior = json.loads(state_path.read_text())
    assert prior["error_type"] == "HTTPError"
    assert prior["submission_endpoint"] == rpc_url
    prior["submission_endpoint"] = legacy_url  # Journal written by the pre-RPC submitter.
    prior["observed_at"] = (datetime.now(timezone.utc) - timedelta(seconds=61)).isoformat()
    atomic_json(state_path, prior)

    def assert_gate_rejects(fields, error, *, unknown_listing=False):
        candidate = {**prior, **fields}
        atomic_json(state_path, candidate)
        if unknown_listing:
            release.api.owned_kernels = None
        with pytest.raises(error):
            release.run(reconcile_retired_push=True)
        assert len(pushes) == 1
        atomic_json(state_path, prior)
        release.api.owned_kernels = []

    assert_gate_rejects({"error_type": "Timeout"}, ValueError)
    assert_gate_rejects({"observed_at": datetime.now(timezone.utc).isoformat()}, ValueError)
    assert_gate_rejects({"submission_endpoint": rpc_url}, ValueError)
    assert_gate_rejects({"error_type": "HTTPError"}, RuntimeError, unknown_listing=True)
    assert release.http.heads == [{"url": legacy_url, "timeout": (10, 30)}]
    assert release.api.kernel_list_calls == 1
    assert not (release.output / "retired_route_reconciliation.json").exists()

    def rpc_loses_response(**kwargs):
        pushes.append("rpc")
        submitting = json.loads(state_path.read_text())
        assert submitting["phase"] == "KERNEL_SUBMITTING"
        assert submitting["submission_endpoint"] == rpc_url
        assert submitting["error_type"] is None
        assert "retired_route_reconciliation" in submitting
        return {"status": "submission_unknown", "error_type": "Timeout", "reconciliation_required": True}

    monkeypatch.setattr(owner, "push_kernel_with_accelerator", rpc_loses_response)
    assert release.run(reconcile_retired_push=True)["status"] == "RECONCILIATION_REQUIRED"
    after_rpc = json.loads(state_path.read_text())
    assert after_rpc["phase"] == "SUBMIT_UNKNOWN"
    assert after_rpc["error_type"] == "Timeout"
    assert after_rpc["submission_endpoint"] == rpc_url
    reconciliation = json.loads((release.output / "retired_route_reconciliation.json").read_text())
    assert reconciliation["prior"]["error_type"] == "HTTPError"
    assert reconciliation["legacy_route_probe_status"] == 404
    assert reconciliation["exact_target_lookup"] == "ABSENT"
    assert reconciliation["complete_owned_listing"] == "ABSENT"
    assert release.api.created == 1
    assert len(pushes) == 2
    assert len(release.http.heads) == 2
    assert release.api.kernel_list_calls == 2

    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    with pytest.raises(ValueError):
        release.run(reconcile_retired_push=True)
    assert len(pushes) == 2
    assert len(release.http.heads) == 2
    assert release.api.kernel_list_calls == 2

    # Even stale legacy fields cannot authorize a second recovery after evidence was recorded.
    stale = {**after_rpc, "error_type": "HTTPError", "submission_endpoint": legacy_url,
             "observed_at": (datetime.now(timezone.utc) - timedelta(seconds=61)).isoformat()}
    atomic_json(state_path, stale)
    with pytest.raises(ValueError):
        release.run(reconcile_retired_push=True)
    assert len(pushes) == 2
    assert len(release.http.heads) == 2
    assert release.api.created == 1


def test_unobserved_kernel_timeout_never_retries_post(release, monkeypatch):
    def lost(**kwargs):
        release.api.pushed += 1
        raise Timeout("secret-bearing transport error")
    monkeypatch.setattr(owner, "push_kernel_with_accelerator", lost)
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    assert release.run()["status"] == "RECONCILIATION_REQUIRED"
    assert release.api.pushed == 1


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


def test_sdk_quota_uses_total_duration_and_subtracts_reservations():
    http = NS(get=lambda *args, **kwargs: pytest.fail("SDK quota uncertainty must not use REST fallback"))
    result = owner._quota(http, 9, api=NS(quota_view=_sdk_quota_observation))
    quota = result["quota"]["gpuQuota"]
    assert result["provider"] == "KaggleSdk.GetAcceleratorQuotaStatistics"
    assert result["quota_semantics"] == "platform_session_hours"
    assert quota["totalTimeAllowed"]["seconds"] == 48 * 3600
    assert quota["timeReserved"] == {"seconds": 20219, "nanos": 400_000_000}
    assert result["available_gpu_hours"] == pytest.approx(12.3835)
    assert result["authorized_ceiling_gpu_hours"] == 9


def test_sdk_quota_unknown_fails_closed_without_rest_fallback():
    calls = []

    def unknown():
        raise RuntimeError("quota observation unavailable")

    http = NS(get=lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(RuntimeError, match="quota observation unavailable"):
        owner._quota(http, 9, api=NS(quota_view=unknown))
    assert calls == []


def test_explicit_session_quota_binds_nine_wall_hours_and_eighteen_physical(release, monkeypatch):
    authority = _session_quota_authority()
    monkeypatch.setattr(owner, "read_packaged_text", lambda *_: json.dumps(authority))
    observations = []
    release.api.quota_view = lambda: (observations.append(True) or _sdk_quota_observation())
    # A REST response with no remaining quota proves that the SDK view is preferred.
    release.http.quota_hours = 0

    result = release.run()

    assert result["status"] == "SUBMISSION_BOUND"
    assert authority["allocated_t4_hours_ceiling"] == 18
    assert authority["kaggle_quota_hours_ceiling"] == 9
    assert authority["gpu"] == {"accelerator": "NvidiaTeslaT4", "allocated_devices": 2,
        "wall_ceiling_seconds": 32400}
    assert len(observations) == 2 and release.api.pushed == 1
    for name in ("quota_before_publication.json", "quota_before_submit.json"):
        receipt = json.loads((release.output / name).read_bytes())
        assert receipt["authorized_ceiling_gpu_hours"] == 9
        assert receipt["available_gpu_hours"] == pytest.approx(12.3835)


def test_forged_session_quota_authority_fails_before_post(release, monkeypatch):
    authority = _session_quota_authority(quota_hours=10)
    monkeypatch.setattr(owner, "read_packaged_text", lambda *_: json.dumps(authority))

    with pytest.raises(ValueError, match="session quota must bind"):
        release.run()

    assert release.api.created == release.api.pushed == 0
    assert not release.output.exists()


def test_extra_upload_file_is_rejected_before_publication(release):
    (release.staged / "accidental-secret.json").write_text("secret")
    with pytest.raises(ValueError, match="unbound files"):
        release.run()
    assert release.api.created == 0


def test_remote_source_corruption_blocks_kernel_post(release):
    release.anonymous.corrupt = True
    with pytest.raises(ValueError, match="hash"):
        release.run()
    assert release.sdk.requests
    assert all(response.closed for response in release.sdk.responses)
    assert all(call["session"].auth is None and not call["request_auth_supplied"]
               for call in release.anonymous.gets)
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
def test_private_forbidden_requires_complete_authenticated_owned_absence():
    from types import SimpleNamespace
    from molgap.kaggle_submission import _owned_absence
    error = SimpleNamespace(response=SimpleNamespace(status_code=403))
    class API:
        CONFIG_NAME_USER = "user"
        def get_config_value(self, name):
            return "owner"
        def kernels_list(self, **kwargs):
            assert kwargs["mine"] and kwargs["user"] == "owner"
            return [SimpleNamespace(ref="owner/existing")]
        def dataset_list(self, **kwargs):
            assert kwargs["mine"] and "user" not in kwargs
            return []
    api = API()
    assert _owned_absence(api, "owner/missing", "kernel", error)
    assert _owned_absence(api, "owner/missing", "dataset", error)
    assert not _owned_absence(api, "owner/existing", "kernel", error)
    assert not _owned_absence(api, "different/missing", "kernel", error)
    assert not _owned_absence(api, "owner/missing", "kernel", SimpleNamespace(response=SimpleNamespace(status_code=500)))
