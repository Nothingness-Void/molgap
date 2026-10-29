"""Platform state machine with synthetic files and fake APIs; no network/job."""
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
from requests import HTTPError, Timeout

from molgap import kaggle_submission as owner
from molgap.experiment_launch import canonical_json
from molgap.experiment_spec import ExperimentSpec
from molgap.training_reproducibility import atomic_json
from test_scientific_input import scientific


class Missing(Exception):
    status = 404


class Denied(Exception):
    response = NS(status_code=403, headers=None)


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
def release(tmp_path, monkeypatch, request, staged_source=None):
    continuation = getattr(request, "param", False)
    version = 2 if continuation else 1
    prepared, output = tmp_path / "prepared", tmp_path / "operations"
    kernel, staged = prepared / "kernel", prepared / "source_dataset"
    kernel.mkdir(parents=True)
    (prepared / "package").mkdir()
    root = Path(__file__).resolve().parents[1]
    spec = ExperimentSpec.from_json((root / "experiments/gptrans_infrastructure_t4_100k/experiment_spec.json").read_text())
    (prepared / "package/experiment_spec.json").write_text(spec.to_json())
    manifest = {"package_identity": "a" * 64}
    atomic_json(prepared / "workflow_report.json", {"status": "PREPARED_FOR_PLATFORM", **manifest})
    if staged_source is None:
        metadata = {"id": "owner/new-kernel", "is_private": True,
                    "dataset_sources": ["owner/new-source", "owner/fixed-data"]}
        dataset_ref = "owner/new-source"
    else:
        staged_source = Path(staged_source)
        metadata = json.loads((staged_source.parent / "kernel/kernel-metadata.json").read_bytes())
        dataset_ref = json.loads((staged_source / "dataset-metadata.json").read_bytes())["id"]
    atomic_json(kernel / "kernel-metadata.json", metadata)
    (kernel / "run.py").write_bytes(b"print('synthetic')\n")
    if staged_source is None:
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
    else:
        shutil.copytree(staged_source, staged)
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
        assert envelope["physical_runs"]["value"][0]["run_identity"] == "kernel77/version1"
        assert envelope["canonical_platform_reference"]["value"] == "owner/new-kernel"
        return {"submission_state": "ACCEPTED", "binding": envelope["binding"],
                "canonical_platform_reference": envelope["canonical_platform_reference"],
                "platform_version": envelope["platform_version"], "physical_runs": envelope["physical_runs"]}
    monkeypatch.setattr(owner, "reconcile_platform_response", reconcile)
    import molgap.experiment_launch as launch
    monkeypatch.setattr(launch, "read_launch_receipt", lambda path, *a, **k: json.loads(Path(path).read_bytes()))

    class Api:
        CONFIG_NAME_USER = "user"
        created = 0
        pushed = 0
        dataset = False
        ready = True
        remote = None
        owned_kernels = []
        kernel_list_calls = 0

        dataset_version = 1
        predecessor_status = "RUNNING"
        denied = 0  # upcoming dataset_status calls that answer 403

        def __init__(self):
            self.slept = []

        def get_config_value(self, _name):
            return metadata["id"].split("/", 1)[0]

        def dataset_status(self, ref):
            if not self.dataset:
                raise Missing()
            if self.denied > 0:
                self.denied -= 1
                raise Denied()
            return "ready" if self.ready else "processing"

        def dataset_list(self, mine, page):
            return [NS(ref=dataset_ref)]

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

        def kernels_list(self, *, mine, user, page, page_size):
            assert (mine, user, page, page_size) == (True, "owner", 1, 20)
            self.kernel_list_calls += 1
            return self.owned_kernels

    api = Api()
    clock = [0.0]
    monkeypatch.setattr(owner, "_poll_clock", lambda: clock[0])
    monkeypatch.setattr(owner, "_poll_sleep", lambda seconds: (
        api.slept.append(seconds), clock.__setitem__(0, clock[0] + seconds)))
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
    monkeypatch.setattr(owner, "_dataset_details", lambda api, ref:
                        NS(ref=ref, is_private=True, current_version_number=api.dataset_version))
    monkeypatch.setattr(owner, "_kernel_source", lambda _api, _ref: api.remote)
    def accepted():
        api.remote = NS(metadata=NS(id=77, current_version_number=version, is_private=True,
            enable_gpu=True, machine_shape="NvidiaTeslaT4", dataset_data_sources=metadata["dataset_sources"]),
            blob=NS(source=(kernel / "run.py").read_text()))
        return {"status": "submitted", "kernel": metadata["id"], "kernel_id": 77,
                "version_number": version, "reconciliation_required": False}
    def push(**kwargs):
        _dispatch_before_post(kwargs)
        api.pushed += 1
        return accepted()
    monkeypatch.setattr(owner, "push_kernel_with_accelerator", push)

    class Http:
        auth = None
        downloads = []
        heads = []
        head_status = 404
        quota_hours = 30

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

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
            if url.startswith("https://storage.example/"):
                return Response(content=signed_payloads[url])
            from urllib.parse import unquote
            assert f"datasetVersionNumber={version}" in url
            dataset_path = "/datasets/download/" + dataset_ref + "/"
            name = unquote(url.split(dataset_path, 1)[1].split("?", 1)[0])
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

    if continuation:
        accepted()
        api.remote.metadata.current_version_number = version - 1
        api.dataset = True

    def run(execute=True, *, reconcile_retired_push=False, reconcile_sdk_transport=False):
        return owner.submit_prepared_workflow(prepared, credentials, output, authority_ref="authority.json",
            monitor_paths=["experiments/example/STATUS.md"], execute=execute, api=api, http_session=http,
            reconcile_retired_push=reconcile_retired_push, reconcile_sdk_transport=reconcile_sdk_transport,
            **recovery)
    return NS(run=run, api=api, http=http, output=output, staged=staged, kernel=kernel,
              prepared=prepared, accepted=accepted, credentials=credentials, sdk=sdk,
              anonymous=anonymous)


def test_check_has_no_network_or_filesystem_publication(release):
    assert release.run(False)["status"] == "READY_FOR_PLATFORM_OPERATIONS"
    assert not release.output.exists()
    assert release.api.created == release.api.pushed == 0


def test_full_submission_and_repeated_call_never_duplicate(scientific, tmp_path, record_testsuite_property):
    """Run the real K1/GPTrans CPU journey through submission and closure."""
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
    summary = journey._assert_real_package_release_post_receipt(release, patch)
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
    assert observation["api_created"] == observation["api_pushed"] == 1
    assert observation["lifecycle"]["continuation_version"] == "2"
    assert observation["lifecycle"]["paused_step"] == 2
    assert observation["lifecycle"]["completed_step"] == 4
    record_testsuite_property("journey_" + observation["family"], json.dumps(observation, sort_keys=True))
    assert observation["lifecycle"]["status"] == "COMPLETE"


@pytest.mark.parametrize("forbidden", [False, True], ids=["processing", "forbidden"])
def test_source_processing_resumes_without_second_create(release, forbidden):
    release.api.ready = False
    release.api.denied = float("inf") if forbidden else 0
    assert release.run()["status"] == ("RECONCILIATION_REQUIRED" if forbidden else "WAIT_SOURCE_READY")
    assert release.api.created == 1 and release.api.pushed == 0
    assert sum(release.api.slept) == owner.SOURCE_POLL_BUDGET_SECONDS
    assert json.loads((release.output / "submission_state.json").read_bytes())["phase"] == "SOURCE_PENDING"
    release.api.ready, release.api.denied = True, 0
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


def test_retired_route_http_error_recovers_once_only_after_confirmed_absence(release, monkeypatch):
    legacy_url = "https://www.kaggle.com/api/v1/kernels/push"
    rpc_url = "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel"
    pushes = []

    def retired_route(**kwargs):
        _dispatch_before_post(kwargs)
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
        _dispatch_before_post(kwargs)
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
    # The push owner returns a structured submission-unknown response for a
    # lost transport; the journal keeps its pre-dispatch error slot empty,
    # while the durable platform response retains the concrete error type.
    assert after_rpc["error_type"] is None
    assert after_rpc["submission_endpoint"] == rpc_url
    platform_response = json.loads((release.output / "platform_response.json").read_text())
    assert platform_response["error_type"] == "Timeout"
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
    monkeypatch.setattr(owner, "read_packaged_text", lambda *_, **__: json.dumps(authority))
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
    monkeypatch.setattr(owner, "read_packaged_text", lambda *_, **__: json.dumps(authority))

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


@pytest.mark.parametrize("source_version", [1, 2])
def test_parallel_transport_starts_after_all_parent_directories_exist(release, monkeypatch, source_version):
    import molgap.kaggle_output_retrieval as transport
    real_download = transport._download_selected
    def checked(**kwargs):
        root = kwargs["destination"]
        assert all((root / path).is_dir() for path in
                   ("initial_states", "prospective/original", "prospective/optimized"))
        return real_download(**kwargs)
    monkeypatch.setattr(transport, "_download_selected", checked)
    if source_version == 1:
        assert release.run()["status"] == "SUBMISSION_BOUND"
    else:
        release.api.dataset = release.api.ready = True
        monkeypatch.setattr(owner, "_dataset_details", lambda api, ref: NS(current_version_number=2))
        result = owner._roundtrip_source(release.api, release.http, release.staged,
            release.output / "resume_roundtrip", "owner/new-source", expected_version=2)
        assert result["version"] == 2
        assert all(request.dataset_version_number == 2 for request in release.sdk.requests)
        assert release.api.created == release.api.pushed == 0
        monkeypatch.setattr(owner, "_dataset_details", lambda api, ref: NS(current_version_number=3))
        with pytest.raises(ValueError, match="pinned immutable version"):
            owner._roundtrip_source(release.api, release.http, release.staged,
                release.output / "drift", "owner/new-source", expected_version=2)


def test_unknown_remote_identity_is_not_absence(release, monkeypatch):
    def unknown(*args):
        raise RuntimeError("API uncertainty")
    monkeypatch.setattr(owner, "_kernel_source", unknown)
    with pytest.raises(RuntimeError):
        release.run()
    assert release.api.created == release.api.pushed == 0


def test_remote_version_drift_blocks_binding_and_never_reposts(release):
    assert release.run()["status"] == "SUBMISSION_BOUND"
    receipt = (release.output / "launch_receipt.json").read_bytes()
    release.api.remote.metadata.current_version_number = 2
    # A bound receipt is immutable and is revalidated locally; a newer remote
    # version must not trigger a remote read or a second publication.
    assert release.run()["status"] == "SUBMISSION_BOUND"
    assert (release.output / "launch_receipt.json").read_bytes() == receipt
    assert release.api.pushed == 1


def test_other_process_lock_blocks_all_mutation(release):
    release.output.mkdir()
    (release.output / ".submission.lock").write_text("another process")
    with pytest.raises(ValueError, match="lock exists"):
        release.run()
    assert release.api.created == release.api.pushed == 0


def test_frozen_authority_cannot_expand_arms_or_seeds(release, monkeypatch):
    monkeypatch.setattr(owner, "read_packaged_text", lambda *_, **__: json.dumps({"user_authorized": True,
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


def _packaged_release(case, tmp_path, monkeypatch, *, resume_prepared=None, recovery_authority=None):
    """Run a prepared scientific input through the platform entry offline."""
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
        # The production Server path supplies its authenticated SDK client to
        # the submission owner.  This wrapper intentionally removes that one
        # argument so the test can observe the existing HTTP RPC boundary;
        # source roundtrips still exercise the SDK transport above.
        return real_push(**{**kwargs, "api": None})

    monkeypatch.setattr(owner, "push_kernel_with_accelerator", mutate_roundtrip_before_recheck)

    def post(url, *, auth, json, timeout):
        current = transport()
        metadata = json_module.loads((prepared / "kernel/kernel-metadata.json").read_bytes())
        assert url == "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel"
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
    """Pause actual CPU training, continue it through v2, retrieve, and close RML."""
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
            FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0, development_workers=0),
                role_audit=output.role_audit),
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
            FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0, development_workers=0),
                role_audit=output.role_audit),
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
        service = sys.modules["kagglesdk.kernels.types.kernels_api_service"]

        class ApiDownloadKernelOutputRequest:
            pass

        service.ApiDownloadKernelOutputRequest = ApiDownloadKernelOutputRequest
        for arm_id, context in contexts.items():
            pinned = remote / arm_id / "output_manifest.json"
            value = json.loads(pinned.read_bytes())
            files = {"output_manifest.json", *(item["path"] for item in value["artifacts"].values())}
            prefix = f"runs/kernel77/version2/{arm_id}"
            payloads = {"https://offline.invalid/" + prefix + "/" + name: remote / arm_id / name for name in files}
            api = _Api(context.account, [_ResponsePage([_File(prefix + "/" + name, url)
                for name, url in ((name, "https://offline.invalid/" + prefix + "/" + name) for name in files)])])
            api.client.kernels.kernels_api_client.download_kernel_output = (
                lambda request: NS(url="https://offline.invalid/" + request.file_path))
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
            native = paths.artifact_root / "retrieved/_resume_preflight" / arm_id
            native.mkdir(parents=True)
            for name in ("runtime_certificate.json", "actual_resume_certificate.json"):
                atomic_json(native / name, {"status": "NOT_QUALIFIED", "reason": "Synthetic CPU journey only"})
    options = dict(diagnose=True, stage_terminal=True, execute=True, outcome="INCONCLUSIVE",
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


def _assert_real_package_release_post_receipt(case, monkeypatch):
    """Exercise exact receipt reuse and the complete continuation lifecycle.

    This intentionally does not instrument the Desktop-only package content
    cache.  The Server path must still retain the durable receipt, avoid a
    second POST/download, and leave terminal RML closure byte-stable.
    """
    from molgap.experiment_launch import read_launch_receipt
    from molgap.experiment_spec import ExperimentSpec

    code, first = case.run()
    assert code == 1, first
    assert first["status"] == "BLOCKED", first
    assert first["stage"] == "release_recheck", json.dumps(first, indent=2)
    assert first["submitted"] is False
    assert first["release_recheck"]["category"] in {"binding_mismatch", "check_failed"}
    assert "workflow" in first["release_recheck"]["changed_checks"]
    state = json.loads((case.output / "submission_state.json").read_bytes())
    assert state["phase"] == "SOURCE_PUBLISHED"
    assert state["release_recheck"] == first["release_recheck"]
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

    case.restore_roundtrip()
    case.lose_next_post()
    code, lost = case.run()
    assert code == 1
    assert lost["status"] == "RECONCILIATION_REQUIRED", lost
    assert lost["submitted"] == "UNKNOWN"
    assert not (case.output / "launch_receipt.json").exists()
    assert case.transport().api.created == case.transport().api.pushed == 1
    code, result = case.run()
    assert code == 0
    assert result["status"] == "SUBMISSION_BOUND", result
    assert case.transport().api.created == case.transport().api.pushed == 1

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
    code, repeated = case.run()
    assert code == 0
    assert repeated["status"] == "SUBMISSION_BOUND"
    assert receipt_path.read_bytes() == retained
    assert case.transport().api.created == case.transport().api.pushed == 1

    lifecycle = _train_retrieve_and_close_frozen_addon(case, spec, manifest, monkeypatch)
    (case.prepared / "package/SOURCE_COMMIT.txt").write_text("0" * 40 + "\n")
    with pytest.raises(ValueError):
        owner.submit_prepared_workflow(case.prepared, case.credentials, case.output,
            authority_ref=case.value["authority_ref"], monitor_paths=case.value["monitor_paths"], execute=False)
    assert case.transport().api.created == case.transport().api.pushed == 1
    return {"status": result["status"], "family": spec.to_dict()["arms"][0]["family"]["name"],
            "lifecycle": lifecycle, "api_created": case.transport().api.created,
            "api_pushed": case.transport().api.pushed}
