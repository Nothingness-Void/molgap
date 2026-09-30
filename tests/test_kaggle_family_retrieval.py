"""Offline tests for selective Kaggle family-output retrieval."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from molgap.experiment_family_workflow import ROLES, RunContext


_REPO = Path(__file__).resolve().parents[1]
_RETRIEVER_PATH = _REPO / "platforms" / "kaggle" / "retrieve_family_outputs.py"
_ROLE_ORDER = ("predictions", "selected_model", "resume", "trace", "contract")


def _context() -> RunContext:
    return RunContext(
        experiment_id="synthetic-family",
        logical_run_id="synthetic-run",
        arm_id="gptrans-reference",
        arm_identity="a" * 64,
        spec_identity="b" * 64,
        family_name="gptrans_t",
        family_version="1",
        source_commit="c" * 40,
        source_archive_sha256="d" * 64,
        package_identity="e" * 64,
        training_recipe_sha256="f" * 64,
        platform="kaggle",
        account="owning-account",
        run_reference="owning-account/family-kernel",
        platform_version="session-version-7",
    )


def _module(monkeypatch):
    # The retriever imports this request type lazily. Keep these tests independent
    # of an installed Kaggle SDK and never construct a real API client.
    names = (
        "kagglesdk",
        "kagglesdk.kernels",
        "kagglesdk.kernels.types",
        "kagglesdk.kernels.types.kernels_api_service",
    )
    for name in names[:-1]:
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    service = ModuleType(names[-1])

    class ApiListKernelSessionOutputRequest:
        pass

    service.ApiListKernelSessionOutputRequest = ApiListKernelSessionOutputRequest
    monkeypatch.setitem(sys.modules, names[-1], service)

    module_name = "_molgap_test_retrieve_family_outputs"
    spec = importlib.util.spec_from_file_location(module_name, _RETRIEVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _ResponsePage:
    def __init__(self, files, next_page_token=None):
        self.files = files
        self.next_page_token = next_page_token


class _File:
    def __init__(self, file_name, url):
        self.file_name = file_name
        self.url = url


class _KernelsClient:
    def __init__(self, pages):
        self.pages = list(pages)
        self.requests = []
        self.kernels_api_client = self

    def list_kernel_session_output(self, request):
        self.requests.append(request)
        return self.pages[len(self.requests) - 1]


class _Client:
    def __init__(self, pages):
        self.kernels = _KernelsClient(pages)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Api:
    CONFIG_NAME_USER = "username"

    def __init__(self, username, pages):
        self.username = username
        self.client = _Client(pages)
        self.client_builds = 0

    def get_config_value(self, key):
        assert key == self.CONFIG_NAME_USER
        return self.username

    def build_kaggle_client(self):
        self.client_builds += 1
        return self.client


class _HttpResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def iter_content(self, chunk_size):
        assert chunk_size == 1024 * 1024
        for offset in range(0, len(self.payload), 7):
            yield self.payload[offset : offset + 7]


class _Http:
    def __init__(self, payloads):
        self.payloads = payloads
        self.urls = []

    def get(self, url, *, stream, timeout):
        assert stream is True
        assert timeout == (30, 60)
        self.urls.append(url)
        payload = self.payloads[url]
        if isinstance(payload, Exception):
            raise payload
        return _HttpResponse(payload)


def _case(tmp_path, monkeypatch, *, username="owning-account"):
    retrieve = _module(monkeypatch)
    context = _context()
    remote_manifest = "runs/session-7/output_manifest.json"
    files = {}
    payloads = {}
    artifacts = {}

    for role in _ROLE_ORDER:
        assert role in ROLES
        relative = f"nested/{role}/artifact.bin"
        payload = f"synthetic-{role}-bytes".encode()
        artifacts[role] = {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        remote_path = f"runs/session-7/{relative}"
        url = f"https://signed.example/private/{role}?token=secret-{role}"
        files[remote_path] = _File(remote_path, url)
        payloads[url] = payload

    manifest = {
        "format": "molgap-family-output-v1",
        "context": context.to_dict(),
        "adapter": "gptrans-v1",
        "artifacts": artifacts,
        "progress": {},
        "runtime": {},
        "costs": [],
    }
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    local_manifest = tmp_path / "pinned-output-manifest.json"
    local_manifest.write_bytes(manifest_bytes)
    manifest_url = "https://signed.example/private/manifest?token=secret-manifest"
    files[remote_manifest] = _File(remote_manifest, manifest_url)
    payloads[manifest_url] = manifest_bytes

    extra_path = "runs/session-7/raw/unselected-debug-dump.bin"
    extra_url = "https://signed.example/private/extra?token=secret-extra"
    files[extra_path] = _File(extra_path, extra_url)
    payloads[extra_url] = b"must not be retrieved"

    file_values = list(files.values())
    pages = [
        _ResponsePage(file_values[:3], next_page_token="next-page"),
        _ResponsePage(file_values[3:], next_page_token=None),
    ]
    api = _Api(username, pages)
    http = _Http(payloads)
    return {
        "retrieve": retrieve,
        "context": context,
        "remote_manifest": remote_manifest,
        "local_manifest": local_manifest,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "destination": tmp_path / "retained",
        "artifacts": artifacts,
        "files": files,
        "payloads": payloads,
        "api": api,
        "http": http,
        "extra_url": extra_url,
        "manifest_url": manifest_url,
    }


def _run(case, *, username=None, manifest_sha256=None):
    return case["retrieve"].retrieve_family_outputs(
        case["api"],
        context=case["context"],
        remote_manifest_path=case["remote_manifest"],
        local_manifest=case["local_manifest"],
        manifest_sha256=manifest_sha256 or case["manifest_sha256"],
        destination=case["destination"],
        http_session=case["http"],
    )


def test_downloads_only_five_manifest_artifacts_and_manifest_with_full_paths(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)

    result = _run(case)

    expected_paths = {entry["path"] for entry in case["artifacts"].values()}
    expected_paths.add("output_manifest.json")
    assert {entry["path"] for entry in result["files"]} == expected_paths
    assert len(case["http"].urls) == 6
    assert case["extra_url"] not in case["http"].urls
    assert result["unselected_outputs_downloaded"] is False
    assert result["kernel"] == case["context"].run_reference
    assert [request.page_token for request in case["api"].client.kernels.requests] == [
        None,
        "next-page",
    ]
    assert all(
        request.user_name == "owning-account"
        and request.kernel_slug == "family-kernel"
        and request.page_size == 100
        for request in case["api"].client.kernels.requests
    )
    for role, entry in case["artifacts"].items():
        retained = case["destination"] / entry["path"]
        assert retained.read_bytes() == f"synthetic-{role}-bytes".encode()
    assert (
        case["destination"] / "output_manifest.json"
    ).read_bytes() == case["local_manifest"].read_bytes()


def test_wrong_authenticated_account_publishes_nothing(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, username="different-account")

    with pytest.raises(ValueError, match="Authenticated account"):
        _run(case)

    assert not case["destination"].exists()
    assert case["api"].client_builds == 0
    assert case["http"].urls == []


def test_wrong_pinned_manifest_hash_publishes_nothing(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="Pinned remote manifest hash mismatch"):
        _run(case, manifest_sha256="0" * 64)

    assert not case["destination"].exists()
    assert case["api"].client_builds == 0
    assert case["http"].urls == []


def test_download_hash_mismatch_does_not_publish_the_bad_file(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    # Trace is last in canonical role order, so this exercises a late hash failure
    # after earlier valid files may have been retained for an idempotent retry.
    relative = case["artifacts"]["trace"]["path"]
    remote_path = f"runs/session-7/{relative}"
    url = case["files"][remote_path].url
    case["http"].payloads[url] = b"corrupt bytes"

    with pytest.raises(ValueError, match="hash disagrees with pinned attempt"):
        _run(case)

    assert not (case["destination"] / relative).exists()
    assert not list((case["destination"] / Path(relative).parent).glob("tmp*"))


def test_identical_retained_files_retry_without_http_downloads(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    for role, entry in case["artifacts"].items():
        target = case["destination"] / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"synthetic-{role}-bytes".encode())
    case["destination"].mkdir(parents=True, exist_ok=True)
    (case["destination"] / "output_manifest.json").write_bytes(
        case["local_manifest"].read_bytes()
    )

    result = _run(case)

    assert case["http"].urls == []
    assert len(result["files"]) == 6
    assert {entry["state"] for entry in result["files"]} == {"ALREADY_RETAINED"}


def test_conflicting_retained_file_is_never_overwritten(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    relative = case["artifacts"]["predictions"]["path"]
    target = case["destination"] / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"local conflicting content")

    with pytest.raises(ValueError, match="Retained output conflicts"):
        _run(case)

    assert target.read_bytes() == b"local conflicting content"
    assert case["files"][f"runs/session-7/{relative}"].url not in case["http"].urls


@pytest.mark.parametrize("exception_type", [RuntimeError, ValueError])
def test_signed_url_is_redacted_from_download_error(
    tmp_path, monkeypatch, exception_type
):
    case = _case(tmp_path, monkeypatch)
    remote_path = f"runs/session-7/{case['artifacts']['predictions']['path']}"
    signed_url = case["files"][remote_path].url
    case["http"].payloads[signed_url] = exception_type(
        f"request failed for {signed_url}"
    )

    with pytest.raises(RuntimeError, match="Selected output retrieval failed") as error:
        _run(case)

    assert signed_url not in str(error.value)
    assert "secret-predictions" not in repr(error.value)
    assert not (
        case["destination"] / case["artifacts"]["predictions"]["path"]
    ).exists()
