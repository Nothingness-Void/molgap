"""Offline tests for the version-pinned Kaggle metadata retrieval boundary."""
from __future__ import annotations

import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from molgap.kaggle_output_retrieval import retrieve_exact_metadata


class _Response:
    status_code = 200

    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def iter_content(self, chunk_size):
        assert chunk_size == 65536
        for offset in range(0, len(self.payload), 5):
            yield self.payload[offset:offset + 5]


class _Http:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def get(self, url, *, stream, timeout):
        assert stream is True
        assert timeout == (30, 60)
        self.calls.append(url)
        return _Response(self.payloads[url])


class _KernelApi:
    def __init__(self, redirect_url):
        self.redirect_url = redirect_url
        self.download_requests = []
        self.list_calls = 0

    def download_kernel_output(self, request):
        self.download_requests.append(request)
        return SimpleNamespace(url=self.redirect_url)

    def list_kernel_session_output(self, request):
        self.list_calls += 1
        raise AssertionError("exact metadata retrieval must not list kernel output")


class _Client:
    def __init__(self, kernel_api):
        self.kernels = SimpleNamespace(kernels_api_client=kernel_api)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Api:
    CONFIG_NAME_USER = "username"

    def __init__(self, client):
        self.username = "owning-account"
        self.client = client
        self.client_builds = 0

    def get_config_value(self, key):
        assert key == self.CONFIG_NAME_USER
        return self.username

    def build_kaggle_client(self):
        self.client_builds += 1
        return self.client


def test_downloads_exact_version_owner_and_path_without_listing(tmp_path, monkeypatch):
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

    class ApiDownloadKernelOutputRequest:
        pass

    service.ApiDownloadKernelOutputRequest = ApiDownloadKernelOutputRequest
    monkeypatch.setitem(sys.modules, names[-1], service)

    path = "gptrans_ema_portability/output_manifest.json"
    payload = json.dumps({"format": "synthetic", "version": 7}).encode()
    url = "https://signed.example/private/output-manifest?token=secret"
    kernel_api = _KernelApi(url)
    api = _Api(_Client(kernel_api))
    http = _Http({url: payload})

    result = retrieve_exact_metadata(
        api,
        account="owning-account",
        kernel="owning-account/gptrans-ema-portability",
        version=7,
        paths=[path],
        destination=tmp_path / "retained",
        http_session=http,
    )

    request = kernel_api.download_requests[0]
    assert request.owner_slug == "owning-account"
    assert request.kernel_slug == "gptrans-ema-portability"
    assert request.version_number == 7
    assert request.file_path == path
    assert kernel_api.list_calls == 0
    assert http.calls == [url]
    assert api.client_builds == 1
    assert result == {
        "kernel": "owning-account/gptrans-ema-portability",
        "version": 7,
        "files": [{"path": path, "sha256": hashlib.sha256(payload).hexdigest()}],
        "transport": "version_specific_exact_file",
        "scientific_acceptance": False,
        "unselected_outputs_downloaded": False,
    }
    assert (tmp_path / "retained" / path).read_bytes() == payload


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 16 * 1024 * 1024 + 1])
def test_invalid_metadata_limit_rejected_before_network(tmp_path, limit):
    api = _Api(_Client(_KernelApi("unused")))
    with pytest.raises(ValueError, match="byte limit"):
        retrieve_exact_metadata(api, account="owning-account",
            kernel="owning-account/job", version=1, paths=["trace.json"],
            destination=tmp_path, max_bytes=limit)
    assert api.client_builds == 0


def test_larger_diagnostic_requires_explicit_bound(tmp_path):
    # The real SDK request type is metadata-only; HTTP and API remain synthetic.
    url = "https://signed.example/private/trace?token=secret"
    payload = json.dumps({"diagnostic": "x" * (1024 * 1024)}).encode()
    api = _Api(_Client(_KernelApi(url)))
    http = _Http({url: payload})
    kwargs = dict(account="owning-account", kernel="owning-account/job",
        version=1, paths=["trace.json"], destination=tmp_path, http_session=http)
    with pytest.raises(RuntimeError, match="no bulk fallback"):
        retrieve_exact_metadata(api, **kwargs)
    assert not (tmp_path / "trace.json").exists()
    result = retrieve_exact_metadata(api, **kwargs, max_bytes=2 * 1024 * 1024)
    assert (tmp_path / "trace.json").read_bytes() == payload
    assert result["scientific_acceptance"] is False
    assert result["files"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert api.client.kernels.kernels_api_client.list_calls == 0
