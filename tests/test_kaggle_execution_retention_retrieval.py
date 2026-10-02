"""Offline selective retrieval tests for the execution retention protocol."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from molgap.experiment_allocation import AllocationLedger
from molgap.experiment_family_workflow import RunContext
from molgap.experiment_launch import canonical_json
from molgap.experiment_retention import (
    LEDGER_FORMAT,
    LEDGER_FILENAME,
    RETENTION_FILENAME,
    seal_execution_retention,
)
from molgap.kaggle_output_retrieval import retrieve_execution_retention
from test_experiment_workflow import _candidate_pair_spec


PACKAGE_ID = "a" * 64
ARCHIVE_ID = "b" * 64
SOURCE_COMMIT = "c" * 40


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _context(spec) -> RunContext:
    return RunContext(
        experiment_id="synthetic-retention",
        logical_run_id="synthetic-retention-run",
        arm_id="k1_candidate",
        arm_identity="d" * 64,
        spec_identity=spec.identity,
        family_name="neural_atom_k1",
        family_version="2",
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=ARCHIVE_ID,
        package_identity=PACKAGE_ID,
        training_recipe_sha256="e" * 64,
        platform="kaggle",
        account="owning-account",
        run_reference="owning-account/retention-kernel",
        platform_version="session-version-9",
    )


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
        for offset in range(0, len(self.payload), 11):
            yield self.payload[offset:offset + 11]


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


@pytest.fixture
def retention_case(tmp_path, monkeypatch):
    # Keep the SDK request object synthetic; retrieval still exercises the real
    # account, pagination, selected-file, hash, and retention validators.
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

    spec = _candidate_pair_spec()
    context = _context(spec)
    root = tmp_path / "execution"
    root.mkdir()
    arm_ids = [arm["arm_id"] for arm in spec.to_dict()["arms"]]
    arm_roots = {arm_id: root / arm_id for arm_id in arm_ids}
    clock = [0.0]
    prior = {"format": LEDGER_FORMAT, "spec_identity": spec.identity, "status": "running"}
    ledger = AllocationLedger(
        spec_identity=spec.identity,
        hardware=["Tesla T4", "Tesla T4"],
        assignments={arm_ids[0]: 0, arm_ids[1]: 1},
        started=0.0,
        prior_segments=[prior],
        clock=lambda: clock[0],
    )
    clock[0] = 12.5
    ledger.write(root, "complete", arm_roots=list(arm_roots.values()))
    (root / "execution_report.json").write_bytes(
        canonical_json({"format": "molgap-kaggle-two-phase-pair-v2", "status": "complete"}).encode()
    )
    sealed = seal_execution_retention(
        root, spec, package_identity=PACKAGE_ID, source_commit=SOURCE_COMMIT,
        source_archive_sha256=ARCHIVE_ID, arm_roots=arm_roots,
    )

    remote_manifest_path = "runs/session-9/execution_retention.json"
    prefix = Path("runs/session-9")
    local_manifest = tmp_path / "pinned-execution-retention.json"
    local_manifest.write_bytes((root / RETENTION_FILENAME).read_bytes())
    files = {}
    payloads = {}
    for relative in [
        RETENTION_FILENAME,
        LEDGER_FILENAME,
        *(f"{arm_id}/{LEDGER_FILENAME}" for arm_id in arm_ids),
        "execution_report.json",
    ]:
        local = root / relative
        remote = (prefix / relative).as_posix()
        url = f"https://signed.example/private/{relative.replace('/', '-')}?token=retention-secret"
        files[remote] = _File(remote, url)
        payloads[url] = local.read_bytes()
    extra_remote = (prefix / "raw/unselected-debug-dump.bin").as_posix()
    extra_url = "https://signed.example/private/extra?token=must-not-download"
    files[extra_remote] = _File(extra_remote, extra_url)
    payloads[extra_url] = b"unselected"
    entries = list(files.values())
    pages = [_ResponsePage(entries[:3], next_page_token="more"),
             _ResponsePage(entries[3:], next_page_token=None)]
    api = _Api("owning-account", pages)
    http = _Http(payloads)
    return {
        "spec": spec,
        "context": context,
        "root": root,
        "arm_ids": arm_ids,
        "remote_manifest_path": remote_manifest_path,
        "local_manifest": local_manifest,
        "manifest_sha256": _sha(local_manifest.read_bytes()),
        "destination": tmp_path / "retained",
        "api": api,
        "http": http,
        "extra_url": extra_url,
        "sealed": sealed,
    }


def _run(case, **kwargs):
    args = {
        "context": case["context"],
        "spec": case["spec"],
        "remote_manifest_path": case["remote_manifest_path"],
        "local_manifest": case["local_manifest"],
        "manifest_sha256": case["manifest_sha256"],
        "destination": case["destination"],
        "http_session": case["http"],
    }
    args.update(kwargs)
    return retrieve_execution_retention(case["api"], **args)


def test_retrieves_only_fixed_retention_files_and_validates_bytes(retention_case):
    case = retention_case
    result = _run(case)

    expected = {
        RETENTION_FILENAME,
        LEDGER_FILENAME,
        *(f"{arm_id}/{LEDGER_FILENAME}" for arm_id in case["arm_ids"]),
        "execution_report.json",
    }
    assert result["status"] == "EXECUTION_RETENTION_RETRIEVED"
    assert result["execution_retention"]["status"] == "EXECUTION_RETENTION_VERIFIED"
    assert {item["path"] for item in result["files"]} == expected
    assert len(case["http"].urls) == len(expected)
    assert case["extra_url"] not in case["http"].urls
    assert (case["destination"] / RETENTION_FILENAME).is_file()
    assert (case["destination"] / LEDGER_FILENAME).is_file()
    assert (case["destination"] / "execution_report.json").is_file()


def test_retention_identity_mismatch_fails_before_remote_listing(retention_case):
    case = retention_case
    case["context"] = replace(case["context"], package_identity="f" * 64)

    with pytest.raises(ValueError, match="identity mismatch"):
        _run(case)

    assert case["api"].client_builds == 0
    assert case["http"].urls == []
    assert not case["destination"].exists()


def test_retention_manifest_cannot_rebind_ledger_path(retention_case):
    case = retention_case
    changed = json.loads(case["local_manifest"].read_text(encoding="utf-8"))
    changed["allocation_ledger"]["root"]["path"] = "raw/unselected-debug-dump.bin"
    raw = canonical_json(changed).encode()
    case["local_manifest"].write_bytes(raw)
    case["manifest_sha256"] = _sha(raw)

    with pytest.raises(ValueError, match="root ledger binding is invalid"):
        _run(case)

    assert case["api"].client_builds == 0
    assert case["http"].urls == []


def test_corrupt_selected_ledger_is_not_published(retention_case):
    case = retention_case
    arm_id = case["arm_ids"][0]
    remote = f"runs/session-9/{arm_id}/{LEDGER_FILENAME}"
    url = case["api"].client.kernels.pages[0].files[0].url
    for item in case["api"].client.kernels.pages[0].files + case["api"].client.kernels.pages[1].files:
        if item.file_name == remote:
            url = item.url
            break
    case["http"].payloads[url] = b"corrupt ledger bytes"

    with pytest.raises(ValueError, match="hash disagrees with pinned attempt"):
        _run(case)

    assert not (case["destination"] / arm_id / LEDGER_FILENAME).exists()
