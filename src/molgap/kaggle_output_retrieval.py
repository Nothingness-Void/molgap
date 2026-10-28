"""Shared bounded retrieval of pinned Kaggle output files.

The platform adapters decide which files are allowed by their own manifest
validators.  This module owns only the transport boundary: authenticate the
exact account, list one kernel's output pages, and stream the selected files
with atomic publication and hash checks.  It never falls back to a bulk
download and never returns signed URLs in an error or result.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any

from .common.transport import download_pinned_files as _download_selected
from .experiment_launch import _safe_local
from .experiment_terminal import _path
from .research_memory.paths import repo_local_path
from .research_memory.trace import file_digest


def _digest(value: Any, label: str) -> str:
    from .experiment_spec import _digest as validate_digest

    validate_digest(value, label)
    return value


def retrieve_exact_metadata(api, *, account: str, kernel: str, version: int,
                            paths: list[str], destination: Path, http_session=None,
                            max_bytes: int = 1024 * 1024) -> dict:
    """Pin small version-specific outputs without listing worker environments.

    Hashes here describe retrieved bytes, not scientific acceptance. The caller
    reconciles the actual attempt and then validates the owning source/manifest.
    Larger scheduled diagnostic JSON may explicitly raise the per-file bound;
    the default remains 1 MiB and no caller can exceed 16 MiB.
    """
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    from .research_memory.trace import atomic_write
    if type(max_bytes) is not int or not 0 < max_bytes <= 16 * 1024 * 1024:
        raise ValueError("Exact metadata byte limit must be within 1..16 MiB")
    if (api.get_config_value(api.CONFIG_NAME_USER) != account
        or kernel.count("/") != 1 or kernel.split("/")[0] != account
        or type(version) is not int or version < 1
        or not paths or len(paths) != len(set(paths)) or len(paths) > 32):
        raise ValueError("Exact output account/version/allowlist differs")
    for path in paths:
        _path(path)
        if PurePosixPath(path).suffix != ".json":
            raise ValueError("Exact metadata retrieval permits JSON only")
    if http_session is None:
        import requests
        http_session = requests.Session()
    root = Path(destination).absolute()
    _safe_local(root)
    root.mkdir(parents=True, exist_ok=True)
    retained = []
    with api.build_kaggle_client() as client:
        for path in paths:
            target = repo_local_path(root, path)
            request = ApiDownloadKernelOutputRequest()
            request.owner_slug, request.kernel_slug = account, kernel.split("/")[1]
            request.version_number, request.file_path = version, path
            phase, status = "redirect", None
            try:
                redirect = client.kernels.kernels_api_client.download_kernel_output(request)
                phase = "http"
                with http_session.get(redirect.url, stream=True, timeout=(30, 60)) as response:
                    status = response.status_code
                    if response.status_code != 200:
                        raise RuntimeError("Exact metadata response was not successful")
                    phase = "body"
                    data = bytearray()
                    for chunk in response.iter_content(65536):
                        data.extend(chunk)
                        if len(data) > max_bytes:
                            phase = "byte_limit"
                            raise ValueError("Exact metadata exceeds its explicit byte limit")
            except Exception as error:
                if status is None:
                    observed = getattr(error, "status_code", None)
                    if observed is None:
                        observed = getattr(getattr(error, "response", None), "status_code", None)
                    status = observed if type(observed) is int else None
                # Finite transport facts only; SDK text can contain signed URLs.
                raise RuntimeError(f"Exact metadata retrieval failed; path={path}; stage={phase}; "
                    f"status={status}; error_type={type(error).__name__}; no bulk fallback") from None
            import json
            json.loads(data)
            pin = hashlib.sha256(data).hexdigest()
            if target.exists() and file_digest(target) != pin:
                raise ValueError("Retained exact metadata differs")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_write(target, bytes(data))
            retained.append(dict(path=path, sha256=pin))
    return dict(kernel=kernel, version=version, files=retained,
        transport="version_specific_exact_file", scientific_acceptance=False,
        unselected_outputs_downloaded=False)


def _selected_files(files: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Selected output files must be a non-empty mapping")
    selected: dict[str, tuple[str, str]] = {}
    local_paths: set[str] = set()
    for remote, value in files.items():
        _path(remote)
        if (not isinstance(value, (tuple, list)) or len(value) != 2
                or type(value[0]) is not str):
            raise ValueError("Selected output binding must contain a local path and SHA256")
        local, digest = value
        _path(local)
        _digest(digest, str(local))
        if remote in selected or local.casefold() in local_paths:
            raise ValueError("Selected output paths are aliased")
        selected[remote] = (local, digest)
        local_paths.add(local.casefold())
    return selected


def _kernel_reference(api, context) -> tuple[str, str, str]:
    if api.get_config_value(api.CONFIG_NAME_USER) != context.account:
        raise ValueError("Authenticated account does not match owning run")
    owner, separator, slug = context.run_reference.partition("/")
    if not separator or "/" in slug or owner != context.account:
        raise ValueError("Expected exact owner/kernel reference")
    return owner, slug, context.run_reference


def _list_selected_urls(api, context, required: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    owner, slug, _ = _kernel_reference(api, context)
    urls: dict[str, str] = {}
    seen_tokens: set[str] = set()
    with api.build_kaggle_client() as client:
        token = None
        for _ in range(1000):
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = owner, slug
            request.page_size, request.page_token = 100, token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files:
                if item.file_name in required:
                    if item.file_name in urls:
                        raise ValueError("Duplicate remote output path")
                    urls[item.file_name] = item.url
            token = response.next_page_token
            if not token:
                break
            if token in seen_tokens:
                raise ValueError("Repeated output pagination token")
            seen_tokens.add(token)
        else:
            raise ValueError("Unbounded output listing")
    if set(urls) != set(required):
        raise ValueError("Required remote output files are missing; no bulk fallback")
    return urls


def retrieve_selected_kaggle_files(api, *, context, remote_manifest_path: str,
                                   files: Mapping[str, Any], destination: Path,
                                   http_session=None) -> dict:
    """Retrieve exactly ``files`` from the reconciled kernel output.

    ``files`` maps each full remote output path to ``(local_relative_path,
    sha256)``.  Callers must first validate their pinned manifest and build this
    allowlist; this function only performs account, listing, download, and
    atomic hash verification.
    """
    _path(remote_manifest_path)
    required = _selected_files(files)
    urls = _list_selected_urls(api, context, required)
    retained = _download_selected(urls=urls, required=required,
                                  destination=Path(destination), http_session=http_session)
    return {"kernel": context.run_reference, "receipt_version": context.platform_version,
            "files": retained, "unselected_outputs_downloaded": False,
            "platform_version_verified_by_output_api": False}


def retrieve_pinned_exact_files(api, *, context, remote_manifest_path: str,
                               files: Mapping[str, Any], destination: Path,
                               http_session=None, sizes=None, on_plan=None, _plan=None) -> dict:
    """Stream manifest-bound files at one physical version, without directory scans.

    A retained environment can contain thousands of unrelated files; enumerating
    it wastes requests and can exhaust the listing endpoint's rate limit.
    The caller still owns manifest and actual run-identity qualification.
    """
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    _path(remote_manifest_path)
    owner, slug, _ = _kernel_reference(api, context)
    version = context.platform_version
    if isinstance(version, str) and version.isascii() and version.isdecimal():
        version = int(version)
    if type(version) is not int or version < 1:
        raise ValueError("Exact artifact retrieval requires a physical version number")
    required = _selected_files(files)
    if len(required) > 4096:
        raise ValueError("Exact artifact allowlist exceeds the bounded request budget")
    # Only the synchronous adapter can supply its already checked plan.
    # Independent calls always inspect the retained bytes again.
    plan = _plan if _plan is not None else plan_pinned_files(required, destination, sizes=sizes)
    if on_plan is not None:
        on_plan(plan)
    reused = {row["remote_path"] for row in plan["files"] if row["state"] == "ALREADY_RETAINED"}
    retained = []
    with api.build_kaggle_client() as client:
        for remote, (local, pin) in required.items():
            target = repo_local_path(Path(destination).absolute(), local)
            _safe_local(target)
            if remote in reused:
                retained.append(dict(path=local, sha256=pin, state="ALREADY_RETAINED"))
                continue
            request = ApiDownloadKernelOutputRequest()
            request.owner_slug, request.kernel_slug = owner, slug
            request.version_number, request.file_path = version, remote
            try:
                redirect = client.kernels.kernels_api_client.download_kernel_output(request)
            except Exception:
                raise RuntimeError("Exact artifact redirect failed; no list/bulk fallback") from None
            retained.extend(_download_selected(urls={remote: redirect.url},
                required={remote: (local, pin)}, destination=destination, http_session=http_session))
    downloaded = sum(item["state"] != "ALREADY_RETAINED" for item in retained)
    return dict(kernel=context.run_reference, receipt_version=context.platform_version, files=retained,
        retrieval_plan=plan,
        transport="version_specific_manifest_pinned_files", unselected_outputs_downloaded=False,
        downloaded_files=downloaded, network_accessed=bool(downloaded),
        platform_version_verified_by_output_api=bool(downloaded))


def plan_pinned_files(files, destination, *, sizes=None):
    """Exact allowlist and known/unknown transfer bytes, before payload requests."""
    rows = []
    for remote, (local, pin) in _selected_files(files).items():
        target = repo_local_path(Path(destination).absolute(), local)
        _safe_local(target)
        present = target.is_file()
        if target.exists() and (not present or file_digest(target) != pin):
            raise ValueError("Retained exact output conflicts with pinned artifact")
        size = target.stat().st_size if present else (sizes or {}).get(remote)
        if size is not None and (type(size) is not int or size < 0):
            raise ValueError("Invalid pinned artifact byte size")
        rows.append(dict(remote_path=remote, local_path=local, sha256=pin, size_bytes=size,
                         state="ALREADY_RETAINED" if present else "DOWNLOAD"))
    missing = [row for row in rows if row["state"] == "DOWNLOAD"]
    return {"files": rows,
        "reused_bytes": sum(row["size_bytes"] for row in rows if row["state"] == "ALREADY_RETAINED"),
        "known_download_bytes": sum(row["size_bytes"] or 0 for row in missing),
        "unknown_size_files": sum(row["size_bytes"] is None for row in missing)}


def retrieve_execution_retention(
    api,
    *,
    context,
    spec,
    remote_manifest_path: str,
    local_manifest: Path,
    manifest_sha256: str,
    destination: Path,
    http_session=None,
    _plan=None,
) -> dict:
    """Retrieve and verify one compact execution-retention manifest.

    The pinned manifest is checked against the caller's Spec/package/source
    context before any remote listing.  Its fixed root/arm ledger paths and
    optional execution report become the complete allowlist passed to the
    bounded transport helper; the retention owner then verifies every
    downloaded byte and ledger field.
    """
    from .experiment_retention import (
        EXECUTION_REPORT_FILENAME,
        RETENTION_FILENAME,
        _load_json,
        _manifest_bytes,
        validate_execution_retention,
        validate_execution_retention_manifest,
    )
    from .experiment_spec import _digest as validate_digest

    pinned = Path(local_manifest).absolute()
    _safe_local(pinned)
    if not pinned.is_file() or file_digest(pinned) != manifest_sha256:
        raise ValueError("Pinned remote manifest hash mismatch")
    validate_digest(manifest_sha256, "manifest_sha256")
    raw = pinned.read_bytes()
    manifest = _load_json(pinned, "execution retention manifest")
    if _manifest_bytes(manifest) != raw:
        raise ValueError("Execution retention manifest must use canonical JSON bytes")
    if spec.identity != context.spec_identity:
        raise ValueError("Execution retention Spec identity mismatch")
    validate_execution_retention_manifest(
        manifest,
        spec,
        package_identity=context.package_identity,
        source_commit=context.source_commit,
        source_archive_sha256=context.source_archive_sha256,
    )

    prefix = PurePosixPath(remote_manifest_path).parent
    required: dict[str, tuple[str, str]] = {
        remote_manifest_path: (RETENTION_FILENAME, manifest_sha256),
    }
    root_ref = manifest["allocation_ledger"]["root"]
    root_remote = (prefix / root_ref["path"]).as_posix()
    required[root_remote] = (root_ref["path"], root_ref["sha256"])
    for arm_id, reference in manifest["allocation_ledger"]["arms"].items():
        remote = (prefix / reference["path"]).as_posix()
        required[remote] = (reference["path"], reference["sha256"])
    report = manifest["execution_report"]
    if report is not None:
        remote = (prefix / report["path"]).as_posix()
        required[remote] = (EXECUTION_REPORT_FILENAME, report["sha256"])

    version = context.platform_version
    transport = (retrieve_pinned_exact_files if isinstance(version, str) and version.isascii()
        and version.isdecimal() else retrieve_selected_kaggle_files)
    result = transport(
        api,
        context=context,
        remote_manifest_path=remote_manifest_path,
        files=required,
        destination=destination,
        http_session=http_session,
        **({"_plan": _plan} if transport is retrieve_pinned_exact_files else {}),
    )
    checked = validate_execution_retention(
        Path(destination),
        spec,
        package_identity=context.package_identity,
        source_commit=context.source_commit,
        source_archive_sha256=context.source_archive_sha256,
    )
    result.update({
        "status": "EXECUTION_RETENTION_RETRIEVED",
        "manifest_sha256": manifest_sha256,
        "execution_retention": {
            "status": checked["status"],
            "manifest": checked["manifest"],
        },
    })
    return result


__all__ = ["retrieve_selected_kaggle_files", "retrieve_execution_retention"]
