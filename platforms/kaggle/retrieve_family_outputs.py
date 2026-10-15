"""Selective retrieval for the family output protocol; no all-output fallback.

The platform skill reconciles the exact kernel/version and retrieves/pins its
small manifest first. This adapter downloads only manifest-bound required files.
Kaggle's output-list API addresses the latest session, so every downloaded byte
is rechecked against that independently pinned manifest rather than trusting a
slug to identify an attempt. Signed URLs never appear in returned receipts.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

from molgap.experiment_family_workflow import RunContext, ROLES, _check_context, _json
from molgap.experiment_family_artifacts import artifact_adapter
from molgap.experiment_launch import _safe_local
from molgap.experiment_terminal import _path
from molgap.research_memory.trace import file_digest


def retrieve_family_outputs(api, *, context: RunContext, remote_manifest_path: str,
                            local_manifest: Path, manifest_sha256: str,
                            destination: Path, http_session=None) -> dict:
    """Use an already authenticated owning-account KaggleApi.

    This function is called by the platform adapter/skill, not experiment_cli.
    local_manifest is the independently retained small remote manifest. No
    kernel submission, retry, inference, or platform-version guess occurs here.
    """
    _path(remote_manifest_path)
    _safe_local(Path(local_manifest).absolute())
    if file_digest(local_manifest) != manifest_sha256:
        raise ValueError("Pinned remote manifest hash mismatch")
    manifest = _json(Path(local_manifest))
    if manifest.get("format") != "molgap-family-output-v1":
        raise ValueError("Unsupported output protocol; use the owning legacy retriever")
    _check_context(manifest["context"], context)
    artifact_adapter(manifest["adapter"], (context.family_name, context.family_version))
    if set(manifest["artifacts"]) != ROLES:
        raise ValueError("Incomplete retention manifest")
    if api.get_config_value(api.CONFIG_NAME_USER) != context.account:
        raise ValueError("Authenticated account does not match owning run")
    prefix = PurePosixPath(remote_manifest_path).parent
    required = {}
    for role, item in manifest["artifacts"].items():
        _path(item["path"])
        from molgap.experiment_spec import _digest
        _digest(item["sha256"], role)
        remote = (prefix / item["path"]).as_posix()
        if remote in required:
            raise ValueError("Aliased retention roles")
        required[remote] = (item["path"], item["sha256"])
    required[remote_manifest_path] = ("output_manifest.json", manifest_sha256)
    root = Path(destination).absolute()
    _safe_local(root)
    root.mkdir(parents=True, exist_ok=True)
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner, separator, slug = context.run_reference.partition("/")
    if not separator or "/" in slug or owner != context.account:
        raise ValueError("Expected exact owner/kernel reference")
    urls, seen_tokens = {}, set()
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
    from molgap.kaggle_output_retrieval import _download_selected
    retained = _download_selected(urls=urls, required=required,
                                  destination=root, http_session=http_session)
    return {"kernel": context.run_reference, "receipt_version": context.platform_version,
            "producer_version": manifest["context"]["platform_version"], "files": retained,
            "manifest_sha256": manifest_sha256, "unselected_outputs_downloaded": False,
            "platform_version_verified_by_output_api": False}
