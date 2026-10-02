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
from molgap.kaggle_output_retrieval import (
    retrieve_execution_retention as _retrieve_execution_retention,
    retrieve_selected_kaggle_files,
)
from molgap.research_memory.trace import file_digest


def retrieve_family_outputs(api, *, context: RunContext, remote_manifest_path: str,
                            local_manifest: Path, manifest_sha256: str,
                            destination: Path, http_session=None) -> dict:
    """Use an already authenticated owning-account KaggleApi.

    This function is called by the platform adapter/skill, not experiment_cli.
    local_manifest is the independently retained small remote manifest. No
    kernel submission, retry, inference, or platform-version guess occurs here.
    """
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
    result = retrieve_selected_kaggle_files(
        api, context=context, remote_manifest_path=remote_manifest_path,
        files=required, destination=destination, http_session=http_session,
    )
    result.update({"producer_version": manifest["context"]["platform_version"],
                   "manifest_sha256": manifest_sha256})
    return result


def retrieve_execution_retention(api, *, context, spec, remote_manifest_path: str,
                                 local_manifest: Path, manifest_sha256: str,
                                 destination: Path, http_session=None) -> dict:
    """Thin Kaggle adapter for the shared execution-retention retriever."""
    return _retrieve_execution_retention(
        api,
        context=context,
        spec=spec,
        remote_manifest_path=remote_manifest_path,
        local_manifest=local_manifest,
        manifest_sha256=manifest_sha256,
        destination=destination,
        http_session=http_session,
    )
