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
    manifest, required = _family_files(context, remote_manifest_path, local_manifest,
                                       manifest_sha256)
    result = retrieve_selected_kaggle_files(
        api, context=context, remote_manifest_path=remote_manifest_path,
        files=required, destination=destination, http_session=http_session,
    )
    result.update({"producer_version": manifest["context"]["platform_version"],
                   "manifest_sha256": manifest_sha256})
    return result


def _family_files(context, remote_manifest_path, local_manifest, manifest_sha256):
    """Keep CLI preflight and the original retriever on the same manifest checks."""
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
    return manifest, required


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


def main(argv=None):
    """Validate an original run locally, or explicitly retrieve its pinned files."""
    import argparse
    from contextlib import redirect_stderr, redirect_stdout
    import io
    from molgap.common.paths import MachinePaths
    from molgap.experiment_launch import canonical_json
    from molgap.experiment_run import _original_run, RunReconciliationRequired
    from molgap.kaggle_output_retrieval import _selected_files

    class Parser(argparse.ArgumentParser):
        def error(self, message):
            # argparse's message may echo a credential-bearing argument.
            raise ValueError("Invalid retrieval arguments")

    parser = Parser(description=main.__doc__, allow_abbrev=False)
    for name in ("run-id", "attempt-id", "arm", "manifest-sha256", "remote-manifest-path"):
        parser.add_argument("--" + name, required=True)
    for name in ("profile", "local-manifest", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--execute", action="store_true")
    stage = "arguments"
    try:
        args = parser.parse_args(argv)
        if bool(args.credentials) != args.execute:
            raise ValueError("Explicit credentials are required only with --execute")
        # SDK diagnostics may contain credentials or signed URLs. Only the
        # adapter's finite JSON receipt is allowed on either output stream.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            stage = "original_run"
            paths = MachinePaths.from_file(args.profile).outside_checkout(args.repo_root)
            spec, options, prepared = _original_run(
                args.profile, args.repo_root, args.run_id, args.attempt_id)
            context = RunContext.from_launch(spec, options["receipt_path"], options["package_dir"],
                expected_package_identity=options["expected_package_identity"], arm_id=args.arm)
            if context.platform != "kaggle":
                raise ValueError("Expected an original Kaggle run")
            stage = "manifest"
            _path(args.remote_manifest_path)
            manifest, files = _family_files(context, args.remote_manifest_path,
                                           args.local_manifest, args.manifest_sha256)
            if len(files) != len(ROLES) + 1:
                raise ValueError("Remote manifest aliases an artifact")
            files = _selected_files(files)
            # _family_files checks known producer versions against the receipt.
            # A normal pre-launch producer has no version; retain that null and
            # bind retrieval through the reconciled receipt and pinned bytes.
            if context.platform_version is None:
                raise ValueError("An exact reconciled receipt version is required")
            stage = "destination"
            destination = args.destination.absolute()
            _safe_local(destination)
            artifact_root = paths.artifact_root
            run = prepared.parent
            if (destination == artifact_root or not destination.is_relative_to(artifact_root)
                    or destination.is_relative_to(run) or run.is_relative_to(destination)):
                raise ValueError("Destination must be under artifact_root, outside the frozen run")
            if destination.exists():
                if not destination.is_dir():
                    raise ValueError("Destination must be a directory")
                retained_manifest = destination / "output_manifest.json"
                _safe_local(retained_manifest)
                if next(destination.iterdir(), None) is not None and (
                        not retained_manifest.is_file()
                        or file_digest(retained_manifest) != args.manifest_sha256):
                    raise ValueError("Existing retrieval must retain the original pinned manifest")
            for relative, pin in files.values():
                target = destination / relative
                _safe_local(target)
                if target.exists() and (not target.is_file() or file_digest(target) != pin):
                    raise ValueError("Retained output conflicts with pinned artifact")
            result = {"status": "READY_FOR_RETRIEVAL", "executed": False,
                      "network_accessed": False, "scientific_acceptance": False,
                      "producer_version": manifest["context"]["platform_version"],
                      "manifest_sha256": args.manifest_sha256}
            if args.execute:
                stage = "platform_retrieval"
                from molgap.kaggle_submission import connect_kaggle
                api = connect_kaggle(args.credentials)
                result = retrieve_family_outputs(api, context=context,
                    remote_manifest_path=args.remote_manifest_path, local_manifest=args.local_manifest,
                    manifest_sha256=args.manifest_sha256, destination=destination)
                result.update(status="RETRIEVED", executed=True, scientific_acceptance=False)
            encoded = canonical_json(result)
    except Exception as error:
        # Never serialize exception text: even local paths may contain secrets.
        status = "RECONCILIATION_REQUIRED" if isinstance(error, RunReconciliationRequired) else "BLOCKED"
        print(canonical_json({"status": status, "stage": stage,
                              "error": "Retrieval checks or operation failed"}))
        return 2
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
