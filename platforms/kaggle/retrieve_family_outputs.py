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
from molgap.experiment_family_artifacts import artifact_adapter, SESSION_OPTIONAL_ARTIFACTS
from molgap.experiment_launch import _safe_local
from molgap.experiment_terminal import _path
from molgap.kaggle_output_retrieval import (
    retrieve_execution_retention as _retrieve_execution_retention,
    retrieve_exact_metadata,
    retrieve_pinned_exact_files,
    retrieve_selected_kaggle_files,
)
from molgap.research_memory.trace import file_digest


def retrieve_native_dependencies(api, *, context, manifest, remote_manifest_path,
        destination, resume_checkpoint_sha256=None, http_session=None, on_plan=None):
    """Retrieve only certificate-bound native dependencies; acceptance stays local."""
    from tempfile import TemporaryDirectory
    from molgap.family_training_runner import inspect_retained_native_qualification
    from molgap.research_memory.paths import repo_local_path
    from molgap.screen_policy import canonical_fingerprint

    destination = Path(destination).absolute()
    prefix = PurePosixPath(remote_manifest_path).parent
    resumed = resume_checkpoint_sha256 is not None
    native = destination.parent / "_resume_preflight" / context.arm_id if resumed else destination
    remote_native = prefix.parent / "_resume_preflight" / context.arm_id if resumed else prefix
    actual_resume = native
    remote_resume = remote_native
    # Older continuation retrievals retained the original certificate beside
    # the output and actual-resume proof in its separate runtime directory.
    if resumed and (destination / "runtime_certificate.json").is_file():
        native = destination
    _safe_local(native)
    version = context.platform_version
    if not isinstance(version, str) or not version.isascii() or not version.isdecimal():
        raise ValueError("Strict dependency retrieval requires an exact numeric version")
    certificate_id = manifest.get("runtime", {}).get("runtime_certificate_id")
    from molgap.experiment_spec import _digest
    _digest(certificate_id, "runtime_certificate_id")

    transfers, plans = [], []

    def announce(plan):
        plans.append(plan)
        if on_plan is not None:
            on_plan(plan)

    def metadata(names, *, remote_root=remote_native, local_root=None):
        local_root = native if local_root is None else local_root
        retained, missing = {}, []
        for name in names:
            path = repo_local_path(local_root, name)
            _safe_local(path)
            if path.is_file():
                retained[name] = path.read_bytes()
            else:
                missing.append(name)
        if not missing:
            return retained
        announce({"phase": "native_metadata", "files": [
            {"remote_path": (remote_root / name).as_posix(), "size_bytes": None}
            for name in missing], "unknown_size_files": len(missing),
            "known_download_bytes": 0, "reused_bytes": sum(len(raw) for raw in retained.values())})
        with TemporaryDirectory(prefix=".native-metadata-", dir=destination.parent) as temporary:
            paths = [(remote_root / name).as_posix() for name in missing]
            retrieve_exact_metadata(api, account=context.account, kernel=context.run_reference,
                version=int(version), paths=paths, destination=Path(temporary), http_session=http_session)
            transfers.extend(paths)
            retained.update({name: repo_local_path(Path(temporary), remote).read_bytes()
                for name, remote in zip(missing, paths)})
        return retained

    from molgap.experiment_family_workflow import _json_bytes
    payloads = metadata(["runtime_certificate.json", "runtime_provenance.json"])
    certificate, provenance = (_json_bytes(payloads[name]) for name in
        ("runtime_certificate.json", "runtime_provenance.json"))
    if (canonical_fingerprint(certificate) != certificate_id
            or certificate.get("provenance_sha256") != canonical_fingerprint(provenance)):
        raise ValueError("Native metadata differs from the manifest-bound certificate")
    from molgap.experiment_launch import publish_immutable_bytes
    for name, payload in payloads.items():
        publish_immutable_bytes(native / name, payload)
    if resumed:
        output_provenance = metadata(["runtime_provenance.json"], remote_root=prefix,
            local_root=destination)["runtime_provenance.json"]
        if canonical_fingerprint(_json_bytes(output_provenance)) != certificate["provenance_sha256"]:
            raise ValueError("Continued output provenance differs from its native certificate")
        publish_immutable_bytes(destination / "runtime_provenance.json", output_provenance)
    if resumed and not (actual_resume / "actual_resume_certificate.json").is_file():
        payload = metadata(["actual_resume_certificate.json"], remote_root=remote_resume,
            local_root=actual_resume)["actual_resume_certificate.json"]
        publish_immutable_bytes(actual_resume / "actual_resume_certificate.json", payload)

    recipe = repo_local_path(destination, manifest["artifacts"]["contract"]["path"])
    for _ in range(4):
        inventory = inspect_retained_native_qualification(native, recipe_path=recipe,
            provenance=provenance, context=context, resume_checkpoint_sha256=resume_checkpoint_sha256,
            resume_directory=actual_resume, inventory_only=True)
        if inventory["status"] == "DEPENDENCIES_VERIFIED":
            return {"status": "DEPENDENCIES_VERIFIED", "native_root": str(native),
                "known_local_bytes": inventory["known_local_bytes"], "retrieval_plans": plans,
                "downloaded_files": transfers, "network_accessed": bool(transfers),
                "scientific_acceptance": "NOT_EVALUATED", "native_qualification": "NOT_EVALUATED"}
        missing = [item for item in inventory["dependencies"] if not item["present"] and item["sha256"] is not None]
        # Metadata frontiers are fetched first; only then request proof archives
        # or other certificate-bound files. Never enumerate unrelated outputs.
        selected = [item for item in missing if Path(item["path"]).suffix == ".json"] or missing
        if not selected:
            raise ValueError("Native dependency has no retained hash binding")
        groups, sizes = {}, {}
        for item in selected:
            path = Path(item["local_path"])
            local_root, remote_root = ((actual_resume, remote_resume)
                if resumed and path.is_relative_to(actual_resume) else (native, remote_native))
            relative = path.relative_to(local_root).as_posix()
            remote = (remote_root / relative).as_posix()
            groups.setdefault(local_root, {})[remote] = (relative, item["sha256"])
            sizes[remote] = item["size_bytes"]
        for local_root, files in groups.items():
            result = retrieve_pinned_exact_files(api, context=context, remote_manifest_path=remote_manifest_path,
                files=files, destination=local_root, http_session=http_session, sizes=sizes, on_plan=announce)
            transfers.extend(item["path"] for item in result["files"] if item["state"] != "ALREADY_RETAINED")
    raise ValueError("Native dependency inventory did not resolve within its bounded frontiers")


def retrieve_family_outputs(api, *, context: RunContext, remote_manifest_path: str,
                            local_manifest: Path, manifest_sha256: str,
                            destination: Path, http_session=None, on_plan=None) -> dict:
    """Use an already authenticated owning-account KaggleApi.

    This function is called by the platform adapter/skill, not experiment_cli.
    local_manifest is the independently retained small remote manifest. No
    kernel submission, retry, inference, or platform-version guess occurs here.
    """
    manifest, required = _family_files(context, remote_manifest_path, local_manifest,
                                       manifest_sha256)
    version = context.platform_version
    transport = (retrieve_pinned_exact_files if isinstance(version, str) and version.isascii()
                 and version.isdecimal() else retrieve_selected_kaggle_files)
    options = {"on_plan": on_plan, "sizes": {remote_manifest_path: local_manifest.stat().st_size}}
    result = transport(
        api, context=context, remote_manifest_path=remote_manifest_path,
        files=required, destination=destination, http_session=http_session,
        **(options if transport is retrieve_pinned_exact_files else {}),
    )
    # The exact-file transport converts the canonical text version to the
    # SDK's integer request field.  Keep the adapter receipt ABI in its
    # original RunContext spelling so callers do not observe transport detail.
    result.update({"producer_version": manifest["context"]["platform_version"],
                   "receipt_version": context.platform_version,
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
    profile = artifact_adapter(manifest["adapter"], (context.family_name, context.family_version))
    # The shared session already defines final-only outputs without prediction
    # payloads. Retrieval transports that profile, while the owning inspector
    # still validates the pinned recipe's role/selection requirements.
    roles = ROLES - {"predictions"} if (profile.protocol == "family-session" and
        "predictions" not in manifest["artifacts"]) else ROLES
    if profile.protocol == "family-session" and "trace_context" in manifest["artifacts"]:
        roles = roles | {"trace_context"}
        if manifest["artifacts"]["trace_context"]["path"] != manifest["artifacts"]["trace"]["path"] + ".context.json":
            raise ValueError("Trace companion is not bound to the declared trace")
    if profile.protocol == "family-session":
        roles = roles | (set(manifest["artifacts"]) & SESSION_OPTIONAL_ARTIFACTS)
    if set(manifest["artifacts"]) != roles:
        raise ValueError("Incomplete retention manifest")
    prefix = PurePosixPath(remote_manifest_path).parent
    # Keep the small pinned manifest in the first publication frontier so a
    # failed payload transfer always leaves the identity needed for safe
    # re-entry; artifact bytes remain independently hash-bound.
    required = {remote_manifest_path: ("output_manifest.json", manifest_sha256)}
    for role, item in manifest["artifacts"].items():
        _path(item["path"])
        from molgap.experiment_spec import _digest
        _digest(item["sha256"], role)
        remote = (prefix / item["path"]).as_posix()
        if remote in required:
            raise ValueError("Aliased retention roles")
        required[remote] = (item["path"], item["sha256"])
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
    from molgap.kaggle_output_retrieval import _selected_files, plan_pinned_files

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
    parser.add_argument("--continuation-receipt", type=Path)
    parser.add_argument("--resume-prepared", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--execute", action="store_true")
    stage = "arguments"
    try:
        args = parser.parse_args(argv)
        if bool(args.credentials) != args.execute:
            raise ValueError("Explicit credentials are required only with --execute")
        if (args.resume_prepared is not None and (not args.strict or args.continuation_receipt is None)
                or args.strict and args.continuation_receipt is not None and args.resume_prepared is None):
            raise ValueError("Strict continuation retrieval requires its original resume preparation")
        # SDK diagnostics may contain credentials or signed URLs. Only the
        # adapter's finite JSON receipt is allowed on either output stream.
        # Plans belong in that receipt, not an outer progress stream which
        # would escape redirection even when a later transport check fails.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            stage = "original_run"
            paths = MachinePaths.from_file(args.profile).outside_checkout(args.repo_root)
            spec, options, prepared = _original_run(
                args.profile, args.repo_root, args.run_id, args.attempt_id)
            from molgap.experiment_run_acceptance import _continuation_options
            options = _continuation_options(spec, options, args.continuation_receipt, args.repo_root)
            context = RunContext.from_launch(spec, options["receipt_path"], options["package_dir"],
                expected_package_identity=options["expected_package_identity"], arm_id=args.arm)
            if context.platform != "kaggle":
                raise ValueError("Expected an original Kaggle run")
            stage = "manifest"
            _path(args.remote_manifest_path)
            manifest, files = _family_files(context, args.remote_manifest_path,
                                           args.local_manifest, args.manifest_sha256)
            if len(files) != len(manifest["artifacts"]) + 1:
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
            version = context.platform_version
            if args.strict and (not isinstance(version, str) or not version.isascii() or not version.isdecimal()):
                raise ValueError("Strict dependency retrieval requires an exact numeric version")
            resume_checkpoint = None
            if args.resume_prepared is not None:
                from molgap.experiment_spec import _digest
                from molgap.experiment_workflow_resume import resolve_terminal_resume
                bindings = resolve_terminal_resume({"spec": spec, "options": options, "prepared": prepared,
                    "outputs": {arm["arm_id"]: {"output_dir": str(args.destination.parent / arm["arm_id"])}
                                for arm in spec.to_dict()["arms"]}}, args.resume_prepared, None)
                resume_checkpoint = _json(bindings[args.arm]["resume_bundle"])["artifacts"]["checkpoint"]["sha256"]
                _digest(resume_checkpoint, "resume checkpoint.sha256")
            retrieval_plan = None
            if isinstance(version, str) and version.isascii() and version.isdecimal():
                retrieval_plan = plan_pinned_files(files, destination,
                    sizes={args.remote_manifest_path: args.local_manifest.stat().st_size})
            result = {"status": "READY_FOR_RETRIEVAL", "executed": False,
                      "network_accessed": False, "scientific_acceptance": False,
                      "producer_version": manifest["context"]["platform_version"],
                      "manifest_sha256": args.manifest_sha256,
                      "retrieval_plan": retrieval_plan}
            if args.strict:
                result["native_dependencies"] = "resolved from pinned certificate before proof transfer"
            if args.execute:
                stage = "platform_retrieval"
                from molgap.kaggle_submission import connect_kaggle
                api = connect_kaggle(args.credentials)
                result = retrieve_family_outputs(api, context=context,
                    remote_manifest_path=args.remote_manifest_path, local_manifest=args.local_manifest,
                    manifest_sha256=args.manifest_sha256, destination=destination)
                if args.strict:
                    stage = "native_dependencies"
                    result["native_dependencies"] = retrieve_native_dependencies(api, context=context,
                        manifest=manifest, remote_manifest_path=args.remote_manifest_path,
                        destination=destination, resume_checkpoint_sha256=resume_checkpoint)
                    result["network_accessed"] = (result.get("network_accessed", False)
                        or result["native_dependencies"]["network_accessed"])
                stage = "retrieval_mapping"
                from molgap.experiment_run_acceptance import record_run_retrieval
                mapping = record_run_retrieval(args.profile, args.repo_root, args.run_id, args.attempt_id,
                    context=context, destination=destination, manifest_sha256=args.manifest_sha256,
                    continuation_receipt=args.continuation_receipt)
                result["retrieval_mapping"] = str(mapping)
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
