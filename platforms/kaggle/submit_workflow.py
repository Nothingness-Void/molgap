"""Platform entry: check or submit one already prepared authorized workflow."""
import argparse
from contextlib import redirect_stdout
from pathlib import Path
import sys

from molgap.experiment_launch import canonical_json
from molgap.kaggle_submission import SourceVerificationError, submit_prepared_workflow


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Normal submit: platforms/local/run.py --profile $profile --script platforms/kaggle/submit_workflow.py "
               "-- --experiment $experiment --repo-root . --credentials $credentials --execute",
    )
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prepared",type=Path,help="Existing prepared workflow or prepare-resume directory")
    inputs.add_argument("--experiment",type=Path,help="Committed editable experiment.json to freeze, prepare and submit")
    parser.add_argument("--recover-prepared", action="store_true",
                        help="Finish a failed published preparation from its frozen inputs (local only, omit --execute)")
    parser.add_argument("--repo-root",type=Path,default=Path.cwd(),help="Owning checkout root (default: cwd)")
    parser.add_argument("--credentials",type=Path,help="External Kaggle kaggle.json path (required with --execute)")
    parser.add_argument("--output",type=Path,help="Operations/journal directory (only with legacy --prepared)")
    parser.add_argument("--authority",help="Authority path inside the immutable source package")
    parser.add_argument("--recovery-authority", type=Path,
                        help="Explicit retained authorization for this checkpoint continuation")
    parser.add_argument("--recovery-authority-sha256",help="Expected SHA-256 of --recovery-authority")
    parser.add_argument("--source-version", type=int, help="Expected source successor version")
    parser.add_argument("--kernel-version", type=int, help="Expected kernel successor version")
    parser.add_argument("--monitor-path",action="append",help="Expected remote output path to monitor (repeatable)")
    parser.add_argument("--execute", action="store_true", help="Publish source and submit kernel (omit for local check only)")
    parser.add_argument("--verbose", action="store_true", help="Print the full owner report")
    args = parser.parse_args(argv)
    stage="arguments"
    try:
        if args.recover_prepared and not args.prepared:
            raise ValueError("--recover-prepared requires --prepared")
        if args.experiment and (args.recovery_authority is not None or args.recovery_authority_sha256 is not None
                                or args.source_version not in {None, 1} or args.kernel_version not in {None, 1}):
            raise ValueError("Continuation requires an existing --prepared resume workflow")
        if args.execute and not args.credentials:
            raise ValueError("--execute requires explicit credentials before preparation")
        from contextlib import nullcontext
        from molgap.experiment_preflight import _shared_release_verification
        from molgap.experiment_package import _PackageVerification
        verification = _PackageVerification()
        scope = _shared_release_verification() if (args.experiment and args.execute) else nullcontext()
        with scope:
            if args.recover_prepared:
                from molgap.kaggle_workflow import recover_prepared_workflow
                stage="local_preparation_recovery"
                with redirect_stdout(sys.stderr):
                    prepared = recover_prepared_workflow(args.prepared, operations=args.output,
                                                         _verification=verification)
                if not args.execute:
                    from molgap.experiment_input import compact_operation_report
                    print(canonical_json(prepared if args.verbose else compact_operation_report(prepared)))
                    return 0 if prepared["status"] == "PREPARED_FOR_PLATFORM" else 1
            if args.experiment:
                from molgap.experiment_input import prepare_experiment_input
                if args.output or args.authority or args.monitor_path:
                    raise ValueError("Editable input already owns the immutable run/authority/monitor bindings")
                stage="local_preparation"
                with redirect_stdout(sys.stderr):
                    prepared=prepare_experiment_input(args.experiment,args.repo_root, _verification=verification)
                if not args.execute or prepared["status"] != "PREPARED_FOR_PLATFORM":
                    from molgap.experiment_input import compact_operation_report
                    print(canonical_json(prepared if args.verbose else compact_operation_report(prepared)))
                    return 0 if prepared["status"] == "PREPARED_FOR_PLATFORM" else 1
                args.prepared=Path(prepared["prepared"]);args.output=Path(prepared["operations"])
                args.authority=prepared["authority_ref"];args.monitor_path=prepared["monitor_paths"]
            elif args.prepared:
                from molgap.experiment_family_workflow import _json
                from molgap.experiment_launch import _safe_local
                from molgap.kaggle_submission import prepared_submission_owner
                report_path = args.prepared.absolute() / "workflow_report.json"
                _safe_local(report_path)
                report = _json(report_path)
                if report.get("format") == "molgap-workflow-recovery-preparation-v1":
                    original = Path(report["original_receipt"])
                    _safe_local(original)
                    original_operations, origin_binding = prepared_submission_owner(report["original_prepared"])
                    if original_operations != original.parent.resolve():
                        if not (args.output and args.authority and args.monitor_path
                                and args.source_version is not None and args.kernel_version is not None):
                            raise ValueError("Legacy resume needs explicit platform arguments")
                    else:
                        operations, binding = prepared_submission_owner(args.prepared)
                        args.output = args.output or operations or args.prepared.with_name(args.prepared.name + "-operations")
                        args.authority = args.authority or origin_binding["authority_ref"]
                        args.monitor_path = args.monitor_path or origin_binding["monitor_paths"]
                        previous = (binding or {}).get("continuation", {})
                        # Reentry takes its version from the bound journal, including after a lost response.
                        args.source_version = args.source_version if args.source_version is not None else previous.get("source_version", 2)
                        kernel_version = int(_json(original)["platform_version"]["value"]) + 1
                        args.kernel_version = args.kernel_version if args.kernel_version is not None else previous.get("kernel_version", kernel_version)
                        if args.recovery_authority is None and previous:
                            args.recovery_authority = Path(previous["authority_path"])
                        if args.recovery_authority_sha256 is None:
                            args.recovery_authority_sha256 = previous.get("authority_sha256")
            if not args.credentials or not args.output or not args.authority or not args.monitor_path:
                raise ValueError("Platform operations require explicit credentials/output/authority/monitor bindings")
            stage="platform_operations"
            result = submit_prepared_workflow(args.prepared, args.credentials, args.output,
                authority_ref=args.authority, monitor_paths=args.monitor_path, execute=args.execute,
                recovery_authority=args.recovery_authority,
                recovery_authority_sha256=args.recovery_authority_sha256,
                source_version=1 if args.source_version is None else args.source_version,
                kernel_version=1 if args.kernel_version is None else args.kernel_version,
                _verification=verification)
    except Exception as error:
        # SDK/HTTP exceptions may contain credential-bearing request details.
        if stage != "platform_operations":
            from molgap.experiment_input import local_preparation_error
            payload=local_preparation_error(error,stage)
        else:
            from molgap.common.kaggle import platform_error_details
            payload={"status":"BLOCKED","error_type":type(error).__name__,"stage":stage,
                     "platform_error":platform_error_details(error, stage=stage, endpoint_class="kaggle_api"),
                     "error_code":"PLATFORM_" + type(error).__name__.upper(),
                     "field":"platform.journal",
                     "next_action":"Inspect the submission journal and reconcile before retrying"}
            if args.output:
                payload["journal"]=str(args.output / "submission_state.json")
            if isinstance(error, SourceVerificationError):
                payload.update(field="source_upload", reason=str(error),
                    error_code="SOURCE_VERIFICATION_FAILED")
            if getattr(error,"release_recheck",None) is not None:
                payload["release_recheck"]=error.release_recheck
                payload["field"]="release.inputs"
                payload["next_action"]="Inspect the failed release check; no automatic retry"
            persistence_error=getattr(error,"journal_error",None)
            if persistence_error is not None:
                payload["journal_persistence_error"]={"error_type":type(persistence_error).__name__,
                    "errno":persistence_error.errno,"winerror":getattr(persistence_error,"winerror",None)}
        print(canonical_json(payload))
        return 2
    from molgap.experiment_input import compact_operation_report
    print(canonical_json(result if args.verbose else compact_operation_report(
        result, prepared=args.prepared, operations=args.output)))
    return 0 if result["status"] in {"READY_FOR_PLATFORM_OPERATIONS", "SUBMISSION_BOUND"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
