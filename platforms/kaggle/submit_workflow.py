"""Platform entry: check or submit one already prepared authorized workflow."""
import argparse
from contextlib import redirect_stdout
from pathlib import Path
import sys

from molgap.experiment_launch import canonical_json
from molgap.kaggle_submission import submit_prepared_workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prepared",type=Path)
    inputs.add_argument("--experiment",type=Path)
    parser.add_argument("--recover-prepared", action="store_true",
                        help="Finish a failed published preparation from its frozen inputs")
    parser.add_argument("--repo-root",type=Path,default=Path.cwd())
    parser.add_argument("--credentials",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--authority",help="Authority path inside the immutable source package")
    parser.add_argument("--recovery-authority", type=Path,
                        help="Explicit retained authorization for this checkpoint continuation")
    parser.add_argument("--recovery-authority-sha256")
    parser.add_argument("--source-version", type=int, default=1,
                        help="Exact source successor version; continuation requires predecessor version minus one")
    parser.add_argument("--kernel-version", type=int, default=1,
                        help="Exact kernel successor version; continuation requires predecessor version minus one")
    parser.add_argument("--monitor-path",action="append")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print the full owner report")
    args = parser.parse_args(argv)
    stage="arguments"
    try:
        if args.recover_prepared and not args.prepared:
            raise ValueError("--recover-prepared requires --prepared")
        if args.experiment and (args.recovery_authority is not None or args.recovery_authority_sha256 is not None
                                or args.source_version != 1 or args.kernel_version != 1):
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
            if not args.credentials or not args.output or not args.authority or not args.monitor_path:
                raise ValueError("Platform operations require explicit credentials/output/authority/monitor bindings")
            stage="platform_operations"
            result = submit_prepared_workflow(args.prepared, args.credentials, args.output,
                authority_ref=args.authority, monitor_paths=args.monitor_path, execute=args.execute,
                recovery_authority=args.recovery_authority,
                recovery_authority_sha256=args.recovery_authority_sha256,
                source_version=args.source_version, kernel_version=args.kernel_version,
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
