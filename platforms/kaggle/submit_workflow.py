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
    parser.add_argument("--repo-root",type=Path,default=Path.cwd())
    parser.add_argument("--credentials",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--authority",help="Authority path inside the immutable source package")
    parser.add_argument("--monitor-path",action="append")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    stage="arguments"
    try:
        if args.execute and not args.credentials:
            raise ValueError("--execute requires explicit credentials before preparation")
        if args.experiment:
            from molgap.experiment_input import prepare_experiment_input
            if args.output or args.authority or args.monitor_path:
                raise ValueError("Editable input already owns the immutable run/authority/monitor bindings")
            stage="local_preparation"
            with redirect_stdout(sys.stderr):
                prepared=prepare_experiment_input(args.experiment,args.repo_root)
            if not args.execute or prepared["status"] != "PREPARED_FOR_PLATFORM":
                print(canonical_json(prepared))
                return 0 if prepared["status"] == "PREPARED_FOR_PLATFORM" else 1
            args.prepared=Path(prepared["prepared"]);args.output=Path(prepared["operations"])
            args.authority=prepared["authority_ref"];args.monitor_path=prepared["monitor_paths"]
        if not args.credentials or not args.output or not args.authority or not args.monitor_path:
            raise ValueError("Platform operations require explicit credentials/output/authority/monitor bindings")
        stage="platform_operations"
        result = submit_prepared_workflow(args.prepared, args.credentials, args.output,
            authority_ref=args.authority, monitor_paths=args.monitor_path, execute=args.execute)
    except Exception as error:
        # SDK/HTTP exceptions may contain credential-bearing request details.
        if stage != "platform_operations":
            from molgap.experiment_input import local_preparation_error
            payload=local_preparation_error(error,stage)
        else:
            payload={"status":"BLOCKED","error_type":type(error).__name__,"stage":stage}
        print(canonical_json(payload))
        return 2
    print(canonical_json(result))
    return 0 if result["status"] in {"READY_FOR_PLATFORM_OPERATIONS", "SUBMISSION_BOUND"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
