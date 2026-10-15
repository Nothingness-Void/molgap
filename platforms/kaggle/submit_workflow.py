"""Platform entry: check or submit one already prepared authorized workflow."""
import argparse
from pathlib import Path

from molgap.experiment_launch import canonical_json
from molgap.kaggle_submission import submit_prepared_workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--credentials", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--authority", required=True, help="Authority path inside the immutable source package")
    parser.add_argument("--monitor-path", action="append", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = submit_prepared_workflow(args.prepared, args.credentials, args.output,
            authority_ref=args.authority, monitor_paths=args.monitor_path, execute=args.execute)
    except Exception as error:
        # SDK/HTTP exceptions may contain credential-bearing request details.
        print(canonical_json({"status": "BLOCKED", "error_type": type(error).__name__}))
        return 2
    print(canonical_json(result))
    return 0 if result["status"] in {"READY_FOR_PLATFORM_OPERATIONS", "SUBMISSION_BOUND"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
