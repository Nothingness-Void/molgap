from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.kaggle_accelerator_push import (
    ALLOWED_ACCELERATORS,
    push_kernel_with_accelerator,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--accelerator", choices=sorted(ALLOWED_ACCELERATORS), required=True)
    parser.add_argument("--release-report", type=Path,
                        help="Recheck local source/recipe/init/entry bindings before sending the request")
    parser.add_argument("--response-output", type=Path,
                        help="Persist the returned platform identity; distinct from scientific acceptance")
    args = parser.parse_args()
    result = push_kernel_with_accelerator(
        package_dir=args.package,
        credential_path=args.credentials,
        accelerator=args.accelerator,
        release_report_path=args.release_report,
    )
    if args.response_output:
        from molgap.training_reproducibility import atomic_json
        atomic_json(args.response_output, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "submitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
