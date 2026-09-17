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
    args = parser.parse_args()
    result = push_kernel_with_accelerator(
        package_dir=args.package,
        credential_path=args.credentials,
        accelerator=args.accelerator,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
