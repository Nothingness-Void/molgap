from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.kaggle_fixed_mirror import FIXED_MIRRORS, prepare_account_mirror


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--role", choices=sorted(FIXED_MIRRORS), required=True)
    args = parser.parse_args()
    report = prepare_account_mirror(args.payload_root, args.account, args.role)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
