from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.kaggle_fixed_mirror import FIXED_MIRRORS, verify_published_mirror


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--role", choices=sorted(FIXED_MIRRORS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record_root = args.record_root.resolve()
    result = verify_published_mirror(
        manifest_path=record_root / "manifest.json",
        remote_files_path=record_root / "remote_files.csv",
        remote_metadata_path=record_root / "dataset-metadata.json",
        account=args.account,
        role=args.role,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
