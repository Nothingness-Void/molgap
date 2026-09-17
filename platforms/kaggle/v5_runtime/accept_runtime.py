from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.kaggle_runtime_acceptance import accept_kaggle_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = accept_kaggle_runtime(
        payload_root=args.payload_root,
        metadata_path=args.metadata,
        expected_account=args.account,
        expected_commit=args.source_commit,
        expected_manifest_sha256=args.manifest_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
