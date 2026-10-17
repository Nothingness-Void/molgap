"""Local-only declaration freeze and package preparation for native qualification."""
import argparse
import json
from pathlib import Path

from molgap.modern2d_qualification_prepare import freeze_qualification_declarations, prepare_package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", allow_abbrev=False)
    freeze.add_argument("--input-metadata", type=Path, required=True)
    freeze.add_argument("--input-metadata-sha256", required=True)
    freeze.add_argument("--init-root", type=Path, required=True)
    package = commands.add_parser("package", allow_abbrev=False)
    package.add_argument("--output-root", type=Path, required=True)
    package.add_argument("--init-root", type=Path, required=True)
    package.add_argument("--mounted-root", type=Path, required=True)
    package.add_argument("--expected-spec-identity", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_qualification_declarations(args.repo_root, args.input_metadata,
            args.input_metadata_sha256, init_root=args.init_root)
    else:
        result = prepare_package(args.repo_root, args.output_root, init_root=args.init_root, mounted_root=args.mounted_root,
            expected_spec_identity=args.expected_spec_identity)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
