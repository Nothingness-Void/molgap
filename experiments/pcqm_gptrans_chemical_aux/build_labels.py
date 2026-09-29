"""Thin CPU cache CLI; inputs must be an authenticated train-only export."""
import argparse
from pathlib import Path
from molgap.chemical_aux_cache import build_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rows", "role", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("rows-sha256", "role-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = build_cache(args.rows, args.rows_sha256, args.role, args.role_sha256, args.output)
    print("ACCEPTED" if result["accepted"] else "LABEL_FAILURES_RETAINED")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
