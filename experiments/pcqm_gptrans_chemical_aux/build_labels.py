"""Thin CPU cache CLI; inputs must be an authenticated train-only export."""
import argparse
import json
from pathlib import Path
from molgap.chemical_aux_cache import build_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rows", "role", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("rows-sha256", "role-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--components", choices=("both", "descriptors", "fingerprints"), default="both")
    parser.add_argument("--parse-policy", choices=("strict", "pcqm_topology"), default="strict")
    parser.add_argument("--descriptor-missing-policy", choices=("reject", "mask_nonfinite"), default="reject")
    parser.add_argument("--collect-all-failures", action="store_true")
    args = parser.parse_args()
    components = ("descriptors", "fingerprints") if args.components == "both" else (args.components,)
    role = json.loads(args.role.read_text(encoding="utf-8"))
    priority = ()
    if role.get("dataset_identity") == "pcqm-fixed100k-v4":
        from molgap.pcqm_k1_scale import UNSANITIZED_OGB_SOURCE_INDICES
        members = set(role.get("source_indices", []))
        priority = tuple(i for i in UNSANITIZED_OGB_SOURCE_INDICES if i in members)
    result = build_cache(args.rows, args.rows_sha256, args.role, args.role_sha256, args.output,
                         components=components, parse_policy=args.parse_policy,
                         descriptor_missing_policy=args.descriptor_missing_policy,
                         fail_fast=not args.collect_all_failures, priority_source_indices=priority)
    print("ACCEPTED" if result["accepted"] else "LABEL_FAILURES_RETAINED")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
