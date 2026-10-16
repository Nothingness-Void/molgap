"""Thin CPU adapter; the shared data owner implements selection and acceptance."""
import argparse
import json
from pathlib import Path

from molgap.pcqm_unified_roles import prepare_unified_roles
from molgap.research_memory.trace import atomic_write, json_bytes


def main():
    parser=argparse.ArgumentParser()
    for name in ("boundary", "fixed-root", "output-root", "rule"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--role-inventory",type=Path)
    args=parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("CPU preparation is limited to four workers")
    try:
        result=prepare_unified_roles(boundary=args.boundary,fixed_root=args.fixed_root,
            output_root=args.output_root,rule=json.loads(args.rule.read_text()),workers=args.workers,
            role_inventory=args.role_inventory)
        print(json.dumps(result,indent=2))
    except Exception as exc:
        # The module validates output confinement before writing anything.
        boundary=args.boundary.resolve(); output=args.output_root.resolve()
        if output != boundary and output.is_relative_to(boundary) and not output.is_relative_to(args.fixed_root.resolve()):
            atomic_write(output/"failure.json",json_bytes({"status":"ROLE_BLOCKED","error_type":type(exc).__name__,
                "reason":str(exc),"gpu_used":False,"training_submitted":False}))
        raise


if __name__ == "__main__":
    main()
