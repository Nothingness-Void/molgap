"""Frozen server GraphState winner: A100 gate then one scratch run."""
import argparse
import json
from pathlib import Path

from molgap.pcqm_geometry_scratch import preflight, representative_preflight, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "train"])
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representative", action="store_true")
    args = parser.parse_args()
    if args.mode == "preflight":
        if args.audit_dir is None:
            parser.error("--audit-dir is required")
        if args.representative:
            result = representative_preflight(args.graph_dir, args.acceptance, args.audit_dir, args.output)
        else:
            result = preflight(args.graph_dir, args.acceptance, args.audit_dir, args.output, graphstate=True)
    else:
        if args.preflight is None:
            parser.error("--preflight is required")
        result = train(args.graph_dir, args.acceptance, args.preflight, args.output, "graphstate")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
