"""Thin entrypoint for the preflight-gated matched scratch experiment."""
import argparse
import json
from pathlib import Path

from molgap.pcqm_geometry_scratch import preflight, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "train"])
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--arm", choices=["triangle", "geometry"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        if args.audit_dir is None:
            parser.error("preflight requires --audit-dir")
        result = preflight(args.graph_dir, args.acceptance, args.audit_dir, args.output)
    else:
        if args.preflight is None or args.arm is None:
            parser.error("train requires --preflight and --arm")
        result = train(args.graph_dir, args.acceptance, args.preflight, args.output, args.arm)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
