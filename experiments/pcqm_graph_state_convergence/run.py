"""Thin CLI for the GraphState full-data convergence continuation."""
import argparse
import json
from pathlib import Path

from molgap.pcqm_graph_state_continuation import preflight, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "train"))
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    if args.mode == "preflight":
        result = preflight(args.graph_dir, args.acceptance, args.source, args.output)
    else:
        if args.preflight is None:
            parser.error("--preflight is required for training")
        result = train(args.graph_dir, args.acceptance, args.source, args.preflight, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
