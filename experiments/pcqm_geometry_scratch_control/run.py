"""CLI for the numerical audit preceding the matched scratch control."""
import argparse
import json
from pathlib import Path

from molgap.pcqm_geometry_audit import run_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.graph_dir, args.acceptance, args.source,
                              args.source_config, args.reference, args.output_dir), indent=2), flush=True)


if __name__ == "__main__":
    main()
