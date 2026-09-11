from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_fixed_datasets import (
    accept_fixed_pcqm_views,
    build_fixed_pcqm_views,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--row-manifest", type=Path, required=True)
    parser.add_argument("--topology-acceptance", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--geometry-acceptance", type=Path, required=True)
    parser.add_argument("--verify-content", action="store_true")
    args = parser.parse_args()
    result = build_fixed_pcqm_views(
        boundary=args.boundary,
        output_root=args.output_root,
        archive=args.archive,
        row_manifest_path=args.row_manifest,
        topology_acceptance_path=args.topology_acceptance,
        geometry_root=args.geometry_root,
        geometry_acceptance_path=args.geometry_acceptance,
    )
    acceptance = accept_fixed_pcqm_views(
        args.output_root, verify_content=args.verify_content
    )
    print(json.dumps({"manifest": result, "acceptance": acceptance}, indent=2))


if __name__ == "__main__":
    main()
