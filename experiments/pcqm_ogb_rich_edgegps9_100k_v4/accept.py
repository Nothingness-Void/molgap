from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_ogb_rich_gps_v4 import validate_completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_completion(args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
