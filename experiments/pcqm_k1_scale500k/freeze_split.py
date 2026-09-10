from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.pcqm_k1_scale import write_scale_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-split", required=True, type=Path)
    parser.add_argument("--shadow-split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = write_scale_split(args.base_split, args.shadow_split, args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in {"train", "added_train", "validation"}}, indent=2))


if __name__ == "__main__":
    main()
