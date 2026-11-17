"""Convert one accepted framework-neutral secondary shard to PyG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.repaired_2m_3d_colab import convert_secondary_raw_shard_to_pyg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = convert_secondary_raw_shard_to_pyg(
        raw_path=args.raw_path,
        raw_report_path=args.raw_report,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
