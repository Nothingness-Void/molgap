"""Prepare hash-bound repaired-2M secondary ETKDG array inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.repaired_2m_3d_colab import prepare_secondary_array_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repaired-csv", type=Path, required=True)
    parser.add_argument("--primary-acceptance", type=Path, required=True)
    parser.add_argument("--primary-metadata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-shard", type=int, default=40)
    parser.add_argument("--stop-shard", type=int, default=100)
    parser.add_argument("--seed", type=int, default=314_159)
    args = parser.parse_args()
    result = prepare_secondary_array_inputs(
        repaired_csv=args.repaired_csv,
        primary_acceptance=args.primary_acceptance,
        primary_metadata_root=args.primary_metadata_root,
        output_dir=args.output_dir,
        start_shard=args.start_shard,
        stop_shard=args.stop_shard,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
