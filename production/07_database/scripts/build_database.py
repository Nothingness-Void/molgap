"""Build an auditable B3LYP prediction ledger from an input CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.database import DEFAULT_MODEL_KEY, run_database_build


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input CSV")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for predictions.csv and manifest.json",
    )
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace predictions.csv and manifest.json if they already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_database_build(
        args.input,
        args.out_dir,
        model_key=args.model_key,
        smiles_column=args.smiles_column,
        id_column=args.id_column,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
