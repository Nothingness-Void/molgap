"""Add a validated round to the local directory uploaded as the Kaggle checkpoint dataset."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=Path, required=True)
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--exclude-csv", type=Path, action="append", default=[])
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--dataset-id", required=True,
                    help="Private Kaggle dataset slug, for example user/molgap-phase8-repair-v2-checkpoints")
    args = ap.parse_args()

    validation = args.round_dir / "checkpoint_validation.json"
    validation_command = [
        sys.executable, str(Path(__file__).with_name("validate_round.py")),
        "--round-dir", str(args.round_dir), "--train-csv", str(args.train_csv), "--out-json", str(validation),
    ]
    for path in args.exclude_csv:
        validation_command.extend(("--exclude-csv", str(path)))
    subprocess.run(validation_command, check=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for path in args.round_dir.glob("phase8_repair_v2_round*.csv"):
        shutil.copy2(path, args.checkpoint_dir / path.name)
    for path in (args.round_dir / "repair_1m_v2_round_manifest.json", validation):
        shutil.copy2(path, args.checkpoint_dir / f"round_{args.round_dir.name}_{path.name}")
    metadata_path = args.checkpoint_dir / "dataset-metadata.json"
    metadata_path.write_text(json.dumps({
        "title": "MolGap Phase8 Repair V2 Checkpoints",
        "id": args.dataset_id,
        "licenses": [{"name": "other"}],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Checkpoint dataset staged: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
