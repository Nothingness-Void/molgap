"""Create the single-file Kaggle upload bundle for one repair-v2 fetch round."""
from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--round-index", type=int, required=True)
    ap.add_argument("--checkpoint-dataset", default=None,
                    help="Optional Kaggle dataset slug containing completed prior-round CSVs")
    args = ap.parse_args()
    if not 1 <= args.round_index <= 10:
        raise ValueError("--round-index must be in [1, 10]")

    source = args.source_dir
    out = args.out_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    runner = (source / "repair_v2_fetch_parallel.py").read_text(encoding="utf-8")
    replacements = {
        "__FETCHER_PAYLOAD__": base64.b64encode((source / "fetch_repair_candidates.py").read_bytes()).decode("ascii"),
        "__SPEC_PAYLOAD__": base64.b64encode((source / "repair_1m_v2_sampling_spec.json").read_bytes()).decode("ascii"),
    }
    for marker, payload in replacements.items():
        runner = runner.replace(marker, payload)
    runner, substitutions = re.subn(r"^ROUND_INDEX = \d+$", f"ROUND_INDEX = {args.round_index}", runner,
                                    count=1, flags=re.MULTILINE)
    if substitutions != 1 or any(marker in runner for marker in replacements):
        raise RuntimeError("Kaggle payload injection failed")
    (out / "repair_v2_fetch_parallel.py").write_text(runner, encoding="utf-8")

    metadata = json.loads((source / "kernel-metadata.json").read_text(encoding="utf-8"))
    if args.checkpoint_dataset:
        datasets = metadata.setdefault("dataset_sources", [])
        if args.checkpoint_dataset not in datasets:
            datasets.append(args.checkpoint_dataset)
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared Kaggle round {args.round_index:02d}: {out}")


if __name__ == "__main__":
    main()
