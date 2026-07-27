"""Create the single-file Kaggle bundle for the overnight general fetch."""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--kernel-title", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--checkpoint-dataset", action="append", default=[])
    args = parser.parse_args()

    source = args.source_dir
    out = args.out_dir
    if out.exists():
        raise FileExistsError(f"Refusing to replace existing package directory: {out}")
    out.mkdir(parents=True)

    runner = (source / "fetch_general_overnight.py").read_text(encoding="utf-8")
    replacements = {
        "__FETCHER_PAYLOAD__": base64.b64encode(
            (source / "fetch_repair_candidates.py").read_bytes()
        ).decode("ascii"),
        "__SPEC_PAYLOAD__": base64.b64encode(
            (source / "sampling_spec_2m.json").read_bytes()
        ).decode("ascii"),
    }
    for marker, payload in replacements.items():
        runner = runner.replace(marker, payload)
    runner, tag_substitutions = re.subn(
        r'^RUN_TAG = "[^"]+"$',
        f'RUN_TAG = "{args.run_tag}"',
        runner,
        count=1,
        flags=re.MULTILINE,
    )
    runner, seed_substitutions = re.subn(
        r"^SEED_BASE = [0-9_]+$",
        f"SEED_BASE = {args.seed_base}",
        runner,
        count=1,
        flags=re.MULTILINE,
    )
    if (
        any(marker in runner for marker in replacements)
        or tag_substitutions != 1
        or seed_substitutions != 1
    ):
        raise RuntimeError("Kaggle payload injection failed")
    (out / "fetch_general_overnight.py").write_text(runner, encoding="utf-8")

    metadata = json.loads((source / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata["id"] = args.kernel_id
    metadata["title"] = args.kernel_title
    metadata["code_file"] = "fetch_general_overnight.py"
    datasets = metadata.setdefault("dataset_sources", [])
    for checkpoint in args.checkpoint_dataset:
        if checkpoint not in datasets:
            datasets.append(checkpoint)
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared overnight general fetch: {out}")


if __name__ == "__main__":
    main()
